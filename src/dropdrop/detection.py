"""Main droplet and inclusion detection pipeline."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .cache import Cache
from .config import load_config
from . import scanprotocol


class Detection:
    """Main pipeline for droplet and inclusion detection."""

    def __init__(self, config=None, store_visualizations=False, use_cache=True,
                 detect_inclusions=True):
        """Initialize pipeline with configuration.

        Args:
            config: Configuration dict. If None, loads from config.json.
            store_visualizations: Whether to store visualization data for UI.
            use_cache: Whether to use caching for expensive computations.
            detect_inclusions: Whether to detect inclusions inside droplets.
        """
        self.config = config if config else load_config()
        self.results_data = []
        self.store_visualizations = store_visualizations
        self.visualization_data = {} if store_visualizations else None
        self.use_cache = use_cache
        self.cache = Cache(self.config) if use_cache else None
        self.detect_inclusions = detect_inclusions
        self._cellpose_model = None
        self._frame_output = None  # tmp dir where per-frame overlays stream

    def parse_filename(self, filename):
        """Extract z-stack index and frame index from filename.

        Files without z-index are treated as single images (z_index=0).
        """
        z_match = re.search(r"_z(\d+)_", filename)
        z_index = int(z_match.group(1)) if z_match else 0

        f_match = re.search(r"a01f(\d+)d4", filename, re.IGNORECASE)
        frame_index = int(f_match.group(1)) if f_match else None

        return z_index, frame_index

    def load_and_group_images(self, input_dir):
        """Load images and group by frame index."""
        input_path = Path(input_dir)

        # Find all image files
        extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        # Group by frame
        frame_groups = defaultdict(list)
        for filepath in image_files:
            z_idx, frame_idx = self.parse_filename(filepath.name)
            if frame_idx is not None:
                frame_groups[frame_idx].append((z_idx, filepath))

        # Sort z-stacks within each frame
        for frame_idx in frame_groups:
            frame_groups[frame_idx].sort(key=lambda x: x[0])

        return frame_groups

    def create_min_projection(self, z_stack_files):
        """Create minimum intensity projection with CLAHE preprocessing."""
        images = []
        for z_idx, filepath in z_stack_files:
            img = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Convert to 8-bit first
                if img.dtype == np.uint16:
                    img = img.astype(np.float32) * 64
                    img = np.clip(img, 0, 65535)
                    img = (img / 256).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)

                images.append(img)

        if not images:
            return None

        # Create min projection
        stack = np.stack(images, axis=0)
        min_proj = np.min(stack, axis=0).astype(np.uint8)

        # Apply CLAHE to normalize local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        min_proj = clahe.apply(min_proj)

        return min_proj

    def detect_droplets_cellpose(self, image):
        """Detect droplets using Cellpose."""
        # Lazy import and model caching
        if self._cellpose_model is None:
            try:
                from cellpose.core import use_gpu
                from cellpose.models import CellposeModel
            except ImportError:
                print("ERROR: Cellpose is required for droplet detection.")
                print("Install with: pip install cellpose")
                sys.exit(1)

            if not use_gpu():
                import platform
                os_name = platform.system()
                if os_name in ("Linux", "Windows"):
                    print("WARNING: CUDA is not available. Cellpose will run on CPU (slow).")
                    print("Install CUDA-enabled PyTorch:")
                    print("  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")
                else:
                    print("WARNING: No GPU detected. Cellpose will run on CPU (slow).")

            self._cellpose_model = CellposeModel(gpu=True)

        masks, flows, styles = self._cellpose_model.eval(
            image,
            normalize=True,
            flow_threshold=self.config["cellpose_flow_threshold"],
            cellprob_threshold=self.config["cellpose_cellprob_threshold"],
        )

        return self.masks_to_coordinates(masks)

    def masks_to_coordinates(self, masks):
        """Convert Cellpose masks to coordinate format."""
        coordinate_list = []

        # Get unique mask IDs (excluding 0 for background)
        unique_ids = np.unique(masks)[1:]

        for mask_id in unique_ids:
            binary_mask = (masks == mask_id).astype(np.uint8)
            coords = self.mask_to_coordinates(binary_mask)
            if coords is not None:
                coordinate_list.append(coords)

        return coordinate_list

    def mask_to_coordinates(self, binary_mask):
        """Convert single binary mask to coordinates."""
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        coords = []
        for point in contour:
            coords.extend([str(point[0][0]), str(point[0][1])])

        return ",".join(coords)

    def coordinates_to_mask(self, coord_string, image_shape):
        """Convert coordinate string back to binary mask."""
        coords = [float(x) for x in coord_string.split(",")]
        points = np.array(coords).reshape(-1, 2).astype(np.int32)

        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        return mask

    def erode_mask(self, mask, erosion_pixels):
        """Erode mask by specified number of pixels."""
        if erosion_pixels <= 0:
            return mask

        kernel_size = 2 * erosion_pixels + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        return cv2.erode(mask, kernel, iterations=1)

    def detect_inclusions_in_droplet(self, image, droplet_mask, store_masked=False):
        """Detect inclusions within a single droplet using black-hat morphology."""
        masked_image = cv2.bitwise_and(image, image, mask=droplet_mask)

        kernel_size = self.config["kernel_size"]
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        blackhat = cv2.morphologyEx(masked_image, cv2.MORPH_BLACKHAT, kernel)

        _, inclusions = cv2.threshold(
            blackhat, self.config["tophat_threshold"], 255, cv2.THRESH_BINARY
        )

        inclusions = cv2.bitwise_and(inclusions, inclusions, mask=droplet_mask)
        filtered_inclusions, count = self.filter_inclusions_by_size(inclusions)

        if store_masked:
            return filtered_inclusions, count, blackhat
        return filtered_inclusions, count

    def filter_inclusions_by_size(self, inclusion_mask):
        """Filter detected inclusions by size constraints and edge proximity."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            inclusion_mask, connectivity=8
        )

        h, w = inclusion_mask.shape
        edge_buffer = self.config.get("edge_buffer", 5)

        filtered_mask = np.zeros_like(inclusion_mask)
        inclusion_count = 0

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w_comp = stats[label, cv2.CC_STAT_WIDTH]
            h_comp = stats[label, cv2.CC_STAT_HEIGHT]

            # Edge check
            if (
                x < edge_buffer
                or y < edge_buffer
                or x + w_comp > w - edge_buffer
                or y + h_comp > h - edge_buffer
            ):
                continue

            # Size check
            if (
                self.config["min_inclusion_area"]
                <= area
                <= self.config["max_inclusion_area"]
            ):
                filtered_mask[labels == label] = 255
                inclusion_count += 1

        return filtered_mask, inclusion_count

    def process_frame(self, frame_idx, min_projection, droplet_coords=None):
        """Process a single frame for droplets and inclusions.

        Streams a per-frame overlay PNG to the tmp dir (when configured) so the
        report can tile every frame, without buffering masks across frames.
        """
        store_viz = self.store_visualizations
        save_overlay = self._frame_output is not None
        overlay = (
            cv2.cvtColor(min_projection, cv2.COLOR_GRAY2BGR) if save_overlay else None
        )

        if store_viz:
            frame_viz = {
                "min_projection": min_projection,
                "droplet_masks": [],
                "eroded_masks": [],
                "inclusion_masks": [],
                "masked_images": [],
            }

        if droplet_coords is None:
            droplet_coords = self.detect_droplets_cellpose(min_projection)

        if not droplet_coords:
            print(f"  Frame {frame_idx}: No droplets detected")
            if store_viz:
                self.visualization_data[frame_idx] = frame_viz
            if save_overlay:
                self._save_overlay(frame_idx, overlay)
            return

        valid_droplet_idx = 0
        for coords in droplet_coords:
            droplet_mask = self.coordinates_to_mask(coords, min_projection.shape)

            contours, _ = cv2.findContours(
                droplet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            M = cv2.moments(contours[0])
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area = cv2.contourArea(contours[0])
            diameter = np.sqrt(4 * area / np.pi)

            if not (
                self.config["min_droplet_diameter"]
                <= diameter
                <= self.config["max_droplet_diameter"]
            ):
                continue

            eroded_mask = self.erode_mask(droplet_mask, self.config["erosion_pixels"])

            if np.sum(eroded_mask) == 0:
                continue

            inclusion_count = 0
            inclusion_mask = np.zeros_like(droplet_mask)

            if self.detect_inclusions:
                if store_viz:
                    inclusion_mask, inclusion_count, blackhat = (
                        self.detect_inclusions_in_droplet(
                            min_projection, eroded_mask, store_masked=True
                        )
                    )
                    frame_viz["masked_images"].append(blackhat)
                else:
                    inclusion_mask, inclusion_count = self.detect_inclusions_in_droplet(
                        min_projection, eroded_mask
                    )

            if save_overlay:
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
                if self.detect_inclusions:
                    overlay[inclusion_mask > 0] = (0, 0, 255)
                    cv2.putText(overlay, str(inclusion_count), (cx - 5, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            if store_viz:
                frame_viz["droplet_masks"].append({
                    "mask": droplet_mask,
                    "center": (cx, cy),
                    "radius": diameter / 2,
                    "inclusions": inclusion_count,
                })
                frame_viz["eroded_masks"].append(eroded_mask)
                frame_viz["inclusion_masks"].append(inclusion_mask)

            self.results_data.append({
                "frame": frame_idx,
                "droplet_id": valid_droplet_idx,
                "center_x": cx,
                "center_y": cy,
                "diameter_px": diameter,
                "diameter_um": diameter * self.config["px_to_um"],
                "area_px": area,
                "area_um2": area * (self.config["px_to_um"] ** 2),
                "inclusions": inclusion_count,
            })

            valid_droplet_idx += 1

        if store_viz:
            self.visualization_data[frame_idx] = frame_viz
        if save_overlay:
            self._save_overlay(frame_idx, overlay)

    def _save_overlay(self, frame_idx, overlay):
        """Write a single frame overlay PNG into the tmp output dir."""
        cv2.imwrite(str(self._frame_output / f"frame_{frame_idx}.png"), overlay)

        frame_data = [d for d in self.results_data if d["frame"] == frame_idx]
        total_inclusions = sum(d["inclusions"] for d in frame_data)
        print(
            f"  Frame {frame_idx}: {len(frame_data)} valid droplets, "
            f"{total_inclusions} total inclusions"
        )

    def run(self, input_dir, output_dir, frame_limit=None):
        """Run the complete pipeline."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\nLoading and grouping images...")
        frame_groups = self.load_and_group_images(input_dir)

        if not frame_groups:
            print("ERROR: No valid images found!")
            return None

        frame_indices = sorted(frame_groups.keys())
        if frame_limit and frame_limit > 0:
            frame_indices = frame_indices[:frame_limit]
            print(f"Processing limited to first {frame_limit} frames")

        print(
            f"Found {len(frame_groups)} frames total, processing {len(frame_indices)} frames\n"
        )

        # Stream a per-frame overlay PNG into the output dir for the tiled report
        self._frame_output = output_path

        cache_hits = 0
        for frame_idx in tqdm(frame_indices, desc="Processing frames"):
            z_stack_files = frame_groups[frame_idx]
            cache_key_file = z_stack_files[0][1].name if z_stack_files else None

            if self.cache and cache_key_file and self.cache.is_valid(cache_key_file):
                cached_data = self.cache.load_frame(cache_key_file)
                min_proj = cached_data["min_projection"]
                droplet_coords = cached_data["droplet_coords"]
                cache_hits += 1
                self.process_frame(frame_idx, min_proj, droplet_coords)
            else:
                min_proj = self.create_min_projection(z_stack_files)

                if min_proj is None:
                    continue

                droplet_coords = self.detect_droplets_cellpose(min_proj)

                if self.cache and cache_key_file:
                    self.cache.save_frame(cache_key_file, min_proj, droplet_coords)

                self.process_frame(frame_idx, min_proj, droplet_coords)

        if cache_hits > 0:
            print(f"\nCache: {cache_hits}/{len(frame_indices)} frames loaded from cache")

        # Persist the field tiling layout from the scanprotocol, if available
        self._write_layout(input_dir, output_path, frame_indices)

        if self.results_data:
            df = pd.DataFrame(self.results_data)
            csv_path = output_path / "data.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")
            self.print_summary(df)
        else:
            print("\nNo droplets detected in any frame!")

        return self.results_data

    def _write_layout(self, input_dir, output_path, frame_indices):
        """Write layout.json describing the field grid for the tiled report."""
        layout = scanprotocol.build_layout(input_dir, len(frame_indices))
        if layout is None:
            return
        layout["frames"] = list(frame_indices)
        with open(output_path / "layout.json", "w") as f:
            json.dump(layout, f)
        print(
            f"Tile layout: {layout['cols']}x{layout['rows']} "
            f"({layout['pattern']})"
        )

    def print_summary(self, df):
        """Print one-line summary."""
        if self.detect_inclusions:
            print(
                f"\nDetected {len(df)} droplets with {df['inclusions'].sum()} inclusions "
                f"({df['inclusions'].mean():.2f} per droplet)"
            )
        else:
            print(f"\nDetected {len(df)} droplets (inclusion detection OFF)")
