#!/usr/bin/env python3
"""
Droplet and Inclusion Detection Pipeline using cellpose
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys
import argparse
from tqdm import tqdm
from scipy import stats
import re
from datetime import datetime
import json
import hashlib
import shutil
import tarfile
import seaborn as sns
import matplotlib.pyplot as plt

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300

# Required: Cellpose
try:
    from cellpose.models import CellposeModel
except ImportError:
    print("You need to have cellpose for this pipeline to work!")
    sys.exit(1)


# region CacheManager
class CacheManager:
    """Global LRU cache for expensive computations, stored in project root."""

    def __init__(self, config):
        cache_cfg = config.get("cache", {})
        self.enabled = cache_cfg.get("enabled", True)
        self.max_frames = cache_cfg.get("max_frames", 100)
        # Cache in project root (where main.py is located)
        self.cache_dir = Path(__file__).parent / ".cache"
        self.metadata_path = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
        self.config = config

    def _load_metadata(self):
        """Load cache metadata from disk."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return self._default_metadata()
        return self._default_metadata()

    def _default_metadata(self):
        """Return default metadata structure."""
        return {"version": "1.0", "config_hash": None, "frames": {}, "access_order": []}

    def _save_metadata(self):
        """Save cache metadata to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _enforce_lru(self):
        """Remove oldest frames if over max_frames limit."""
        while len(self.metadata["access_order"]) > self.max_frames:
            oldest_key = self.metadata["access_order"].pop(0)
            cache_file = self.cache_dir / f"{oldest_key}.npz"
            if cache_file.exists():
                cache_file.unlink()
            self.metadata["frames"].pop(oldest_key, None)

    def get_config_hash(self):
        """Hash detection-related config keys that affect caching."""
        keys = [
            "cellpose_flow_threshold",
            "cellpose_cellprob_threshold",
            "min_droplet_diameter",
            "max_droplet_diameter",
        ]
        data = {k: self.config.get(k) for k in keys}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    def _get_cache_key(self, source_filename):
        """Generate cache key from source filename (not full path)."""
        name = Path(source_filename).stem
        return hashlib.sha256(name.encode()).hexdigest()[:16]

    def is_valid(self, source_filename):
        """Check if cache is valid for frame by source filename."""
        if not self.enabled:
            return False
        current_hash = self.get_config_hash()
        if self.metadata.get("config_hash") != current_hash:
            return False
        cache_key = self._get_cache_key(source_filename)
        cache_file = self.cache_dir / f"{cache_key}.npz"
        return cache_file.exists()

    def load_frame(self, source_filename):
        """Load cached data by source filename and update access order."""
        cache_key = self._get_cache_key(source_filename)
        cache_file = self.cache_dir / f"{cache_key}.npz"
        data = np.load(cache_file, allow_pickle=True)

        # Update LRU order
        if cache_key in self.metadata["access_order"]:
            self.metadata["access_order"].remove(cache_key)
        self.metadata["access_order"].append(cache_key)
        self._save_metadata()

        return {
            "min_projection": data["min_projection"],
            "droplet_coords": list(data["droplet_coords"]),
        }

    def save_frame(self, source_filename, min_proj, droplet_coords):
        """Save frame data by source filename and enforce LRU limit."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._get_cache_key(source_filename)
        cache_file = self.cache_dir / f"{cache_key}.npz"

        np.savez(
            cache_file,
            min_projection=min_proj,
            droplet_coords=np.array(droplet_coords, dtype=object),
        )

        # Update metadata
        self.metadata["config_hash"] = self.get_config_hash()
        self.metadata["frames"][cache_key] = {
            "source": str(source_filename),
            "cached_at": datetime.now().isoformat(),
        }

        # Update LRU order
        if cache_key in self.metadata["access_order"]:
            self.metadata["access_order"].remove(cache_key)
        self.metadata["access_order"].append(cache_key)

        self._enforce_lru()
        self._save_metadata()

    def clear(self):
        """Clear entire cache."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.metadata = self._default_metadata()
        print("Cache cleared.")


# endregion


# region Settings and CLI helpers
def parse_settings(settings_str):
    """Parse compact settings string.

    Format: key=value,key=value
    Keys: d[ilution], p[oisson], c[ount], l[abel]

    Examples:
        "d=1000,p=on,c=6.5e5,l=experiment1"
        "dilution=500,poisson=off"
    """
    settings = {
        "dilution": 500,
        "poisson": True,
        "count": 6.5e5,
        "label": None,
    }

    if not settings_str:
        return settings

    key_map = {
        "d": "dilution",
        "dilution": "dilution",
        "p": "poisson",
        "poisson": "poisson",
        "c": "count",
        "count": "count",
        "l": "label",
        "label": "label",
    }

    for part in settings_str.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key_map.get(key.strip().lower(), key.strip().lower())

        if key == "dilution":
            settings["dilution"] = int(value)
        elif key == "poisson":
            settings["poisson"] = value.lower() in ("on", "yes", "true", "1")
        elif key == "count":
            settings["count"] = float(value)
        elif key == "label":
            settings["label"] = value.strip()

    return settings


def prompt_settings():
    """Interactive prompts for settings when --settings not provided."""
    settings = {"dilution": 500, "poisson": True, "count": 6.5e5, "label": None}

    print("\n--- Project Settings ---")

    # Poisson analysis
    use_poisson = input("Use Poisson analysis? [yes/no] (yes): ").strip().lower()
    settings["poisson"] = use_poisson != "no"

    if settings["poisson"]:
        # Bead count
        count_input = input("Stock count/uL [6.5e5]: ").strip()
        if count_input:
            try:
                settings["count"] = float(count_input)
            except ValueError:
                print(f"  Invalid value, using default: {settings['count']}")

        # Dilution
        dilution_input = input("Dilution factor [500]: ").strip()
        if dilution_input:
            try:
                settings["dilution"] = int(dilution_input)
            except ValueError:
                print(f"  Invalid value, using default: {settings['dilution']}")

    # Label
    label_input = input("Project label (optional, press Enter to skip): ").strip()
    settings["label"] = label_input if label_input else None

    print("------------------------\n")
    return settings


def generate_project_name(settings):
    """Generate project directory name from date and label."""
    date_str = datetime.now().strftime("%Y%m%d")
    if settings.get("label"):
        return f"{date_str}_{settings['label']}"
    return date_str


# endregion


class DropletInclusionPipeline:
    """Main pipeline for droplet and inclusion detection."""

    def __init__(self, store_visualizations=False, use_cache=True):
        """Initialize pipeline with configuration."""
        self.config = self.load_config()
        self.results_data = []
        self.store_visualizations = store_visualizations
        self.visualization_data = {} if store_visualizations else None
        self.use_cache = use_cache
        self.cache = CacheManager(self.config) if use_cache else None

    def load_config(self):
        """Load configuration from root config.json or use defaults."""
        default_config = {
            # Cellpose parameters
            "cellpose_flow_threshold": 0.4,
            "cellpose_cellprob_threshold": 0.0,
            # Erosion parameters
            "erosion_pixels": 5,
            # Inclusion detection parameters
            "kernel_size": 7,
            "tophat_threshold": 30,
            "min_inclusion_area": 7,
            "max_inclusion_area": 50,
            "edge_buffer": 5,
            # Droplet filtering
            "min_droplet_diameter": 80,
            "max_droplet_diameter": 200,
            # Conversion factor
            "px_to_um": 1.14,
        }

        config_path = Path("config.json")
        if config_path.exists():
            print(f"Loading config from: {config_path}")
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                # Merge with defaults (loaded config overrides defaults)
                default_config.update(loaded_config)
                return default_config
        else:
            print("Using default configuration (no config.json found)")
            return default_config

    def parse_filename(self, filename):
        """Extract z-stack index and frame index from filename.

        Files without z-index are treated as single images (z_index=0).
        """
        z_match = re.search(r"_z(\d+)_", filename)
        z_index = int(z_match.group(1)) if z_match else 0  # Default to 0 for non-z-stack

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
                    # Apply multiplication in original bit depth
                    img = img.astype(np.float32) * 64
                    # Now convert to 8-bit based on the data type maximum
                    img = np.clip(img, 0, 65535)  # Clip to 16-bit max
                    img = (img / 256).astype(np.uint8)
                else:
                    # Already in 8-bit range
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
        model = CellposeModel(gpu=True)

        masks, flows, styles = model.eval(
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
            # Create binary mask for this droplet
            binary_mask = (masks == mask_id).astype(np.uint8)
            coords = self.mask_to_coordinates(binary_mask)
            if coords is not None:
                coordinate_list.append(coords)

        return coordinate_list

    def mask_to_coordinates(self, binary_mask):
        """Convert single binary mask to coordinates."""
        # Find contours
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Get largest contour
        contour = max(contours, key=cv2.contourArea)

        # Convert to coordinate string format
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

        # Create circular kernel for erosion
        kernel_size = 2 * erosion_pixels + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        # Apply erosion
        eroded = cv2.erode(mask, kernel, iterations=1)

        return eroded

    def detect_inclusions_in_droplet(self, image, droplet_mask, store_masked=False):
        """Detect inclusions within a single droplet using top-hat."""

        # Create masked image - apply mask to original image
        masked_image = cv2.bitwise_and(image, image, mask=droplet_mask)

        # Apply morphological black-hat (top-hat for dark features) to masked region only
        kernel_size = self.config["kernel_size"]
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        # Apply black-hat transform to masked image
        blackhat = cv2.morphologyEx(masked_image, cv2.MORPH_BLACKHAT, kernel)

        # Threshold to identify dark spots
        _, inclusions = cv2.threshold(
            blackhat, self.config["tophat_threshold"], 255, cv2.THRESH_BINARY
        )

        # Ensure we only count inclusions within the mask area
        inclusions = cv2.bitwise_and(inclusions, inclusions, mask=droplet_mask)

        # Filter by size and edge proximity
        filtered_inclusions, count = self.filter_inclusions_by_size(inclusions)

        if store_masked:
            return filtered_inclusions, count, blackhat
        return filtered_inclusions, count

    def filter_inclusions_by_size(self, inclusion_mask):
        """Filter detected inclusions by size constraints and edge proximity."""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            inclusion_mask, connectivity=8
        )

        # Get image dimensions for edge filtering
        h, w = inclusion_mask.shape
        edge_buffer = self.config.get("edge_buffer", 5)

        # Create filtered mask
        filtered_mask = np.zeros_like(inclusion_mask)
        inclusion_count = 0

        # Skip background (label 0)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]

            # Get bounding box for edge detection
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w_comp = stats[label, cv2.CC_STAT_WIDTH]
            h_comp = stats[label, cv2.CC_STAT_HEIGHT]

            # Check if inclusion is too close to image edge
            if (
                x < edge_buffer
                or y < edge_buffer
                or x + w_comp > w - edge_buffer
                or y + h_comp > h - edge_buffer
            ):
                continue

            # Check size constraints
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

        Args:
            frame_idx: Frame index for results tracking
            min_projection: Min intensity projection image
            droplet_coords: Optional pre-computed droplet coordinates (from cache)
        """
        # Initialize frame visualization data if needed
        if self.store_visualizations:
            frame_viz = {
                "min_projection": min_projection,
                "droplet_masks": [],
                "eroded_masks": [],
                "inclusion_masks": [],
                "masked_images": [],
            }

        # Detect droplets if not provided (cache miss or no cache)
        if droplet_coords is None:
            droplet_coords = self.detect_droplets_cellpose(min_projection)

        if not droplet_coords:
            print(f"  Frame {frame_idx}: No droplets detected")
            if self.store_visualizations:
                self.visualization_data[frame_idx] = frame_viz
            return

        # Process each droplet
        valid_droplet_idx = 0
        for coords in droplet_coords:
            # Convert coordinates to mask
            droplet_mask = self.coordinates_to_mask(coords, min_projection.shape)

            # Calculate droplet properties
            contours, _ = cv2.findContours(
                droplet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            # Get properties
            M = cv2.moments(contours[0])
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area = cv2.contourArea(contours[0])
            diameter = np.sqrt(4 * area / np.pi)

            # Skip if outside size range
            if not (
                self.config["min_droplet_diameter"]
                <= diameter
                <= self.config["max_droplet_diameter"]
            ):
                continue

            eroded_mask = self.erode_mask(droplet_mask, self.config["erosion_pixels"])

            if np.sum(eroded_mask) == 0:
                continue

            # Detect inclusions in eroded area
            if self.store_visualizations:
                inclusion_mask, inclusion_count, blackhat = (
                    self.detect_inclusions_in_droplet(
                        min_projection, eroded_mask, store_masked=True
                    )
                )
                frame_viz["masked_images"].append(
                    blackhat
                )  # Store blackhat of masked region
            else:
                inclusion_mask, inclusion_count = self.detect_inclusions_in_droplet(
                    min_projection, eroded_mask
                )

            # Store visualization data if needed (only for valid droplets)
            if self.store_visualizations:
                frame_viz["droplet_masks"].append(
                    {
                        "mask": droplet_mask,
                        "center": (cx, cy),
                        "radius": diameter / 2,
                        "inclusions": inclusion_count,
                    }
                )
                frame_viz["eroded_masks"].append(eroded_mask)
                frame_viz["inclusion_masks"].append(inclusion_mask)

            # Add to results data (only valid droplets that survived erosion)
            self.results_data.append(
                {
                    "frame": frame_idx,
                    "droplet_id": valid_droplet_idx,
                    "center_x": cx,
                    "center_y": cy,
                    "diameter_px": diameter,
                    "diameter_um": diameter * self.config["px_to_um"],
                    "area_px": area,
                    "area_um2": area * (self.config["px_to_um"] ** 2),
                    "inclusions": inclusion_count,
                }
            )

            valid_droplet_idx += 1  # Increment only for valid droplets

        # Store frame visualization data
        if self.store_visualizations:
            self.visualization_data[frame_idx] = frame_viz

        # Print frame summary
        frame_data = [d for d in self.results_data if d["frame"] == frame_idx]
        total_inclusions = sum(d["inclusions"] for d in frame_data)
        print(
            f"  Frame {frame_idx}: {len(frame_data)} valid droplets (after erosion), {total_inclusions} total inclusions"
        )

    def run(self, input_dir, output_dir, frame_limit=None):
        """Run the complete pipeline."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load and group images
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

        # Process each frame
        cache_hits = 0
        for frame_idx in tqdm(frame_indices, desc="Processing frames"):
            z_stack_files = frame_groups[frame_idx]
            # Use first file in z-stack as cache key (they share the same frame)
            cache_key_file = z_stack_files[0][1].name if z_stack_files else None

            # Try to load from cache
            if self.cache and cache_key_file and self.cache.is_valid(cache_key_file):
                cached_data = self.cache.load_frame(cache_key_file)
                min_proj = cached_data["min_projection"]
                droplet_coords = cached_data["droplet_coords"]
                cache_hits += 1
                # Process with cached data
                self.process_frame(frame_idx, min_proj, droplet_coords)
            else:
                # Create min projection and detect droplets
                min_proj = self.create_min_projection(z_stack_files)

                if min_proj is None:
                    continue

                # Detect droplets (expensive operation)
                droplet_coords = self.detect_droplets_cellpose(min_proj)

                # Save to cache
                if self.cache and cache_key_file:
                    self.cache.save_frame(cache_key_file, min_proj, droplet_coords)

                # Process frame with freshly detected coords
                self.process_frame(frame_idx, min_proj, droplet_coords)

        if cache_hits > 0:
            print(f"\nCache: {cache_hits}/{len(frame_indices)} frames loaded from cache")

        # Save results to CSV
        if self.results_data:
            df = pd.DataFrame(self.results_data)
            csv_path = output_path / "data.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")

            # Print summary statistics
            self.print_summary(df)
        else:
            print("\nNo droplets detected in any frame!")

        return self.results_data

    def print_summary(self, df):
        """Print one-line summary."""
        print(
            f"\nDetected {len(df)} droplets with {df['inclusions'].sum()} inclusions "
            f"({df['inclusions'].mean():.2f} per droplet)"
        )


class BaseWindow:
    """Base class for all window-based interfaces."""

    def __init__(self, visualization_data):
        self.visualization_data = visualization_data
        self.frames = sorted(visualization_data.keys())
        self.current_index = 0
        self.window_name = "Window"

    def navigate(self):
        """Handle keyboard navigation - common for all windows."""
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:  # q or ESC
            return False
        elif key == 83 or key == ord(" "):  # Right arrow or space
            self.current_index = (self.current_index + 1) % len(self.frames)
        elif key == 81:  # Left arrow
            self.current_index = (self.current_index - 1) % len(self.frames)
        elif key == 13:  # Enter
            if self.current_index < len(self.frames) - 1:
                self.current_index += 1
            else:
                return False  # Exit on last frame

        return True

    def get_current_frame_data(self):
        """Get current frame visualization data."""
        return self.visualization_data[self.frames[self.current_index]]

    def run(self):
        """Main window loop - to be overridden."""
        raise NotImplementedError


class Viewer(BaseWindow):
    """Interactive viewer for detection results."""

    def __init__(self, visualization_data, results_df):
        super().__init__(visualization_data)
        self.df = results_df
        self.mode = "steps"
        self.window_name = "Droplet Detection Viewer"

    def create_overlay(self, frame_idx):
        """Create overlay visualization from stored data."""
        frame_data = self.visualization_data[frame_idx]
        min_proj = frame_data["min_projection"]

        # Convert to BGR
        overlay = cv2.cvtColor(min_proj, cv2.COLOR_GRAY2BGR)

        # Draw all droplets from stored masks
        for i, droplet_info in enumerate(frame_data["droplet_masks"]):
            cx, cy = droplet_info["center"]
            radius = int(droplet_info["radius"])
            inclusions = droplet_info["inclusions"]

            # Color based on inclusions
            color = (
                (0, 0, 255) if inclusions > 0 else (0, 255, 0)
            )  # Red if inclusions, green if not
            cv2.circle(overlay, (int(cx), int(cy)), radius, color, 2)

            # Draw erosion boundary
            eroded_radius = radius - self.df.iloc[0].get(
                "erosion_pixels", 10
            )  # Use config value
            if eroded_radius > 0:
                cv2.circle(overlay, (int(cx), int(cy)), eroded_radius, (0, 255, 255), 1)

            # Center point
            cv2.circle(overlay, (int(cx), int(cy)), 3, (255, 0, 0), -1)

            # Show inclusion count if > 0
            if inclusions > 0:
                cv2.putText(
                    overlay,
                    str(inclusions),
                    (int(cx) - 10, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        # Add statistics
        frame_df = self.df[self.df["frame"] == frame_idx]
        total_droplets = len(frame_df)
        total_inclusions = frame_df["inclusions"].sum()

        info_text = f"Frame {frame_idx} | Droplets: {total_droplets} | Inclusions: {int(total_inclusions)}"
        cv2.putText(
            overlay, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        return overlay

    def create_steps(self, frame_idx):
        """Create processing steps visualization from stored data."""
        frame_data = self.visualization_data[frame_idx]
        min_proj = frame_data["min_projection"]
        h, w = min_proj.shape

        images = []

        # 1. Min Projection
        min_bgr = cv2.cvtColor(min_proj, cv2.COLOR_GRAY2BGR)
        images.append(("Min Projection", min_bgr))

        # 2. Detected Droplets (Cellpose)
        droplet_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for i, mask in enumerate(frame_data["droplet_masks"]):
            droplet_mask = mask["mask"]
            color_val = (i * 30) % 200 + 55
            droplet_overlay[droplet_mask > 0] = [color_val, color_val, 0]
        images.append(("Cellpose Detection", droplet_overlay))

        # 3. Eroded Masks
        eroded_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for eroded_mask in frame_data["eroded_masks"]:
            eroded_overlay[eroded_mask > 0] = [0, 200, 200]  # Cyan for eroded areas
        images.append(("Eroded Masks", eroded_overlay))

        # 4. Black-hat on Masked Regions (combined view)
        if "masked_images" in frame_data and frame_data["masked_images"]:
            blackhat_combined = np.zeros((h, w), dtype=np.uint8)
            for masked_blackhat in frame_data["masked_images"]:
                blackhat_combined = cv2.bitwise_or(blackhat_combined, masked_blackhat)
            blackhat_bgr = cv2.cvtColor(blackhat_combined, cv2.COLOR_GRAY2BGR)
            images.append(("Black-hat (Masked)", blackhat_bgr))

        # 5. Detected Inclusions
        inclusion_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for inclusion_mask in frame_data["inclusion_masks"]:
            inclusion_overlay[:, :, 2] = cv2.bitwise_or(
                inclusion_overlay[:, :, 2], inclusion_mask
            )  # Red channel for inclusions
        images.append(("Detected Inclusions", inclusion_overlay))

        # 6. Final Result (overlay)
        final_overlay = self.create_overlay(frame_idx)
        images.append(("Final Result", final_overlay))

        # Create grid layout
        cols = 3
        rows = 2
        collage = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 240

        for idx, (title, img) in enumerate(images[:6]):  # Max 6 images in 2x3 grid
            row = idx // cols
            col = idx % cols

            # Ensure image is correct size
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))

            # Add title
            img_copy = img.copy()
            cv2.putText(
                img_copy, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

            # Place in grid
            collage[row * h : (row + 1) * h, col * w : (col + 1) * w] = img_copy

        # Add frame counter
        cv2.putText(
            collage,
            f"Frame {frame_idx}/{max(self.frames)}",
            (10, rows * h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        return collage

    def run(self):
        """Run viewer with mode switching."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_index = (self.current_index + 1) % len(self.frames)

        cv2.setMouseCallback(self.window_name, mouse_callback)

        while True:
            frame_idx = self.frames[self.current_index]

            if self.mode == "overlay":
                display_img = self.create_overlay(frame_idx)
            else:
                display_img = self.create_steps(frame_idx)

            cv2.imshow(self.window_name, display_img)

            # Check for mode switch
            key = cv2.waitKey(1) & 0xFF
            if key == ord("m"):
                self.mode = "overlay" if self.mode == "steps" else "steps"
                continue

            # Handle navigation
            if not self.navigate():
                break

        cv2.destroyAllWindows()


class InclusionEditor(BaseWindow):
    """Interactive editor for inclusion corrections."""

    def __init__(self, visualization_data, results_data):
        super().__init__(visualization_data)
        self.results_data = results_data
        self.window_name = "Inclusion Editor"
        self.inclusions = {}  # {frame_idx: [(x, y), ...]}
        self.right_mouse_down = False  # Track right mouse button state
        self.mouse_pos = (0, 0)  # Track current mouse position
        self.initialize_inclusions()

    def initialize_inclusions(self):
        """Initialize inclusions from detected masks - use centroids only."""
        for frame_idx in self.frames:
            self.inclusions[frame_idx] = []
            frame_data = self.visualization_data[frame_idx]

            # Combine all inclusion masks
            if "inclusion_masks" in frame_data:
                for mask in frame_data["inclusion_masks"]:
                    if np.any(mask):
                        # Find connected components and get centroids
                        num_labels, labels, stats, centroids = (
                            cv2.connectedComponentsWithStats(
                                mask.astype(np.uint8), connectivity=8
                            )
                        )
                        # Skip background (label 0)
                        for i in range(1, num_labels):
                            cx, cy = centroids[i]
                            self.inclusions[frame_idx].append((int(cx), int(cy)))

    def remove_inclusion_at(self, x, y):
        """Remove inclusion nearest to position if within threshold."""
        frame_idx = self.frames[self.current_index]
        if self.inclusions[frame_idx]:
            distances = [
                np.sqrt((x - ix) ** 2 + (y - iy) ** 2)
                for ix, iy in self.inclusions[frame_idx]
            ]
            min_dist = min(distances)
            if min_dist < 20:
                idx = distances.index(min_dist)
                ix, iy = self.inclusions[frame_idx].pop(idx)
                print(f"Removed inclusion at: {ix},{iy}")
                return True
        return False

    def draw_frame(self):
        """Draw current frame with inclusions."""
        frame_data = self.get_current_frame_data()
        min_proj = frame_data["min_projection"]
        frame_idx = self.frames[self.current_index]

        # Convert to BGR
        display = cv2.cvtColor(min_proj, cv2.COLOR_GRAY2BGR)

        # Draw each inclusion as semi-transparent red circle
        for x, y in self.inclusions[frame_idx]:
            overlay = display.copy()
            cv2.circle(overlay, (x, y), 7, (0, 0, 255), -1)  # 15px diameter
            display = cv2.addWeighted(display, 0.5, overlay, 0.5, 0)

        # Add status
        count = len(self.inclusions[frame_idx])
        status = f"Frame {frame_idx} | Inclusions: {count}"
        cv2.putText(
            display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        # Add controls hint
        hint = "Left: Add | Right(hold): Remove | c: Clear all | Arrows: Navigate | q/Esc: Exit"
        cv2.putText(
            display,
            hint,
            (10, display.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        return display

    def update_results_with_inclusions(self):
        """Update results with correct per-droplet inclusion counts."""
        for row in self.results_data:
            frame_idx = row["frame"]

            # Find inclusions that belong to this specific droplet
            droplet_inclusions = 0

            if frame_idx in self.inclusions:
                # Check each inclusion position against droplet location
                cx, cy = row["center_x"], row["center_y"]
                radius = row["diameter_px"] / 2

                for ix, iy in self.inclusions[frame_idx]:
                    # Check if inclusion is within this droplet
                    dist = np.sqrt((ix - cx) ** 2 + (iy - cy) ** 2)
                    if dist <= radius:
                        droplet_inclusions += 1

            row["inclusions"] = droplet_inclusions
            row["detected"] = False

        return self.results_data

    def run(self):
        """Run interactive editor."""
        print("\nINTERACTIVE INCLUSION EDITOR")
        print(
            "Left: Add | Right(hold): Remove | c: Clear all | Arrows: Navigate | q/Esc: Exit\n"
        )

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        def mouse_callback(event, x, y, flags, param):
            self.mouse_pos = (x, y)
            frame_idx = self.frames[self.current_index]

            if event == cv2.EVENT_LBUTTONDOWN:
                self.inclusions[frame_idx].append((x, y))
                print(f"Added inclusion at: {x},{y}")
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.right_mouse_down = True
                self.remove_inclusion_at(x, y)
            elif event == cv2.EVENT_RBUTTONUP:
                self.right_mouse_down = False
            elif event == cv2.EVENT_MOUSEMOVE and self.right_mouse_down:
                # Continue removing while right button held
                self.remove_inclusion_at(x, y)

        cv2.setMouseCallback(self.window_name, mouse_callback)

        while True:
            display = self.draw_frame()
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(30) & 0xFF  # Faster refresh for smooth removal

            if key == ord("c"):
                # Clear all inclusions in current frame
                frame_idx = self.frames[self.current_index]
                count = len(self.inclusions[frame_idx])
                self.inclusions[frame_idx] = []
                print(f"Cleared {count} inclusions from frame {frame_idx}")
            elif key == ord("q") or key == 27:  # q or ESC
                break
            elif key == 83 or key == ord(" "):  # Right arrow or space
                self.current_index = (self.current_index + 1) % len(self.frames)
            elif key == 81:  # Left arrow
                self.current_index = (self.current_index - 1) % len(self.frames)
            elif key == 13:  # Enter
                if self.current_index < len(self.frames) - 1:
                    self.current_index += 1
                else:
                    break

        cv2.destroyAllWindows()

        return self.update_results_with_inclusions()


class DropletStatistics:
    """Simplified statistical analysis for droplet detection."""

    def __init__(self, results_csv, settings=None):
        """Initialize with results data and optional settings."""
        self.df = pd.read_csv(results_csv)
        self.settings = settings or {}

        # Get Poisson parameters from settings or use defaults
        self.bead_count = self.settings.get("count", 6.5e5)  # Beads per uL
        self.dilution = self.settings.get("dilution", 1000)
        self.use_poisson = self.settings.get("poisson", True)

    def calculate_poisson(self, median_diameter_um):
        """Calculate theoretical Poisson distribution."""
        # Droplet volume in mL
        radius_um = median_diameter_um / 2
        volume_ml = (4 / 3) * np.pi * (radius_um**3) * 1e-9

        # Lambda (expected inclusions)
        lambda_val = (self.bead_count / (self.dilution * 2)) * volume_ml

        # Generate distribution
        max_inc = int(self.df["inclusions"].max()) + 3
        x_range = np.arange(0, max_inc + 1)
        theoretical = stats.poisson.pmf(x_range, lambda_val)

        return x_range, theoretical, lambda_val

    def plot_size_distribution(self, output_path):
        """Plot droplet diameter distribution."""
        fig, ax = plt.subplots(figsize=(8, 5))

        diameters = self.df["diameter_um"].values
        ax.hist(diameters, bins=25, color="steelblue", edgecolor="black", alpha=0.7)

        mean_d = np.mean(diameters)
        median_d = np.median(diameters)

        ax.axvline(mean_d, color="red", linestyle="--", label=f"Mean: {mean_d:.1f}")
        ax.axvline(
            median_d, color="green", linestyle="--", label=f"Median: {median_d:.1f}"
        )

        ax.set_xlabel("Diameter (µm)")
        ax.set_ylabel("Count")
        ax.set_title("Droplet Size Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "size_distribution.png", dpi=200)
        plt.close()

        return mean_d, median_d

    def plot_poisson_comparison(self, output_path):
        """Plot detected vs theoretical Poisson with chi-squared test."""
        fig, ax = plt.subplots(figsize=(10, 6))

        median_d = self.df["diameter_um"].median()
        x_range, theoretical, lambda_val = self.calculate_poisson(median_d)

        # Actual distribution
        actual = self.df["inclusions"].value_counts().sort_index()
        n_droplets = len(self.df)

        # Chi-squared test
        chi2, p_value = self.perform_chi_squared(actual, theoretical, n_droplets)

        # Prepare data for plotting
        detected_pct = []
        theoretical_pct = theoretical * 100

        for i in x_range:
            if i in actual.index:
                detected_pct.append(actual[i] / n_droplets * 100)
            else:
                detected_pct.append(0)

        # Plot bars
        x = np.arange(len(x_range))
        width = 0.35

        ax.bar(
            x - width / 2,
            detected_pct,
            width,
            label="Detected",
            color="royalblue",
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            theoretical_pct[: len(x)],
            width,
            label=f"Poisson (λ={lambda_val:.3f})",
            color="coral",
            alpha=0.8,
        )

        # Add chi-squared result
        if p_value is not None:
            result_text = f"χ² = {chi2:.2f}, p = {p_value:.4f}"
            if p_value > 0.05:
                result_text += "\n✓ Follows Poisson"
            else:
                result_text += "\n✗ Deviates from Poisson"
            ax.text(
                0.98,
                0.85,
                result_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        ax.set_xlabel("Inclusions per Droplet")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Inclusion Distribution: Detected vs Theoretical")
        ax.set_xticks(x)
        ax.set_xticklabels(x_range)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_path / "poisson_comparison.png", dpi=200)
        plt.close()

        return lambda_val, chi2, p_value

    def perform_chi_squared(self, observed_counts, theoretical_probs, n_total):
        """Perform chi-squared goodness-of-fit test."""
        # Prepare observed and expected frequencies
        observed = []
        expected = []

        for i in observed_counts.index:
            if i < len(theoretical_probs):
                obs = observed_counts[i]
                exp = theoretical_probs[i] * n_total

                # Collect all bins first (we'll filter later)
                observed.append(obs)
                expected.append(exp)

        # Convert to arrays
        observed = np.array(observed)
        expected = np.array(expected)

        # Filter bins with expected < 5 (but keep at least 2 bins)
        mask = expected >= 5
        if mask.sum() < 2:
            return None, None

        observed_filtered = observed[mask]
        expected_filtered = expected[mask]

        # Normalize expected to match observed sum (fixes floating point mismatch)
        expected_filtered = expected_filtered * (
            observed_filtered.sum() / expected_filtered.sum()
        )

        # Chi-squared test
        chi2, p_value = stats.chisquare(observed_filtered, expected_filtered)
        return chi2, p_value

    def run_analysis(self, output_dir):
        """Run analysis and print results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Generate size distribution plot (always)
        mean_d, median_d = self.plot_size_distribution(output_path)

        # Generate Poisson comparison plot (if enabled)
        lambda_val, chi2, p_value = None, None, None
        if self.use_poisson:
            lambda_val, chi2, p_value = self.plot_poisson_comparison(output_path)

        # Calculate stats
        total_droplets = len(self.df)
        total_inclusions = int(self.df["inclusions"].sum())
        with_inclusions = int((self.df["inclusions"] > 0).sum())
        std_d = self.df["diameter_um"].std()

        # Generate summary.txt
        self._write_summary(
            output_path,
            mean_d=mean_d,
            median_d=median_d,
            std_d=std_d,
            total_droplets=total_droplets,
            total_inclusions=total_inclusions,
            with_inclusions=with_inclusions,
            lambda_val=lambda_val,
            chi2=chi2,
            p_value=p_value,
        )

        # Print summary to console
        print("\nSTATISTICAL SUMMARY")
        print("-" * 40)
        print(f"Droplets: {total_droplets}")
        print(f"Mean diameter: {mean_d:.1f} µm")
        print(
            f"Inclusions: {total_inclusions} total, {total_inclusions / total_droplets:.2f} per droplet"
        )
        print(
            f"With inclusions: {with_inclusions} ({with_inclusions / total_droplets * 100:.1f}%)"
        )

        if self.use_poisson and lambda_val is not None:
            print(f"Theoretical λ: {lambda_val:.3f}")

            if p_value is not None:
                print(f"\nChi-squared test:")
                print(f"  χ² = {chi2:.2f}, p = {p_value:.4f}")
                if p_value > 0.05:
                    print("  → Distribution follows Poisson (p > 0.05)")
                else:
                    print("  → Distribution deviates from Poisson (p < 0.05)")

        print(f"\nOutput saved to: {output_path}")

    def _write_summary(self, output_path, **stats):
        """Write summary.txt file with all settings and statistics."""
        project_name = output_path.name
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        input_dir = self.settings.get("input_dir", "N/A")
        total_frames = self.df["frame"].nunique()

        lines = [
            "=" * 80,
            "DROPDROP ANALYSIS SUMMARY".center(80),
            "=" * 80,
            "",
            f"Project: {project_name}",
            f"Date: {timestamp}",
            f"Input: {input_dir} ({total_frames} frames)",
            "",
            "SETTINGS",
            "-" * 40,
            f"Poisson Analysis: {'ON' if self.use_poisson else 'OFF'}",
        ]

        if self.use_poisson:
            lines.extend([
                f"Stock Concentration: {self.bead_count:.2e} beads/uL",
                f"Dilution Factor: {self.dilution}x",
            ])

        lines.extend([
            "",
            "RESULTS",
            "-" * 40,
            f"Total Frames Processed: {total_frames}",
            f"Total Droplets Detected: {stats['total_droplets']:,}",
            f"Total Beads Detected: {stats['total_inclusions']:,}",
            "",
            "Droplet Statistics:",
            f"  Mean Diameter: {stats['mean_d']:.1f} um",
            f"  Median Diameter: {stats['median_d']:.1f} um",
            f"  Std Deviation: {stats['std_d']:.1f} um",
            "",
            "Bead Statistics:",
            f"  Mean per Droplet: {stats['total_inclusions'] / stats['total_droplets']:.2f}",
            f"  Droplets with Beads: {stats['with_inclusions']} ({stats['with_inclusions'] / stats['total_droplets'] * 100:.1f}%)",
        ])

        if self.use_poisson and stats.get("lambda_val") is not None:
            lines.extend([
                "",
                "POISSON ANALYSIS",
                "-" * 40,
                f"Theoretical Lambda: {stats['lambda_val']:.3f}",
            ])

            if stats.get("p_value") is not None:
                result = "FOLLOWS" if stats["p_value"] > 0.05 else "DEVIATES FROM"
                lines.extend([
                    f"Chi-squared: {stats['chi2']:.2f}",
                    f"P-value: {stats['p_value']:.4f}",
                    f"Result: Distribution {result} Poisson (p {'>' if stats['p_value'] > 0.05 else '<'} 0.05)",
                ])

        lines.extend([
            "",
            "=" * 80,
            "Generated by DropDrop v2.0",
            "=" * 80,
        ])

        summary_path = output_path / "summary.txt"
        with open(summary_path, "w") as f:
            f.write("\n".join(lines))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Droplet and inclusion detection pipeline using Cellpose"
    )

    parser.add_argument(
        "input_dir", type=str, help="Input directory containing z-stack images"
    )

    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default=None,
        help="Output directory (default: ./results/<date>_<label>)",
    )

    parser.add_argument(
        "-s",
        "--settings",
        type=str,
        default=None,
        help='Compact settings: "d=1000,p=on,c=6.5e5,l=label" (d=dilution, p=poisson, c=count, l=label)',
    )

    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument(
        "--view", action="store_true", help="Enable interactive viewer after processing"
    )
    viewer_group.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactive inclusion correction mode",
    )

    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=None,
        help="Process only the first N frames (for testing)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching for this run",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache before processing",
    )
    parser.add_argument(
        "-z",
        "--gzip",
        action="store_true",
        help="Archive project directory as .tar.gz after completion",
    )
    args = parser.parse_args()

    # Check input directory exists
    if not Path(args.input_dir).exists():
        print(f"ERROR: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)

    # Get settings (from --settings or interactive prompts)
    if args.settings:
        settings = parse_settings(args.settings)
    else:
        settings = prompt_settings()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        project_name = generate_project_name(settings)
        output_dir = Path("results") / project_name

    # Store settings for later use
    settings["input_dir"] = str(Path(args.input_dir).resolve())

    # Initialize and run pipeline
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {output_dir}")
    if settings["poisson"]:
        print(f"Poisson: ON (count={settings['count']:.2e}, dilution={settings['dilution']})")
    else:
        print("Poisson: OFF")
    if args.number:
        print(f"Frame limit: {args.number}")

    # Create pipeline with visualization storage if viewer is requested
    store_viz = args.view or args.interactive
    use_cache = not args.no_cache
    pipeline = DropletInclusionPipeline(store_visualizations=store_viz, use_cache=use_cache)

    # Handle cache clear request
    if args.clear_cache and pipeline.cache:
        pipeline.cache.clear()

    results = pipeline.run(args.input_dir, str(output_dir), frame_limit=args.number)

    if results:
        print("\nPipeline completed successfully!")

        # Interactive editing mode
        if args.interactive and pipeline.visualization_data:
            print("\nLaunching interactive inclusion editor...")
            editor = InclusionEditor(pipeline.visualization_data, results)
            results = editor.run()  # Update results with manual corrections

            # Save updated results
            df = pd.DataFrame(results)
            csv_path = output_dir / "data.csv"
            df.to_csv(csv_path, index=False)
            print(f"Updated results saved to: {csv_path}")

        # Always generate statistics (after any interactive corrections)
        print("\nGenerating statistical analysis...")
        csv_path = output_dir / "data.csv"
        stats_module = DropletStatistics(csv_path, settings)
        stats_module.run_analysis(str(output_dir))

        # Launch viewer if requested (no editing, just viewing)
        if args.view and pipeline.visualization_data:
            print("\nLaunching interactive viewer...")
            df = pd.DataFrame(results)
            viewer = Viewer(pipeline.visualization_data, df)
            viewer.run()

        # Archive project if requested
        if args.gzip:
            archive_name = f"{output_dir}.tar.gz"
            print(f"\nArchiving project to: {archive_name}")
            with tarfile.open(archive_name, "w:gz") as tar:
                tar.add(output_dir, arcname=output_dir.name)
            print(f"Archive created: {archive_name}")


if __name__ == "__main__":
    main()
