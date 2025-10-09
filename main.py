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


class DropletInclusionPipeline:
    """Main pipeline for droplet and inclusion detection."""

    def __init__(self, store_visualizations=False):
        """Initialize pipeline with configuration."""
        self.config = self.load_config()
        self.results_data = []
        self.store_visualizations = store_visualizations
        self.visualization_data = {} if store_visualizations else None

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
        """Extract z-stack index and frame index from filename."""
        z_match = re.search(r"_z(\d+)_", filename)
        z_index = int(z_match.group(1)) if z_match else None

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
            if z_idx is not None and frame_idx is not None:
                frame_groups[frame_idx].append((z_idx, filepath))

        # Sort z-stacks within each frame
        for frame_idx in frame_groups:
            frame_groups[frame_idx].sort(key=lambda x: x[0])

        return frame_groups

    def create_min_projection(self, z_stack_files):
        """Create minimum intensity projection from z-stack"""
        images = []
        for z_idx, filepath in z_stack_files:
            # Handle different bit depth
            img = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Scale image to 0-255 range if needed
                if img.dtype == np.uint16:
                    p2, p98 = np.percentile(img, (2, 98))
                    img = np.clip(img, p2, p98)
                    img = ((img - p2) / (p98 - p2) * 255).astype(np.uint8)
                elif img.dtype != np.uint8:
                    img_min = img.min()
                    img_max = img.max()
                    if img_max > img_min:
                        img = ((img - img_min) / (img_max - img_min) * 255).astype(
                            np.uint8
                        )
                    else:
                        img = np.zeros_like(img, dtype=np.uint8)

                images.append(img)

        if not images:
            return None

        # Stack and compute minimum
        stack = np.stack(images, axis=0)
        min_proj = np.min(stack, axis=0)

        return min_proj.astype(np.uint8)

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

    def process_frame(self, frame_idx, min_projection):
        """Process a single frame for droplets and inclusions."""
        # Initialize frame visualization data if needed
        if self.store_visualizations:
            frame_viz = {
                "min_projection": min_projection,
                "droplet_masks": [],
                "eroded_masks": [],
                "inclusion_masks": [],
                "masked_images": [],
            }

        # Detect droplets
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
        for frame_idx in tqdm(frame_indices, desc="Processing frames"):
            # Create min projection
            min_proj = self.create_min_projection(frame_groups[frame_idx])

            if min_proj is None:
                continue

            # Process frame
            self.process_frame(frame_idx, min_proj)

        # Save results to CSV
        if self.results_data:
            df = pd.DataFrame(self.results_data)
            csv_path = output_path / "results.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")

            # Print summary statistics
            self.print_summary(df)
        else:
            print("\nNo droplets detected in any frame!")

        return self.results_data

    def print_summary(self, df):
        """Print summary statistics."""
        print("\n" + "=" * 50)
        print("PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Total frames processed: {df['frame'].nunique()}")
        print(f"Total droplets detected: {len(df)}")
        print(f"Total inclusions detected: {df['inclusions'].sum()}")
        print(f"\nDroplet statistics:")
        print(
            f"  Mean diameter: {df['diameter_um'].mean():.2f} ± {df['diameter_um'].std():.2f} µm"
        )
        print(
            f"  Diameter range: {df['diameter_um'].min():.2f} - {df['diameter_um'].max():.2f} µm"
        )
        print(f"\nInclusion statistics:")
        print(f"  Mean per droplet: {df['inclusions'].mean():.2f}")
        print(f"  Max per droplet: {df['inclusions'].max()}")
        print(
            f"  Droplets with inclusions: {(df['inclusions'] > 0).sum()} ({(df['inclusions'] > 0).sum() / len(df) * 100:.1f}%)"
        )

        # Distribution of inclusions
        inclusion_counts = df["inclusions"].value_counts().sort_index()
        print(f"\nInclusion distribution:")
        for count, num_droplets in inclusion_counts.head(10).items():
            print(f"  {count} inclusions: {num_droplets} droplets")


class InteractiveViewer:
    """Interactive viewer for detection results - displays pre-computed data only."""

    def __init__(self, visualization_data, results_df):
        """Initialize viewer with pre-computed visualization data."""
        self.visualization_data = visualization_data
        self.df = results_df
        self.frames = sorted(visualization_data.keys())
        self.current_index = 0

        # Display settings
        self.mode = "steps"  # 'steps' or 'overlay'
        self.window_name = "Droplet Detection Viewer"

        if not self.frames:
            print("Error: No visualization data available")
            return

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

        # 2. Detected Droplets (original masks)
        droplet_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for i, mask in enumerate(frame_data["droplet_masks"]):
            # Create colored overlay from stored mask
            droplet_mask = mask["mask"]
            color_val = (i * 30) % 200 + 55  # Different colors for each droplet
            droplet_overlay[droplet_mask > 0] = [color_val, color_val, 0]
        images.append(("Detected Droplets", droplet_overlay))

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
        """Run interactive viewer."""
        print("\n" + "=" * 50)
        print("INTERACTIVE VIEWER")
        print("=" * 50)
        print("Controls:")
        print("  → / Space / Click : Next frame")
        print("  ←                 : Previous frame")
        print("  m                 : Toggle mode (steps/overlay)")
        print("  q / ESC           : Quit")
        print("=" * 50 + "\n")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Mouse callback for navigation
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_index = (self.current_index + 1) % len(self.frames)

        cv2.setMouseCallback(self.window_name, mouse_callback)

        while True:
            # Get current frame
            frame_idx = self.frames[self.current_index]

            # Create visualization based on mode
            if self.mode == "overlay":
                display_img = self.create_overlay(frame_idx)
                mode_text = "Overlay"
            else:  # steps mode
                display_img = self.create_steps(frame_idx)
                mode_text = "Steps"

            # Add navigation info
            nav_text = f"[{self.current_index + 1}/{len(self.frames)}] Mode: {mode_text} (press 'm' to toggle)"
            h = display_img.shape[0]
            cv2.putText(
                display_img,
                nav_text,
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

            # Show image
            cv2.imshow(self.window_name, display_img)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # q or ESC
                break
            elif key == ord(" ") or key == 83:  # Space or right arrow
                self.current_index = (self.current_index + 1) % len(self.frames)
            elif key == 81:  # Left arrow
                self.current_index = (self.current_index - 1) % len(self.frames)
            elif key == ord("m"):  # Toggle mode
                self.mode = "overlay" if self.mode == "steps" else "steps"
                print(f"Switched to {self.mode} mode")

        cv2.destroyAllWindows()
        print("\nViewer closed")


class DropletStatistics:
    """Statistical analysis and visualization for droplet detection results."""

    def __init__(self, results_csv, config=None):
        """Initialize with results data and configuration."""
        self.df = pd.read_csv(results_csv)
        self.config = config or {}

        # Poisson parameters
        self.bead_count = 6.5e5  # Beads per uL
        self.dilution = 1000

    def calculate_theoretical_poisson(self, median_diameter_um):
        """Calculate theoretical Poisson distribution for inclusions.

        Based on:
        - droplet_volume = (4/3) * π * (diameter/2)³ * 10^-9 (to convert µm³ to mL)
        - lambda = bead_count / (dilution * 2) * droplet_volume
        - Factor of 2 for mixing beads with buffer in equal volumes
        """
        # Calculate droplet volume in mL
        radius_um = median_diameter_um / 2
        droplet_volume_um3 = (4 / 3) * np.pi * (radius_um**3)
        droplet_volume_ml = droplet_volume_um3 * 1e-9

        # Calculate lambda (expected inclusions per droplet)
        lambda_val = (self.bead_count / (self.dilution * 2)) * droplet_volume_ml

        # Calculate probabilities for inclusion counts
        max_inclusions = int(self.df["inclusions"].max()) + 5
        inclusion_range = np.arange(0, max_inclusions + 1)
        theoretical_probs = stats.poisson.pmf(inclusion_range, lambda_val)

        return inclusion_range, theoretical_probs, lambda_val

    def plot_diameter_distribution(self, output_path):
        """Create and save droplet diameter distribution histogram."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get diameter data
        diameters = self.df["diameter_um"].values

        # Create histogram
        n_bins = min(30, len(np.unique(diameters)))
        counts, bins, patches = ax.hist(
            diameters, bins=n_bins, color="steelblue", edgecolor="black", alpha=0.7
        )

        # Add statistics
        mean_d = np.mean(diameters)
        median_d = np.median(diameters)
        std_d = np.std(diameters)

        # Add vertical lines for mean and median
        ax.axvline(
            mean_d,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_d:.1f} µm",
        )
        ax.axvline(
            median_d,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_d:.1f} µm",
        )

        # Labels and title
        ax.set_xlabel("Droplet Diameter (µm)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Droplet Diameter Distribution", fontsize=14, fontweight="bold")

        # Add text box with statistics
        stats_text = f"n = {len(diameters)}\nMean = {mean_d:.1f} µm\n"
        stats_text += f"Median = {median_d:.1f} µm\nSD = {std_d:.1f} µm"
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=10,
        )

        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_path / "diameter_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        return mean_d, median_d, std_d

    def plot_poisson_comparison(self, output_path):
        """Create Poisson distribution comparison plot."""
        fig, ax = plt.subplots(figsize=(12, 7))

        # Get median diameter for theoretical calculation
        median_diameter = self.df["diameter_um"].median()

        # Calculate theoretical Poisson
        inclusion_range, theoretical_probs, lambda_val = (
            self.calculate_theoretical_poisson(median_diameter)
        )

        # Get actual inclusion distribution
        actual_counts = self.df["inclusions"].value_counts().sort_index()
        max_inclusions = int(actual_counts.index.max())

        # Ensure we have all inclusion counts from 0 to max
        all_inclusions = np.arange(0, max(max_inclusions + 1, len(inclusion_range)))
        actual_percentages = []

        total_droplets = len(self.df)
        for i in all_inclusions:
            if i in actual_counts.index:
                actual_percentages.append(actual_counts[i] / total_droplets * 100)
            else:
                actual_percentages.append(0)

        # Prepare theoretical percentages
        theoretical_percentages = theoretical_probs[: len(all_inclusions)] * 100

        # Create grouped bar plot
        x = np.arange(len(all_inclusions))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            actual_percentages,
            width,
            label="Detected",
            color="royalblue",
            alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2,
            theoretical_percentages,
            width,
            label=f"Theoretical (λ={lambda_val:.3f})",
            color="coral",
            alpha=0.8,
        )

        # Labels and title
        ax.set_xlabel("Number of Inclusions", fontsize=12)
        ax.set_ylabel("Percentage of Droplets (%)", fontsize=12)
        ax.set_title(
            "Inclusion Distribution: Detected vs Theoretical Poisson",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(all_inclusions)

        # Add legend and grid
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        # Add chi-square test if applicable (with proper frequency matching)
        if len(actual_counts) > 1:
            try:
                # Get the range of observed values
                observed_range = actual_counts.index.values

                # Calculate expected counts only for observed range
                expected_probs_subset = []
                observed_values = []

                for inc_val in observed_range:
                    if inc_val < len(theoretical_probs):
                        expected_probs_subset.append(theoretical_probs[inc_val])
                        observed_values.append(actual_counts[inc_val])

                if len(expected_probs_subset) > 1:
                    # Convert to numpy arrays
                    observed = np.array(observed_values)
                    expected_probs_array = np.array(expected_probs_subset)

                    # Normalize expected probabilities to sum to 1 for observed range
                    expected_probs_normalized = (
                        expected_probs_array / expected_probs_array.sum()
                    )

                    # Calculate expected counts
                    expected_counts = expected_probs_normalized * observed.sum()

                    # Only use bins with expected count > 5 for chi-square
                    mask = expected_counts > 5
                    if mask.sum() > 1:
                        chi2, p_value = stats.chisquare(
                            observed[mask], expected_counts[mask]
                        )

                        test_text = f"χ² = {chi2:.2f}, p = {p_value:.4f}"
                        ax.text(
                            0.98,
                            0.85,
                            test_text,
                            transform=ax.transAxes,
                            verticalalignment="top",
                            horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
                            fontsize=10,
                        )
            except Exception as e:
                # If chi-square test fails, just skip it
                print(f"Chi-square test skipped: {e}")

        # Add parameters text
        params_text = f"Median diameter: {median_diameter:.1f} µm\n"
        params_text += f"Bead count: {self.bead_count:.1e}/mL\n"
        params_text += f"Dilution: 1:{self.dilution}\n"
        params_text += f"Expected λ: {lambda_val:.3f}"

        ax.text(
            0.98,
            0.70,
            params_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
        )

        plt.tight_layout()
        plt.savefig(
            output_path / "poisson_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        return lambda_val, actual_counts, theoretical_probs

    def plot_inclusion_distribution(self, output_path):
        """Create a simple inclusion count distribution plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get inclusion counts
        inclusion_counts = self.df["inclusions"].value_counts().sort_index()

        # Create bar plot
        ax.bar(
            inclusion_counts.index,
            inclusion_counts.values,
            color="darkgreen",
            edgecolor="black",
            alpha=0.7,
        )

        # Labels and title
        ax.set_xlabel("Number of Inclusions per Droplet", fontsize=12)
        ax.set_ylabel("Number of Droplets", fontsize=12)
        ax.set_title("Inclusion Count Distribution", fontsize=14, fontweight="bold")

        # Add statistics
        mean_inc = self.df["inclusions"].mean()
        median_inc = self.df["inclusions"].median()

        stats_text = f"Mean: {mean_inc:.2f}\nMedian: {median_inc:.1f}\n"
        stats_text += f"Total droplets: {len(self.df)}\n"
        stats_text += f"With inclusions: {(self.df['inclusions'] > 0).sum()}"

        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=10,
        )

        ax.grid(True, alpha=0.3, axis="y")

        # Set integer x-axis
        ax.set_xticks(
            range(
                int(inclusion_counts.index.min()), int(inclusion_counts.index.max()) + 1
            )
        )

        plt.tight_layout()
        plt.savefig(
            output_path / "inclusion_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        return mean_inc, median_inc

    def generate_report(self, output_path):
        """Generate comprehensive statistical report."""
        output_path = Path(output_path)

        # Generate plots
        print("\nGenerating statistical plots...")

        # 1. Diameter distribution
        mean_d, median_d, std_d = self.plot_diameter_distribution(output_path)
        print("  ✓ Diameter distribution plot created")

        # 2. Poisson comparison
        lambda_val, actual_counts, theoretical_probs = self.plot_poisson_comparison(
            output_path
        )
        print("  ✓ Poisson comparison plot created")

        # 3. Inclusion distribution
        mean_inc, median_inc = self.plot_inclusion_distribution(output_path)
        print("  ✓ Inclusion distribution plot created")

        # Generate text report
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("DROPLET DETECTION STATISTICAL REPORT")
        report_lines.append("=" * 60)
        report_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append("")

        # Overall statistics
        report_lines.append("OVERALL STATISTICS")
        report_lines.append("-" * 30)
        report_lines.append(f"Total frames analyzed: {self.df['frame'].nunique()}")
        report_lines.append(f"Total droplets detected: {len(self.df)}")
        report_lines.append(f"Total inclusions detected: {self.df['inclusions'].sum()}")
        report_lines.append("")

        # Droplet statistics
        report_lines.append("DROPLET SIZE STATISTICS")
        report_lines.append("-" * 30)
        report_lines.append(f"Mean diameter: {mean_d:.2f} µm")
        report_lines.append(f"Median diameter: {median_d:.2f} µm")
        report_lines.append(f"Standard deviation: {std_d:.2f} µm")
        report_lines.append(f"Min diameter: {self.df['diameter_um'].min():.2f} µm")
        report_lines.append(f"Max diameter: {self.df['diameter_um'].max():.2f} µm")
        report_lines.append("")

        # Inclusion statistics
        report_lines.append("INCLUSION STATISTICS")
        report_lines.append("-" * 30)
        report_lines.append(f"Mean inclusions per droplet: {mean_inc:.2f}")
        report_lines.append(f"Median inclusions per droplet: {median_inc:.1f}")
        report_lines.append(
            f"Max inclusions in a single droplet: {self.df['inclusions'].max()}"
        )
        report_lines.append(
            f"Droplets with inclusions: {(self.df['inclusions'] > 0).sum()} "
            f"({(self.df['inclusions'] > 0).sum() / len(self.df) * 100:.1f}%)"
        )
        report_lines.append(
            f"Droplets without inclusions: {(self.df['inclusions'] == 0).sum()} "
            f"({(self.df['inclusions'] == 0).sum() / len(self.df) * 100:.1f}%)"
        )
        report_lines.append("")

        # Poisson analysis
        report_lines.append("POISSON ANALYSIS")
        report_lines.append("-" * 30)
        report_lines.append(f"Theoretical λ (expected inclusions): {lambda_val:.3f}")
        report_lines.append(f"Based on median droplet diameter: {median_d:.1f} µm")
        report_lines.append(f"Bead concentration: {self.bead_count:.1e} beads/mL")
        report_lines.append(f"Dilution factor: 1:{self.dilution}")
        report_lines.append("")

        # Inclusion distribution
        report_lines.append("INCLUSION DISTRIBUTION")
        report_lines.append("-" * 30)
        for inc_count in sorted(actual_counts.index):
            count = actual_counts[inc_count]
            percentage = count / len(self.df) * 100
            report_lines.append(
                f"  {inc_count} inclusions: {count} droplets ({percentage:.1f}%)"
            )

        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 60)

        # Save report
        report_text = "\n".join(report_lines)
        report_path = output_path / "statistical_report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)

        # Print to console
        print("\n" + report_text)
        print(f"\nReport saved to: {report_path}")

        return report_text

    def run_analysis(self, output_dir):
        """Run complete statistical analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 50)
        print("STATISTICAL ANALYSIS")
        print("=" * 50)

        # Generate report and plots
        self.generate_report(output_path)

        print("\n✓ Statistical analysis complete!")
        print(f"  Results saved in: {output_path}")
        print("  Generated files:")
        print("    - diameter_distribution.png")
        print("    - poisson_comparison.png")
        print("    - inclusion_distribution.png")
        print("    - statistical_report.txt")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Droplet and inclusion detection pipeline using Cellpose"
    )

    parser.add_argument(
        "input_dir", type=str, help="Input directory containing z-stack images"
    )

    parser.add_argument("output_dir", type=str, help="Output directory for results")

    parser.add_argument(
        "--view", action="store_true", help="Enable interactive viewer after processing"
    )

    parser.add_argument(
        "--stats", action="store_true", help="Generate statistical analysis and plots"
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=None,
        help="Process only the first N frames (for testing)",
    )

    args = parser.parse_args()

    # Check input directory exists
    if not Path(args.input_dir).exists():
        print(f"ERROR: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)

    # Initialize and run pipeline
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.number:
        print(f"Frame limit: {args.number}")

    # Create pipeline with visualization storage if viewer is requested
    pipeline = DropletInclusionPipeline(store_visualizations=args.view)
    results = pipeline.run(args.input_dir, args.output_dir, frame_limit=args.number)

    if results:
        print("\n✓ Pipeline completed successfully!")

        # Generate statistics if requested
        if args.stats:
            print("\nGenerating statistical analysis...")
            csv_path = Path(args.output_dir) / "results.csv"
            stats_module = DropletStatistics(csv_path, pipeline.config)
            stats_module.run_analysis(args.output_dir)

        # Launch viewer if requested
        if args.view:
            csv_path = Path(args.output_dir) / "results.csv"
            if csv_path.exists() and pipeline.visualization_data:
                print("\nLaunching interactive viewer...")
                df = pd.DataFrame(results)
                viewer = InteractiveViewer(pipeline.visualization_data, df)
                viewer.run()
            else:
                print("Warning: Visualization data not available")


if __name__ == "__main__":
    main()
