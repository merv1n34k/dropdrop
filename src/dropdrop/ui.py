"""User interface components for visualization and editing."""

import cv2
import numpy as np


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
                return False

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

        overlay = cv2.cvtColor(min_proj, cv2.COLOR_GRAY2BGR)

        for i, droplet_info in enumerate(frame_data["droplet_masks"]):
            cx, cy = droplet_info["center"]
            radius = int(droplet_info["radius"])
            inclusions = droplet_info["inclusions"]

            color = (0, 0, 255) if inclusions > 0 else (0, 255, 0)
            cv2.circle(overlay, (int(cx), int(cy)), radius, color, 2)

            eroded_radius = radius - self.df.iloc[0].get("erosion_pixels", 10)
            if eroded_radius > 0:
                cv2.circle(overlay, (int(cx), int(cy)), eroded_radius, (0, 255, 255), 1)

            cv2.circle(overlay, (int(cx), int(cy)), 3, (255, 0, 0), -1)

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

        min_bgr = cv2.cvtColor(min_proj, cv2.COLOR_GRAY2BGR)
        images.append(("Min Projection", min_bgr))

        droplet_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for i, mask in enumerate(frame_data["droplet_masks"]):
            droplet_mask = mask["mask"]
            color_val = (i * 30) % 200 + 55
            droplet_overlay[droplet_mask > 0] = [color_val, color_val, 0]
        images.append(("Cellpose Detection", droplet_overlay))

        eroded_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for eroded_mask in frame_data["eroded_masks"]:
            eroded_overlay[eroded_mask > 0] = [0, 200, 200]
        images.append(("Eroded Masks", eroded_overlay))

        if "masked_images" in frame_data and frame_data["masked_images"]:
            blackhat_combined = np.zeros((h, w), dtype=np.uint8)
            for masked_blackhat in frame_data["masked_images"]:
                blackhat_combined = cv2.bitwise_or(blackhat_combined, masked_blackhat)
            blackhat_bgr = cv2.cvtColor(blackhat_combined, cv2.COLOR_GRAY2BGR)
            images.append(("Black-hat (Masked)", blackhat_bgr))

        inclusion_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for inclusion_mask in frame_data["inclusion_masks"]:
            inclusion_overlay[:, :, 2] = cv2.bitwise_or(
                inclusion_overlay[:, :, 2], inclusion_mask
            )
        images.append(("Detected Inclusions", inclusion_overlay))

        final_overlay = self.create_overlay(frame_idx)
        images.append(("Final Result", final_overlay))

        cols = 3
        rows = 2
        collage = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 240

        for idx, (title, img) in enumerate(images[:6]):
            row = idx // cols
            col = idx % cols

            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))

            img_copy = img.copy()
            cv2.putText(
                img_copy, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

            collage[row * h : (row + 1) * h, col * w : (col + 1) * w] = img_copy

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

            key = cv2.waitKey(1) & 0xFF
            if key == ord("m"):
                self.mode = "overlay" if self.mode == "steps" else "steps"
                continue

            if not self.navigate():
                break

        cv2.destroyAllWindows()


class InclusionEditor(BaseWindow):
    """Interactive editor for inclusion corrections."""

    def __init__(self, visualization_data, results_data):
        super().__init__(visualization_data)
        self.results_data = results_data
        self.window_name = "Inclusion Editor"
        self.inclusions = {}
        self.right_mouse_down = False
        self.mouse_pos = (0, 0)
        self.initialize_inclusions()

    def initialize_inclusions(self):
        """Initialize inclusions from detected masks - use centroids only."""
        for frame_idx in self.frames:
            self.inclusions[frame_idx] = []
            frame_data = self.visualization_data[frame_idx]

            if "inclusion_masks" in frame_data:
                for mask in frame_data["inclusion_masks"]:
                    if np.any(mask):
                        num_labels, labels, stats, centroids = (
                            cv2.connectedComponentsWithStats(
                                mask.astype(np.uint8), connectivity=8
                            )
                        )
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

        display = cv2.cvtColor(min_proj, cv2.COLOR_GRAY2BGR)

        for x, y in self.inclusions[frame_idx]:
            overlay = display.copy()
            cv2.circle(overlay, (x, y), 7, (0, 0, 255), -1)
            display = cv2.addWeighted(display, 0.5, overlay, 0.5, 0)

        count = len(self.inclusions[frame_idx])
        status = f"Frame {frame_idx} | Inclusions: {count}"
        cv2.putText(
            display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

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
            droplet_inclusions = 0

            if frame_idx in self.inclusions:
                cx, cy = row["center_x"], row["center_y"]
                radius = row["diameter_px"] / 2

                for ix, iy in self.inclusions[frame_idx]:
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
                self.remove_inclusion_at(x, y)

        cv2.setMouseCallback(self.window_name, mouse_callback)

        while True:
            display = self.draw_frame()
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("c"):
                frame_idx = self.frames[self.current_index]
                count = len(self.inclusions[frame_idx])
                self.inclusions[frame_idx] = []
                print(f"Cleared {count} inclusions from frame {frame_idx}")
            elif key == ord("q") or key == 27:
                break
            elif key == 83 or key == ord(" "):
                self.current_index = (self.current_index + 1) % len(self.frames)
            elif key == 81:
                self.current_index = (self.current_index - 1) % len(self.frames)
            elif key == 13:
                if self.current_index < len(self.frames) - 1:
                    self.current_index += 1
                else:
                    break

        cv2.destroyAllWindows()

        return self.update_results_with_inclusions()
