"""User interface components for visualization and editing."""

import cv2
import numpy as np

from .config import load_config


class BaseWindow:
    """Base class for all window-based interfaces."""

    def __init__(self, visualization_data):
        self.visualization_data = visualization_data
        self.frames = sorted(visualization_data.keys())
        self.current_index = 0
        self.window_name = "Window"

    def get_current_frame_data(self):
        """Get current frame visualization data."""
        return self.visualization_data[self.frames[self.current_index]]

    def run(self):
        """Main window loop - to be overridden."""
        raise NotImplementedError


class Editor(BaseWindow):
    """Interactive editor for viewing and editing droplet/inclusion data."""

    LAYERS = [
        "Min Projection",
        "Cellpose Detection",
        "Eroded Masks",
        "Black-hat",
        "Detected Inclusions",
        "Overlay",
    ]

    def __init__(self, visualization_data, results_data, detect_inclusions=True):
        super().__init__(visualization_data)
        self.results_data = results_data
        self.window_name = "DropDrop Editor"
        self.detect_inclusions = detect_inclusions
        self.config = load_config()

        # Mode: "edit" or "view"
        self.mode = "edit"
        self.view_layer = 0

        # Editor state
        self.inclusions = {}
        self.undo_stack = {}
        self.disabled_droplets = {}
        self.right_mouse_down = False
        self.mouse_pos = (0, 0)
        self.show_droplets = True

        self.initialize_inclusions()

    def initialize_inclusions(self):
        """Initialize inclusions from detected masks - use centroids only."""
        for frame_idx in self.frames:
            self.inclusions[frame_idx] = []
            self.undo_stack[frame_idx] = []
            self.disabled_droplets[frame_idx] = set()
            frame_data = self.visualization_data[frame_idx]

            if self.detect_inclusions and "inclusion_masks" in frame_data:
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

    # -- Droplet interaction --

    def get_droplet_at(self, x, y):
        """Get droplet index at position, or None if not found."""
        frame_idx = self.frames[self.current_index]
        frame_data = self.visualization_data[frame_idx]

        for i, droplet_info in enumerate(frame_data.get("droplet_masks", [])):
            cx, cy = droplet_info["center"]
            radius = droplet_info["radius"]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist <= radius:
                return i
        return None

    def toggle_droplet(self, droplet_idx):
        """Toggle droplet enabled/disabled state."""
        frame_idx = self.frames[self.current_index]
        if droplet_idx in self.disabled_droplets[frame_idx]:
            self.disabled_droplets[frame_idx].remove(droplet_idx)
            print(f"Enabled droplet {droplet_idx}")
        else:
            self.disabled_droplets[frame_idx].add(droplet_idx)
            print(f"Disabled droplet {droplet_idx}")

    # -- Inclusion editing --

    def add_inclusion(self, x, y):
        """Add inclusion at position with undo tracking."""
        frame_idx = self.frames[self.current_index]
        pos = (x, y)
        self.inclusions[frame_idx].append(pos)
        self.undo_stack[frame_idx].append(("add", pos))
        print(f"Added inclusion at: {x},{y}")

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
                pos = self.inclusions[frame_idx].pop(idx)
                self.undo_stack[frame_idx].append(("remove", pos))
                print(f"Removed inclusion at: {pos}")
                return True
        return False

    def clear_inclusions(self):
        """Clear all inclusions in current frame with undo tracking."""
        frame_idx = self.frames[self.current_index]
        if self.inclusions[frame_idx]:
            old_inclusions = self.inclusions[frame_idx].copy()
            self.undo_stack[frame_idx].append(("clear", old_inclusions))
            self.inclusions[frame_idx] = []
            print(f"Cleared {len(old_inclusions)} inclusions from frame {frame_idx}")

    def undo(self):
        """Undo last action in current frame."""
        frame_idx = self.frames[self.current_index]
        if not self.undo_stack[frame_idx]:
            print("Nothing to undo")
            return

        action, data = self.undo_stack[frame_idx].pop()

        if action == "add":
            self.inclusions[frame_idx].remove(data)
            print(f"Undo: removed inclusion at {data}")
        elif action == "remove":
            self.inclusions[frame_idx].append(data)
            print(f"Undo: restored inclusion at {data}")
        elif action == "clear":
            self.inclusions[frame_idx] = data
            print(f"Undo: restored {len(data)} inclusions")

    # -- Display methods --

    @staticmethod
    def outlined_text(img, text, pos, font, scale, color, thickness):
        """Draw text with black outline for readability."""
        cv2.putText(img, text, pos, font, scale, (0, 0, 0), thickness + 2)
        cv2.putText(img, text, pos, font, scale, color, thickness)

    def draw_frame(self):
        """Draw current frame with droplet masks and inclusions (edit mode)."""
        frame_data = self.get_current_frame_data()
        min_proj = frame_data["min_projection"]
        frame_idx = self.frames[self.current_index]

        display = cv2.cvtColor(min_proj, cv2.COLOR_GRAY2BGR)

        # Draw droplet boundaries
        if self.show_droplets:
            for i, droplet_info in enumerate(frame_data.get("droplet_masks", [])):
                cx, cy = droplet_info["center"]
                radius = int(droplet_info["radius"])
                is_disabled = i in self.disabled_droplets[frame_idx]
                color = (128, 128, 128) if is_disabled else (0, 255, 0)
                thickness = 1 if is_disabled else 2
                cv2.circle(display, (int(cx), int(cy)), radius, color, thickness)
                cv2.circle(display, (int(cx), int(cy)), 3, color, -1)
                if is_disabled:
                    self.outlined_text(display, "X", (int(cx) - 8, int(cy) + 8),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

        # Draw inclusions
        if self.detect_inclusions:
            for x, y in self.inclusions[frame_idx]:
                overlay = display.copy()
                cv2.circle(overlay, (x, y), 7, (0, 0, 255), -1)
                display = cv2.addWeighted(display, 0.5, overlay, 0.5, 0)

        # Status bar
        total_droplets = len(frame_data.get("droplet_masks", []))
        active_droplets = total_droplets - len(self.disabled_droplets[frame_idx])
        if self.detect_inclusions:
            count = len(self.inclusions[frame_idx])
            status = f"[EDIT] Frame {frame_idx} | Droplets: {active_droplets}/{total_droplets} | Inclusions: {count}"
        else:
            status = f"[EDIT] Frame {frame_idx} | Droplets: {active_droplets}/{total_droplets} | Inclusions: OFF"
        self.outlined_text(
            display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        # Hint bar
        if self.detect_inclusions:
            hint = "Left: Add | Right: Remove | s: Toggle droplet | u: Undo | c: Clear | d: Droplets | m: View | Arrow/Space: Navigate | q: Exit"
        else:
            hint = "s: Toggle droplet | d: Droplets | m: View | Arrow/Space: Navigate | q: Exit"
        self.outlined_text(
            display, hint,
            (10, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        return display

    def create_overlay(self, frame_idx):
        """Create overlay visualization with colored droplets and inclusion counts."""
        frame_data = self.visualization_data[frame_idx]
        min_proj = frame_data["min_projection"]
        erosion_px = self.config.get("erosion_pixels", 10)

        overlay = cv2.cvtColor(min_proj, cv2.COLOR_GRAY2BGR)

        for i, droplet_info in enumerate(frame_data.get("droplet_masks", [])):
            cx, cy = droplet_info["center"]
            radius = int(droplet_info["radius"])
            is_disabled = i in self.disabled_droplets.get(frame_idx, set())

            if is_disabled:
                cv2.circle(overlay, (int(cx), int(cy)), radius, (128, 128, 128), 1)
                continue

            # Count live inclusions inside this droplet
            inc_count = 0
            if self.detect_inclusions:
                for ix, iy in self.inclusions.get(frame_idx, []):
                    dist = np.sqrt((ix - cx) ** 2 + (iy - cy) ** 2)
                    if dist <= radius:
                        inc_count += 1

            color = (0, 0, 255) if inc_count > 0 else (0, 255, 0)
            cv2.circle(overlay, (int(cx), int(cy)), radius, color, 2)

            eroded_radius = radius - erosion_px
            if eroded_radius > 0:
                cv2.circle(overlay, (int(cx), int(cy)), eroded_radius, (0, 255, 255), 1)

            cv2.circle(overlay, (int(cx), int(cy)), 3, (255, 0, 0), -1)

            if inc_count > 0:
                self.outlined_text(
                    overlay, str(inc_count),
                    (int(cx) - 10, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                )

        return overlay

    def create_layer(self, frame_idx, layer_index):
        """Render a single processing layer full-screen."""
        frame_data = self.visualization_data[frame_idx]
        min_proj = frame_data["min_projection"]
        h, w = min_proj.shape

        layer_name = self.LAYERS[layer_index]

        if layer_name == "Min Projection":
            img = cv2.cvtColor(min_proj, cv2.COLOR_GRAY2BGR)

        elif layer_name == "Cellpose Detection":
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for i, mask in enumerate(frame_data.get("droplet_masks", [])):
                droplet_mask = mask["mask"]
                color_val = (i * 30) % 200 + 55
                img[droplet_mask > 0] = [color_val, color_val, 0]

        elif layer_name == "Eroded Masks":
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for eroded_mask in frame_data.get("eroded_masks", []):
                img[eroded_mask > 0] = [0, 200, 200]

        elif layer_name == "Black-hat":
            blackhat_combined = np.zeros((h, w), dtype=np.uint8)
            for masked_blackhat in frame_data.get("masked_images", []):
                blackhat_combined = cv2.bitwise_or(blackhat_combined, masked_blackhat)
            img = cv2.cvtColor(blackhat_combined, cv2.COLOR_GRAY2BGR)

        elif layer_name == "Detected Inclusions":
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for inclusion_mask in frame_data.get("inclusion_masks", []):
                img[:, :, 2] = cv2.bitwise_or(img[:, :, 2], inclusion_mask)

        elif layer_name == "Overlay":
            img = self.create_overlay(frame_idx)

        else:
            img = cv2.cvtColor(min_proj, cv2.COLOR_GRAY2BGR)

        # Title label
        self.outlined_text(
            img, f"[VIEW] Frame {frame_idx} | {layer_name}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        # Hint bar
        hint = "v: Next layer | m: Edit mode | Arrow/Space: Navigate | q: Exit"
        self.outlined_text(
            img, hint,
            (10, img.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        return img

    # -- Results --

    def update_results_with_inclusions(self):
        """Update results with correct per-droplet inclusion counts."""
        filtered_results = []

        for row in self.results_data:
            frame_idx = row["frame"]
            droplet_id = row["droplet_id"]

            # Skip disabled droplets
            if droplet_id in self.disabled_droplets.get(frame_idx, set()):
                continue

            # Count inclusions for this droplet
            droplet_inclusions = 0
            if self.detect_inclusions and frame_idx in self.inclusions:
                cx, cy = row["center_x"], row["center_y"]
                radius = row["diameter_px"] / 2

                for ix, iy in self.inclusions[frame_idx]:
                    dist = np.sqrt((ix - cx) ** 2 + (iy - cy) ** 2)
                    if dist <= radius:
                        droplet_inclusions += 1

            row["inclusions"] = droplet_inclusions
            row["detected"] = False
            filtered_results.append(row)

        return filtered_results

    # -- Main loop --

    def run(self):
        """Run the editor."""
        print("\nDROPDROP EDITOR")
        if self.detect_inclusions:
            print("Left: Add | Right: Remove | s: Toggle droplet | u: Undo | c: Clear | d: Droplets | m: View | Arrow/Space: Navigate | q: Exit\n")
        else:
            print("s: Toggle droplet | d: Droplets | m: View | Arrow/Space: Navigate | q: Exit\n")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        def mouse_callback(event, x, y, flags, param):
            self.mouse_pos = (x, y)

            # Only handle mouse editing in edit mode with inclusions enabled
            if self.mode != "edit" or not self.detect_inclusions:
                return

            if event == cv2.EVENT_LBUTTONDOWN:
                self.add_inclusion(x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.right_mouse_down = True
                self.remove_inclusion_at(x, y)
            elif event == cv2.EVENT_RBUTTONUP:
                self.right_mouse_down = False
            elif event == cv2.EVENT_MOUSEMOVE and self.right_mouse_down:
                self.remove_inclusion_at(x, y)

        cv2.setMouseCallback(self.window_name, mouse_callback)

        try:
            while True:
                frame_idx = self.frames[self.current_index]

                if self.mode == "edit":
                    display = self.draw_frame()
                else:
                    display = self.create_layer(frame_idx, self.view_layer)

                cv2.imshow(self.window_name, display)

                key = cv2.waitKeyEx(30)

                # Mode switching
                if key == ord("m"):
                    if self.mode == "edit":
                        self.mode = "view"
                        self.view_layer = 0
                        print(f"View mode: {self.LAYERS[self.view_layer]}")
                    else:
                        self.mode = "edit"
                        print("Edit mode")
                    continue

                # View mode: cycle layers
                if key == ord("v") and self.mode == "view":
                    self.view_layer = (self.view_layer + 1) % len(self.LAYERS)
                    print(f"Layer: {self.LAYERS[self.view_layer]}")
                    continue

                # Edit mode keys (inclusion editing)
                if self.mode == "edit" and self.detect_inclusions:
                    if key == ord("c"):
                        self.clear_inclusions()
                        continue
                    elif key == ord("u"):
                        self.undo()
                        continue

                # Shared keys (both modes)
                if key == ord("d"):
                    self.show_droplets = not self.show_droplets
                    print(f"Droplet visibility: {'ON' if self.show_droplets else 'OFF'}")
                elif key == ord("s") and self.mode == "edit":
                    droplet_idx = self.get_droplet_at(*self.mouse_pos)
                    if droplet_idx is not None:
                        self.toggle_droplet(droplet_idx)
                elif key == ord("q") or key == 27:
                    break
                elif key in (83, 63235, 65363, 2555904) or key == ord(" "):
                    self.current_index = (self.current_index + 1) % len(self.frames)
                elif key in (81, 63234, 65361, 2424832):
                    self.current_index = (self.current_index - 1) % len(self.frames)
        except KeyboardInterrupt:
            print("\nEditor interrupted, applying current edits...")

        cv2.destroyAllWindows()

        return self.update_results_with_inclusions()
