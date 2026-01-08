#!/usr/bin/env python3
"""
DXF Shape Maker for Satellite Mission Planning

Interactive tool for creating custom path shapes for satellite missions.
Creates DXF files compatible with shape-following missions within
3m x 3m workspace.

Drawing features:
- Interactive mouse-based drawing interface
- Multiple shape types: lines, circles, arcs, splines
- Real-time shape preview with coordinate display
- Grid snapping for precision alignment
- Multi-layer support for complex paths
- Undo/redo functionality

Shape types supported:
- Lines: Straight paths between points
- Circles: Circular paths with center and radius
- Arcs: Partial circular paths
- Splines: Smooth curves through control points

Controls:
- Left click: Add point to current shape
- Right click: Finish current shape and start new one
- 'c' key: Switch to circle mode
- 'a' key: Switch to arc mode
- 'l' key: Switch to line mode (default)
- 's' key: Switch to spline mode
- 'g' key: Toggle grid snapping
- 'z' key: Undo last point
- 'r' key: Reset all shapes
- 'e' key: Export to DXF file
- 'q' key: Quit application

Export format:
- DXF R2010 format for CAD compatibility
- Coordinate system matches satellite workspace
- Ready for use in shape-following missions
"""

import os
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import matplotlib

# Force interactive backend
matplotlib.use("TkAgg")  # Use Tkinter backend for better interactivity
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

try:
    import ezdxf

    DXF_AVAILABLE = True
except ImportError:
    DXF_AVAILABLE = False
    print("  ezdxf not available. Install with: pip install ezdxf")


def to_3d(point_2d: Tuple[float, float]) -> Tuple[float, float, float]:
    """Convert 2D point to 3D with Z=0."""
    return (point_2d[0], point_2d[1], 0.0)


class DXFShapeMaker:
    """Interactive DXF shape creation tool."""

    def __init__(self):
        self.area_size = 3.0  # meters
        self.grid_size = 0.1  # 10cm grid

        # Drawing state
        self.current_points = []
        self.shapes = []
        self.drawing_mode = "line"  # "line", "circle", "arc", "spline"
        self.grid_snap = True

        # Circle/Arc specific
        self.circle_center = None
        self.arc_start_angle = None
        self._last_mouse_pos: Optional[Tuple[float, float]] = None

        # Setup matplotlib
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        # Instructions
        self.show_instructions()

    def setup_plot(self):
        """Setup the drawing area with grid and boundaries."""
        # Set equal aspect ratio and limits
        self.ax.set_xlim(-self.area_size / 2, self.area_size / 2)
        self.ax.set_ylim(-self.area_size / 2, self.area_size / 2)
        self.ax.set_aspect("equal")

        # Draw boundary
        boundary = patches.Rectangle(
            (-self.area_size / 2, -self.area_size / 2),
            self.area_size,
            self.area_size,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            zorder=1,
        )
        self.ax.add_patch(boundary)

        # Draw grid
        self.draw_grid()

        # Labels and title
        self.ax.set_xlabel("X Position (meters)")
        self.ax.set_ylabel("Y Position (meters)")
        self.ax.set_title("DXF Shape Maker - 3m × 3m Drawing Area")
        self.ax.grid(True, alpha=0.3)

        # Add mode indicator
        self.mode_text = self.ax.text(
            -1.4,
            1.4,
            f"Mode: {self.drawing_mode.upper()}",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
        )

        # Add snap indicator
        snap_status = "ON" if self.grid_snap else "OFF"
        self.snap_text = self.ax.text(
            -1.4,
            1.2,
            f"Grid Snap: {snap_status}",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
        )

    def draw_grid(self):
        """Draw grid lines for snapping."""
        if not self.grid_snap:
            return

        # Vertical lines
        x_lines = np.arange(
            -self.area_size / 2, self.area_size / 2 + self.grid_size, self.grid_size
        )
        for x in x_lines:
            self.ax.axvline(float(x), color="gray", alpha=0.2, linewidth=0.5)

        # Horizontal lines
        y_lines = np.arange(
            -self.area_size / 2, self.area_size / 2 + self.grid_size, self.grid_size
        )
        for y in y_lines:
            self.ax.axhline(float(y), color="gray", alpha=0.2, linewidth=0.5)

    def snap_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        """Snap coordinates to grid if enabled."""
        if not self.grid_snap:
            return x, y

        snapped_x = round(x / self.grid_size) * self.grid_size
        snapped_y = round(y / self.grid_size) * self.grid_size
        return snapped_x, snapped_y

    def is_in_bounds(self, x: float, y: float) -> bool:
        """Check if point is within the 3m x 3m area."""
        return bool(abs(x) <= self.area_size / 2 and abs(y) <= self.area_size / 2)

    def on_click(self, event):
        """Handle mouse clicks for drawing."""
        if event.inaxes != self.ax:
            return

        x, y = self.snap_to_grid(event.xdata, event.ydata)

        if not self.is_in_bounds(x, y):
            print(f" Point ({x:.2f}, {y:.2f}) is outside the 3m × 3m area")
            return

        if event.button == 1:  # Left click
            self.add_point(x, y)
        elif event.button == 3:  # Right click
            self.finish_current_shape()

    def add_point(self, x: float, y: float):
        """Add a point to the current shape (stored as 3D)."""
        if self.drawing_mode == "line" or self.drawing_mode == "spline":
            self.current_points.append((x, y, 0.0))
            print(f" Added point: ({x:.2f}, {y:.2f}, 0.00)")

        elif self.drawing_mode == "circle":
            if self.circle_center is None:
                self.circle_center = (x, y)
                print(f" Circle center: ({x:.2f}, {y:.2f}). " "Click again for radius.")
            else:
                radius = np.sqrt(
                    (x - self.circle_center[0]) ** 2 + (y - self.circle_center[1]) ** 2
                )
                self.add_circle(self.circle_center, radius)
                self.circle_center = None

        elif self.drawing_mode == "arc":
            if self.circle_center is None:
                self.circle_center = (x, y)
                print(f" Arc center: ({x:.2f}, {y:.2f}). " "Click for start point.")
            elif self.arc_start_angle is None:
                radius = np.sqrt(
                    (x - self.circle_center[0]) ** 2 + (y - self.circle_center[1]) ** 2
                )
                self.arc_start_angle = np.arctan2(
                    y - self.circle_center[1], x - self.circle_center[0]
                )
                print(
                    f" Arc start angle: "
                    f"{np.degrees(self.arc_start_angle):.1f}°. "
                    "Click for end point."
                )
            else:
                end_angle = np.arctan2(y - self.circle_center[1], x - self.circle_center[0])
                radius = np.sqrt(
                    (x - self.circle_center[0]) ** 2 + (y - self.circle_center[1]) ** 2
                )
                self.add_arc(self.circle_center, radius, self.arc_start_angle, end_angle)
                self.circle_center = None
                self.arc_start_angle = None

        self.update_display()

    def add_circle(self, center: Tuple[float, float], radius: float):
        """Add a complete circle to shapes."""
        # Sample circle into points
        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        points = []
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            if self.is_in_bounds(x, y):
                points.append((x, y))

        if len(points) >= 3:  # Need at least 3 points for a shape
            # Convert circle sample points to 3D
            points_3d = [(p[0], p[1], 0.0) for p in points]
            self.shapes.append(
                {
                    "type": "circle",
                    "center": (center[0], center[1], 0.0),
                    "radius": radius,
                    "points": points_3d,
                }
            )
            print(
                f" Added circle: center=({center[0]:.2f}, {center[1]:.2f}), "
                f"radius={radius:.2f}m"
            )
        else:
            print(" Circle too large for 3m × 3m area")

    def add_arc(
        self,
        center: Tuple[float, float],
        radius: float,
        start_angle: float,
        end_angle: float,
    ):
        """Add an arc to shapes."""
        # Ensure end_angle > start_angle
        if end_angle < start_angle:
            end_angle += 2 * np.pi

        # Sample arc into points
        arc_span = end_angle - start_angle
        num_points = max(3, int(np.degrees(arc_span) / 10))
        angles = np.linspace(start_angle, end_angle, num_points)

        points = []
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            if self.is_in_bounds(x, y):
                points.append((x, y))

        if len(points) >= 2:
            points_3d = [(p[0], p[1], 0.0) for p in points]
            self.shapes.append(
                {
                    "type": "arc",
                    "center": (center[0], center[1], 0.0),
                    "radius": radius,
                    "start_angle": start_angle,
                    "end_angle": end_angle,
                    "points": points_3d,
                }
            )
            print(
                f" Added arc: center=({center[0]:.2f}, {center[1]:.2f}), "
                f"radius={radius:.2f}m, span={np.degrees(arc_span):.1f}°"
            )
        else:
            print(" Arc outside 3m × 3m area")

    def finish_current_shape(self):
        """Finish the current line/spline shape."""
        if len(self.current_points) >= 2:
            self.shapes.append({"type": self.drawing_mode, "points": self.current_points.copy()})
            print(f" Added {self.drawing_mode}: " f"{len(self.current_points)} points")
            self.current_points = []
            self.update_display()
        else:
            print(f" Need at least 2 points for a {self.drawing_mode}")

    def on_key(self, event):
        """Handle keyboard shortcuts."""
        if event.key == "l":
            self.drawing_mode = "line"
            self.reset_drawing_state()
        elif event.key == "c":
            self.drawing_mode = "circle"
            self.reset_drawing_state()
        elif event.key == "a":
            self.drawing_mode = "arc"
            self.reset_drawing_state()
        elif event.key == "s":
            self.drawing_mode = "spline"
            self.reset_drawing_state()
        elif event.key == "g":
            self.grid_snap = not self.grid_snap
            self.update_snap_display()
        elif event.key == "z":
            self.undo_last_point()
        elif event.key == "r":
            self.reset_all()
        elif event.key == "e":
            self.export_dxf()
        elif event.key == "q":
            plt.close()
            return

        self.update_mode_display()
        self.update_display()

    def reset_drawing_state(self):
        """Reset current drawing state when changing modes."""
        self.current_points = []
        self.circle_center = None
        self.arc_start_angle = None

    def undo_last_point(self):
        """Remove the last added point."""
        if self.current_points:
            removed = self.current_points.pop()
            print(f"↶ Undid point: ({removed[0]:.2f}, {removed[1]:.2f})")
            self.update_display()
        elif self.circle_center is not None:
            self.circle_center = None
            self.arc_start_angle = None
            print("↶ Reset circle/arc")
        else:
            print(" Nothing to undo")

    def reset_all(self):
        """Clear all shapes and current drawing."""
        self.shapes = []
        self.current_points = []
        self.circle_center = None
        self.arc_start_angle = None
        print(" Reset all shapes")
        self.update_display()

    def update_mode_display(self):
        """Update the mode indicator."""
        self.mode_text.set_text(f"Mode: {self.drawing_mode.upper()}")

    def update_snap_display(self):
        """Update the grid snap indicator."""
        snap_status = "ON" if self.grid_snap else "OFF"
        self.snap_text.set_text(f"Grid Snap: {snap_status}")
        # Redraw grid
        self.ax.clear()
        self.setup_plot()
        self.update_display()

    def on_mouse_move(self, event):
        """Handle mouse movement for preview."""
        if event.inaxes == self.ax and event.xdata is not None:
            x, y = self.snap_to_grid(event.xdata, event.ydata)
            # Update status in terminal
            if self._last_mouse_pos is not None:
                if (
                    abs(x - self._last_mouse_pos[0]) > 0.01
                    or abs(y - self._last_mouse_pos[1]) > 0.01
                ):
                    print(f"  Mouse: ({x:.2f}, {y:.2f}) - " f"Mode: {self.drawing_mode}")
            self._last_mouse_pos = (x, y)

    def update_display(self):
        """Redraw all shapes and current drawing."""
        for artist in self.ax.get_children():
            label = artist.get_label() if hasattr(artist, "get_label") else ""
            if isinstance(label, str) and label.startswith("shape_"):
                artist.remove()

        # Draw completed shapes
        for i, shape in enumerate(self.shapes):
            if shape["type"] in ["line", "spline"]:
                points = np.array(shape["points"])
                if len(points) >= 2:
                    self.ax.plot(
                        points[:, 0],
                        points[:, 1],
                        "b-",
                        linewidth=2,
                        label=f"shape_{i}",
                        alpha=0.8,
                    )
                    # Mark points
                    self.ax.plot(
                        points[:, 0],
                        points[:, 1],
                        "bo",
                        markersize=4,
                        label=f"shape_{i}_points",
                    )

            elif shape["type"] == "circle":
                circle = Circle(
                    shape["center"],
                    shape["radius"],
                    fill=False,
                    color="blue",
                    linewidth=2,
                    label=f"shape_{i}",
                )
                self.ax.add_patch(circle)
                # Mark center
                self.ax.plot(
                    shape["center"][0],
                    shape["center"][1],
                    "bo",
                    markersize=6,
                    label=f"shape_{i}_center",
                )

            elif shape["type"] == "arc":
                # Draw arc using points
                points = np.array(shape["points"])
                if len(points) >= 2:
                    self.ax.plot(
                        points[:, 0],
                        points[:, 1],
                        "b-",
                        linewidth=2,
                        label=f"shape_{i}",
                        alpha=0.8,
                    )
                # Mark center
                self.ax.plot(
                    shape["center"][0],
                    shape["center"][1],
                    "bo",
                    markersize=4,
                    label=f"shape_{i}_center",
                )

        # Draw current points being added
        if self.current_points:
            current_array = np.array(self.current_points)
            self.ax.plot(
                current_array[:, 0],
                current_array[:, 1],
                "r--",
                linewidth=2,
                alpha=0.7,
                label="current",
            )
            self.ax.plot(
                current_array[:, 0],
                current_array[:, 1],
                "ro",
                markersize=6,
                label="current_points",
            )

        # Draw circle/arc preview
        if self.circle_center is not None:
            self.ax.plot(
                self.circle_center[0],
                self.circle_center[1],
                "go",
                markersize=8,
                label="center",
            )

        self.fig.canvas.draw()

    def export_dxf(self):
        """Export all shapes to a DXF file."""
        if not DXF_AVAILABLE:
            print(" Cannot export: ezdxf not installed")
            return

        if not self.shapes and not self.current_points:
            print(" No shapes to export")
            return

        # Create DXF document
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()

        # Add completed shapes
        for shape in self.shapes:
            if shape["type"] == "line":
                points = shape["points"]
                for i in range(len(points) - 1):
                    msp.add_line(points[i], points[i + 1])

            elif shape["type"] == "spline":
                points = shape["points"]
                # Use add_polyline3d for true 3D support
                msp.add_polyline3d(points)

            elif shape["type"] == "circle":
                msp.add_circle(shape["center"], shape["radius"])

            elif shape["type"] == "arc":
                center = shape["center"]
                radius = shape["radius"]
                start_angle = np.degrees(shape["start_angle"])
                end_angle = np.degrees(shape["end_angle"])
                msp.add_arc(center, radius, start_angle, end_angle)

        if len(self.current_points) >= 2:
            for i in range(len(self.current_points) - 1):
                msp.add_line(self.current_points[i], self.current_points[i + 1])

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"DXF_Files/custom_shape_{timestamp}.dxf"

        # Ensure directory exists
        os.makedirs("DXF_Files", exist_ok=True)

        # Save file
        doc.saveas(filename)
        print(f" Exported to: {filename}")
        print(f"   Total shapes: {len(self.shapes)}")
        print(f"   File size: {os.path.getsize(filename)} bytes")

        print("\n To use this shape in satellite missions:")
        print("   1. Run Mission.py")
        print("   2. Select Mode 3 (Profile Following)")
        print("   3. Choose option 1 (Load from DXF file)")
        print(f"   4. Enter path: {filename}")

    def show_instructions(self):
        """Display usage instructions."""
        print("\n" + "=" * 60)
        print(" DXF SHAPE MAKER - INSTRUCTIONS")
        print("=" * 60)
        print("Drawing Area: 3m × 3m (±1.5m from center)")
        print("Grid Resolution: 10cm (when snap enabled)")
        print()
        print("MOUSE CONTROLS:")
        print("  Left Click  : Add point / Set circle-arc parameters")
        print("  Right Click : Finish current shape")
        print()
        print("KEYBOARD SHORTCUTS:")
        print("  'l' : Line mode (default)")
        print("  'c' : Circle mode (click center, then radius)")
        print("  'a' : Arc mode (click center, start point, end point)")
        print("  's' : Spline mode (smooth curves)")
        print("  'g' : Toggle grid snapping")
        print("  'z' : Undo last point")
        print("  'r' : Reset all shapes")
        print("  'e' : Export to DXF file")
        print("  'q' : Quit")
        print()
        print("SHAPE DRAWING TIPS:")
        print("  • Use grid snapping for precise alignment")
        print("  • Right-click to finish lines/splines")
        print("  • Circles: click center, then edge for radius")
        print("  • Arcs: click center, start angle, end angle")
        print("  • All shapes must fit within the 3m × 3m area")
        print("=" * 60)

    def run(self):
        """Start the interactive shape maker."""
        print(" Starting DXF Shape Maker...")
        print("Close the window or press 'q' to quit.")

        # Enable interactive mode
        plt.ion()

        # Make sure the figure is focused and interactive
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Show the plot
        plt.show(block=True)


def main():
    """Main function to run the DXF Shape Maker."""
    print(" Checking matplotlib backend...")
    print(f"   Current backend: {matplotlib.get_backend()}")

    if not DXF_AVAILABLE:
        print("  Warning: ezdxf not installed. Install with: pip install ezdxf")
        print("   You can still use the tool, but export will be disabled.")

    print(" Testing matplotlib interactivity...")

    try:
        maker = DXFShapeMaker()
        maker.run()
    except Exception as e:
        print(f" Error starting DXF Shape Maker: {e}")
        print("\n Trying alternative backend...")
        try:
            matplotlib.use("Qt5Agg")
            print(f"   Switched to backend: {matplotlib.get_backend()}")
            maker = DXFShapeMaker()
            maker.run()
        except Exception as e2:
            print(f" Qt5Agg backend failed: {e2}")
            print("\n Trying basic backend...")
            try:
                matplotlib.use("Agg")
                print(" Non-interactive backend - drawing will not work")
                print(" Try installing tkinter: " "sudo apt-get install python3-tk (Linux)")
                print(" Or try installing PyQt5: pip install PyQt5")
            except Exception as e3:
                print(f" All backends failed: {e3}")
                print(" Check your matplotlib installation")


if __name__ == "__main__":
    main()
