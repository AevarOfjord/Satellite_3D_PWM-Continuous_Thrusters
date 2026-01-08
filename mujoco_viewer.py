#!/usr/bin/env python3
"""
MuJoCo replay viewer with timeline slider for saved simulation data.

Usage:
  Mac:
  MUJOCO_GL=glfw venv/bin/mjpython mujoco_viewer.py
  MUJOCO_GL=glfw venv/bin/mjpython mujoco_viewer.py Data/Simulation/07-01-2026_14-25-34
  
  Windows:
  python mujoco_viewer.py
  python mujoco_viewer.py Data/Simulation/07-01-2026_14-25-34
  python mujoco_viewer.py -Data/Simulation/07-01-2026_14-25-34
"""

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
from mujoco import viewer as mujoco_viewer
import pandas as pd

from src.satellite_control.utils.orientation_utils import euler_xyz_to_quat_wxyz


DEFAULT_THRUSTER_COLORS: Dict[int, Tuple[float, float, float, float]] = {
    i: (0.0, 0.45, 1.0, 0.35) for i in range(1, 13)
}
TARGET_COLOR = (0.2, 1.0, 0.4, 0.9)
TRAJ_COLOR = (0.0, 0.8, 1.0, 0.8)
FONT_5X7 = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10001", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "V": ["10001", "10001", "10001", "10001", "01010", "01010", "00100"],
    ":": ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
    ",": ["00000", "00000", "00000", "00000", "00100", "00100", "01000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00100", "00100"],
    "-": ["00000", "00000", "00000", "01110", "00000", "00000", "00000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


def _latest_run_dir(base: Path) -> Path:
    if not base.exists():
        raise FileNotFoundError(f"Base path not found: {base}")
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run folders in {base}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _resolve_csv(run_dir: Path, source: str) -> Path:
    if source == "physics":
        csv_path = run_dir / "physics_data.csv"
    else:
        csv_path = run_dir / "control_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return csv_path


def _to_numeric(series: pd.Series, default: float = 0.0) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    values = values.fillna(default)
    return values.to_numpy(dtype=float)


def _parse_command_vector(value: object) -> np.ndarray:
    if value is None:
        return np.zeros(0)
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.array(value, dtype=float)
    text = str(value).strip().strip("[]")
    if not text:
        return np.zeros(0)
    try:
        return np.array([float(x) for x in text.replace(",", " ").split()], dtype=float)
    except Exception:
        return np.zeros(0)


def _preprocess_argv(argv: list) -> list:
    if len(argv) == 2:
        candidate = argv[1]
        if candidate.startswith("-") and not candidate.startswith("--"):
            if "/" in candidate or "\\" in candidate:
                return [argv[0], candidate[1:]]
    return argv


def main() -> int:
    argv = _preprocess_argv(sys.argv)

    parser = argparse.ArgumentParser(description="Replay MuJoCo simulation with slider.")
    parser.add_argument("run_dir", nargs="?", help="Simulation run directory.")
    parser.add_argument("--csv", type=str, help="Path to control_data.csv or physics_data.csv.")
    parser.add_argument(
        "--source",
        choices=["control", "physics"],
        default="control",
        help="Which CSV to use when run_dir is provided.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/satellite_3d.xml",
        help="MuJoCo model XML to render.",
    )
    args = parser.parse_args(argv[1:])

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        run_dir = csv_path.parent
    else:
        base = Path("Data") / "Simulation"
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            run_dir = _latest_run_dir(base)
        csv_path = _resolve_csv(run_dir, args.source)

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise RuntimeError(f"Empty CSV: {csv_path}")

    time_col = None
    if "Control_Time" in df.columns:
        time_col = "Control_Time"
    elif "Time" in df.columns:
        time_col = "Time"

    if time_col:
        time_vals = _to_numeric(df[time_col], default=0.0)
    elif "Actual_Time_Interval" in df.columns:
        dt_guess = float(
            pd.to_numeric(df["Actual_Time_Interval"], errors="coerce").fillna(0.0).mean()
        )
        if dt_guess <= 0:
            dt_guess = 0.05
        time_vals = np.arange(len(df)) * dt_guess
    else:
        time_vals = np.arange(len(df), dtype=float)

    diffs = np.diff(time_vals)
    diffs = diffs[diffs > 0]
    frame_dt = float(np.median(diffs)) if diffs.size > 0 else 0.05

    x = _to_numeric(df.get("Current_X", pd.Series([0.0] * len(df))))
    y = _to_numeric(df.get("Current_Y", pd.Series([0.0] * len(df))))
    z = _to_numeric(df.get("Current_Z", pd.Series([0.0] * len(df))))

    roll = _to_numeric(df.get("Current_Roll", pd.Series([0.0] * len(df))))
    pitch = _to_numeric(df.get("Current_Pitch", pd.Series([0.0] * len(df))))
    yaw = _to_numeric(df.get("Current_Yaw", pd.Series([0.0] * len(df))))

    vx = _to_numeric(df.get("Current_VX", pd.Series([0.0] * len(df))))
    vy = _to_numeric(df.get("Current_VY", pd.Series([0.0] * len(df))))
    vz = _to_numeric(df.get("Current_VZ", pd.Series([0.0] * len(df))))
    wx = _to_numeric(df.get("Current_WX", pd.Series([0.0] * len(df))))
    wy = _to_numeric(df.get("Current_WY", pd.Series([0.0] * len(df))))
    wz = _to_numeric(df.get("Current_WZ", pd.Series([0.0] * len(df))))

    target_x = float(df.get("Target_X", pd.Series([0.0])).iloc[0])
    target_y = float(df.get("Target_Y", pd.Series([0.0])).iloc[0])
    target_z = float(df.get("Target_Z", pd.Series([0.0])).iloc[0])
    traj = np.column_stack([x, y, z])

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "satellite_geom")
    if geom_id != -1:
        model.geom_rgba[geom_id, 3] = 0.75
    for geom_idx in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_idx)
        if name and ("floor" in name.lower() or "ground" in name.lower()):
            model.geom_rgba[geom_idx, 3] = 0.0
        elif model.geom_type[geom_idx] == mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_rgba[geom_idx, 3] = 0.0

    state = {
        "play": False,
        "idx": 0,
        "speed": 1.0,
        "running": True,
        "request_seek": False,
        "active_ids": set(),
    }
    slider_updating = {"active": False}
    viewer_handle = None
    frame_slider = None
    hud_cache = {"size": None, "image": None}

    def _get_viewport():
        if viewer_handle is None:
            return None
        viewport = getattr(viewer_handle, "viewport", None)
        if viewport is not None:
            return viewport
        sim_ref = getattr(viewer_handle, "_sim", None)
        sim = None
        if callable(sim_ref):
            try:
                sim = sim_ref()
            except Exception:
                sim = None
        elif sim_ref is not None:
            try:
                sim = sim_ref()
            except Exception:
                sim = None
        if sim is not None:
            return getattr(sim, "viewport", None)
        return None

    def _draw_text(img: np.ndarray, text: str, x: int, y: int, scale: int) -> None:
        height, width, _ = img.shape
        color = np.array([235, 240, 248], dtype=np.uint8)
        line_height = (7 + 2) * scale
        char_width = (5 + 1) * scale
        cursor_y = y
        for line in text.splitlines():
            cursor_x = x
            for ch in line:
                pattern = FONT_5X7.get(ch.upper())
                if pattern is not None:
                    for row, bits in enumerate(pattern):
                        for col, bit in enumerate(bits):
                            if bit != "1":
                                continue
                            px = cursor_x + col * scale
                            py = cursor_y + row * scale
                            if px >= width or py >= height:
                                continue
                            img[py : py + scale, px : px + scale] = color
                cursor_x += char_width
                if cursor_x >= width:
                    break
            cursor_y += line_height
            if cursor_y >= height:
                break

    def _update_overlay(idx: int) -> None:
        if viewer_handle is None:
            return
        scn = viewer_handle.user_scn
        with viewer_handle.lock():
            scn.ngeom = 0
            max_geom = len(scn.geoms)

            def add_geom():
                if scn.ngeom >= max_geom:
                    return None
                geom = scn.geoms[scn.ngeom]
                scn.ngeom += 1
                return geom

            # Target sphere
            geom = add_geom()
            if geom is not None:
                size = np.array([0.05, 0.0, 0.0], dtype=float)
                pos = np.array([target_x, target_y, target_z], dtype=float)
                mat = np.eye(3, dtype=float).reshape(9)
                rgba = np.array(TARGET_COLOR, dtype=np.float32)
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    size,
                    pos,
                    mat,
                    rgba,
                )

            # Trajectory line segments (downsampled)
            if idx > 0:
                max_segments = 200
                step = max(1, int(np.ceil(idx / max_segments)))
                for j in range(0, idx, step):
                    p1 = traj[j]
                    p2 = traj[min(j + step, idx)]
                    geom = add_geom()
                    if geom is None:
                        break
                    size = np.zeros(3, dtype=float)
                    pos = np.zeros(3, dtype=float)
                    mat = np.eye(3, dtype=float).reshape(9)
                    rgba = np.array(TRAJ_COLOR, dtype=np.float32)
                    mujoco.mjv_initGeom(
                        geom,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        size,
                        pos,
                        mat,
                        rgba,
                    )
                    mujoco.mjv_connector(
                        geom,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        2.0,
                        p1.astype(float),
                        p2.astype(float),
                    )

    def _update_hud(idx: int, active_ids: set) -> None:
        if viewer_handle is None:
            return
        viewport = _get_viewport()
        if viewport is None or viewport.width <= 0 or viewport.height <= 0:
            return
        panel_w = max(220, int(viewport.width * 0.24))
        panel_h = max(130, int(viewport.height * 0.18))
        margin = max(10, int(viewport.width * 0.01))
        left = max(0, viewport.width - panel_w - margin)
        bottom = max(0, viewport.height - panel_h - margin)

        size_key = (panel_w, panel_h)
        if hud_cache["size"] != size_key:
            panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            panel[:] = (12, 14, 18)
            border = max(1, int(panel_h * 0.04))
            panel[:border, :, :] = (28, 32, 40)
            panel[-border:, :, :] = (28, 32, 40)
            panel[:, :border, :] = (28, 32, 40)
            panel[:, -border:, :] = (28, 32, 40)
            hud_cache["size"] = size_key
            hud_cache["image"] = panel

        label_lines = [
            f"Time: {time_vals[idx]:.2f}s",
            f"Pos: {x[idx]:.2f}, {y[idx]:.2f}, {z[idx]:.2f}",
            f"Vel: {vx[idx]:.2f}, {vy[idx]:.2f}, {vz[idx]:.2f}",
            f"Dist: {np.linalg.norm(traj[idx] - np.array([target_x, target_y, target_z])):.2f} m",
            f"Active: {len(active_ids)}",
        ]
        panel = hud_cache["image"].copy()
        pad_x = max(6, int(panel_w * 0.05))
        pad_y = max(6, int(panel_h * 0.08))
        _draw_text(panel, "\n".join(label_lines), pad_x, pad_y, scale=2)
        rect = mujoco.MjrRect(left, bottom, panel_w, panel_h)
        try:
            viewer_handle.set_images((rect, panel))
        except Exception:
            pass

    def set_frame(idx: int, update_slider: bool = False) -> None:
        idx = int(np.clip(idx, 0, len(df) - 1))
        state["idx"] = idx

        quat = euler_xyz_to_quat_wxyz((roll[idx], pitch[idx], yaw[idx]))
        if viewer_handle is not None:
            with viewer_handle.lock():
                data.qpos[0:3] = [x[idx], y[idx], z[idx]]
                data.qpos[3:7] = quat
                data.qvel[0:3] = [vx[idx], vy[idx], vz[idx]]
                data.qvel[3:6] = [wx[idx], wy[idx], wz[idx]]
                mujoco.mj_forward(model, data)
        else:
            data.qpos[0:3] = [x[idx], y[idx], z[idx]]
            data.qpos[3:7] = quat
            data.qvel[0:3] = [vx[idx], vy[idx], vz[idx]]
            data.qvel[3:6] = [wx[idx], wy[idx], wz[idx]]
            mujoco.mj_forward(model, data)

        cmd_vec = _parse_command_vector(df.get("Command_Vector", pd.Series([None])).iloc[idx])
        active_ids = set()
        if cmd_vec.size > 0:
            thruster_count = max(12, cmd_vec.size)
            active_ids = {i + 1 for i, val in enumerate(cmd_vec) if val > 0.5}
            for thruster_id in range(1, thruster_count + 1):
                site_name = f"thruster{thruster_id}"
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                if site_id == -1:
                    continue
                if thruster_id in active_ids:
                    model.site_rgba[site_id] = [1.0, 0.2, 0.2, 1.0]
                else:
                    model.site_rgba[site_id] = list(
                        DEFAULT_THRUSTER_COLORS.get(thruster_id, (0.5, 0.5, 0.5, 0.3))
                    )

        state["active_ids"] = active_ids
        _update_overlay(idx)
        _update_hud(idx, active_ids)
        # Ensure scene overlays are refreshed each frame.
        try:
            viewer_handle.sync()
        except Exception:
            pass

        if update_slider and frame_slider is not None:
            slider_updating["active"] = True
            frame_slider.set_val(idx)
            slider_updating["active"] = False

    def on_slider_change(val: float) -> None:
        if slider_updating["active"]:
            return
        state["play"] = False
        set_frame(int(val), update_slider=False)

    def on_play(_event) -> None:
        state["play"] = not state["play"]

    def on_step(_event) -> None:
        state["play"] = False
        set_frame(state["idx"] + 1, update_slider=True)

    def on_speed_change(val: float) -> None:
        state["speed"] = float(val)

    def on_close(_event) -> None:
        state["running"] = False

    def on_key(key: int) -> None:
        if key in (ord(" "), ord("p"), ord("P")):
            state["play"] = not state["play"]
        elif key in (ord("."), ord("s"), ord("S")):
            state["play"] = False
            set_frame(state["idx"] + 1, update_slider=True)
        elif key in (ord(","), ord("a"), ord("A")):
            state["play"] = False
            set_frame(state["idx"] - 1, update_slider=True)
        elif key in (ord("+"), ord("=")):
            state["speed"] = min(4.0, state["speed"] + 0.1)
        elif key in (ord("-"), ord("_")):
            state["speed"] = max(0.1, state["speed"] - 0.1)
        elif key in (ord("g"), ord("G")):
            state["play"] = False
            state["request_seek"] = True
        elif key in (ord("r"), ord("R")):
            state["play"] = False
            set_frame(0, update_slider=True)

    try:
        viewer_handle = mujoco_viewer.launch_passive(
            model,
            data,
            key_callback=on_key,
            show_left_ui=False,
            show_right_ui=False,
        )
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "mjpython" in msg:
            print(
                "MuJoCo viewer on macOS requires mjpython.\n"
                "Run: venv/bin/mjpython mujoco_viewer.py <run-dir>\n"
                "If the viewer does not open, try: MUJOCO_GL=glfw venv/bin/mjpython "
                "mujoco_viewer.py <run-dir>"
            )
            return 1
        raise

    viewer_handle.cam.lookat[:] = [0.0, 0.0, 0.0]
    viewer_handle.cam.distance = 6.0
    viewer_handle.cam.elevation = 30.0
    viewer_handle.cam.azimuth = 45.0

    use_gui_controls = platform.system() != "Darwin"
    fig = None
    if use_gui_controls:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        plt.ion()
        fig = plt.figure(figsize=(8, 3))
        fig.canvas.manager.set_window_title("MuJoCo Replay Controls")
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.95)

        ax_slider = fig.add_axes([0.08, 0.15, 0.84, 0.05])
        ax_speed = fig.add_axes([0.08, 0.07, 0.55, 0.03])
        ax_play = fig.add_axes([0.68, 0.03, 0.12, 0.08])
        ax_step = fig.add_axes([0.82, 0.03, 0.12, 0.08])

        frame_slider = Slider(
            ax_slider,
            "Frame",
            0,
            len(df) - 1,
            valinit=0,
            valstep=1,
            valfmt="%0.0f",
        )
        speed_slider = Slider(ax_speed, "Speed", 0.1, 4.0, valinit=1.0, valstep=0.1)
        play_btn = Button(ax_play, "Play/Pause")
        step_btn = Button(ax_step, "Step")

        frame_slider.on_changed(on_slider_change)
        speed_slider.on_changed(on_speed_change)
        play_btn.on_clicked(on_play)
        step_btn.on_clicked(on_step)
        fig.canvas.mpl_connect("close_event", on_close)
    else:
        print(
            "Replay controls (macOS): Space/P play-pause, S/. step, A/, back, "
            "+/- speed, G go-to frame"
        )

    set_frame(0, update_slider=True)

    last_tick = time.time()
    while viewer_handle.is_running() and state["running"]:
        if state["request_seek"]:
            try:
                raw = input(f"Go to frame (0-{len(df) - 1}): ").strip()
                if raw:
                    set_frame(int(raw), update_slider=True)
            except Exception:
                pass
            state["request_seek"] = False
        _update_hud(state["idx"], state.get("active_ids", set()))
        if state["play"]:
            now = time.time()
            if now - last_tick >= frame_dt / max(state["speed"], 0.1):
                if state["idx"] < len(df) - 1:
                    set_frame(state["idx"] + 1, update_slider=True)
                else:
                    state["play"] = False
                last_tick = now
        if fig is not None:
            import matplotlib.pyplot as plt

            plt.pause(0.001)
        viewer_handle.sync()

    viewer_handle.close()
    if fig is not None:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
