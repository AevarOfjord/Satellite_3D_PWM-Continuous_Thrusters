#!/usr/bin/env python3
"""
HTML replay viewer with timeline slider for saved simulation data.

Usage:
  python3 mujoco_viewer_html.py
  python3 mujoco_viewer_html.py Data/Simulation/07-01-2026_14-39-43
  python3 mujoco_viewer_html.py --csv Data/Simulation/07-01-2026_14-39-43/control_data.csv
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.satellite_control.config import SatelliteConfig


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


def _preprocess_argv(argv: List[str]) -> List[str]:
    if len(argv) == 2:
        candidate = argv[1]
        if candidate.startswith("-") and not candidate.startswith("--"):
            if "/" in candidate or "\\" in candidate:
                return [argv[0], candidate[1:]]
    return argv


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


def _rotation_matrix_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _cube_vertices(half_size: float) -> np.ndarray:
    return np.array(
        [
            [-half_size, -half_size, -half_size],
            [half_size, -half_size, -half_size],
            [half_size, half_size, -half_size],
            [-half_size, half_size, -half_size],
            [-half_size, -half_size, half_size],
            [half_size, -half_size, half_size],
            [half_size, half_size, half_size],
            [-half_size, half_size, half_size],
        ],
        dtype=float,
    )


def _cube_triangles() -> Tuple[List[int], List[int], List[int]]:
    # 12 triangles (two per face)
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 3, 5, 7, 4, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 2, 6, 6, 7, 1, 6, 5, 7, 6, 4, 7]
    return i, j, k


def main() -> int:
    argv = _preprocess_argv(sys.argv)
    parser = argparse.ArgumentParser(description="Generate HTML replay viewer.")
    parser.add_argument("run_dir", nargs="?", help="Simulation run directory.")
    parser.add_argument("--csv", type=str, help="Path to control_data.csv or physics_data.csv.")
    parser.add_argument(
        "--source",
        choices=["control", "physics"],
        default="control",
        help="Which CSV to use when run_dir is provided.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for playback.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=600,
        help="Maximum frames to include (auto adjusts stride).",
    )
    args = parser.parse_args(argv[1:])

    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is not installed. Run: venv/bin/pip install plotly")
        return 1

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

    time_col = "Control_Time" if "Control_Time" in df.columns else "Time"
    time_vals = (
        _to_numeric(df[time_col], default=0.0)
        if time_col in df.columns
        else np.arange(len(df), dtype=float)
    )

    x = _to_numeric(df.get("Current_X", pd.Series([0.0] * len(df))))
    y = _to_numeric(df.get("Current_Y", pd.Series([0.0] * len(df))))
    z = _to_numeric(df.get("Current_Z", pd.Series([0.0] * len(df))))

    roll = _to_numeric(df.get("Current_Roll", pd.Series([0.0] * len(df))))
    pitch = _to_numeric(df.get("Current_Pitch", pd.Series([0.0] * len(df))))
    yaw = _to_numeric(df.get("Current_Yaw", pd.Series([0.0] * len(df))))

    command_vec = df.get("Command_Vector", pd.Series([""] * len(df)))

    stride = max(1, int(args.stride))
    if len(x) > args.max_frames:
        stride = max(stride, int(np.ceil(len(x) / args.max_frames)))

    thruster_ids = sorted(SatelliteConfig.THRUSTER_POSITIONS.keys())
    thruster_body = np.array(
        [SatelliteConfig.THRUSTER_POSITIONS[i] for i in thruster_ids], dtype=float
    )
    half_size = float(SatelliteConfig.SATELLITE_SIZE) / 2.0
    cube = _cube_vertices(half_size)
    tri_i, tri_j, tri_k = _cube_triangles()

    frames = []
    steps = []
    active_color = "#ff3b30"
    inactive_color = "#404040"

    for idx in range(0, len(x), stride):
        frame_name = str(idx)
        rot = _rotation_matrix_xyz(roll[idx], pitch[idx], yaw[idx])
        pos = np.array([x[idx], y[idx], z[idx]])

        cube_world = cube @ rot.T + pos
        thr_world = thruster_body @ rot.T + pos

        cmd_vec = _parse_command_vector(command_vec.iloc[idx])
        active = {i + 1 for i, val in enumerate(cmd_vec) if val > 0.5}
        colors = [
            active_color if thruster_id in active else inactive_color
            for thruster_id in thruster_ids
        ]
        sizes = [6 if thruster_id in active else 3 for thruster_id in thruster_ids]

        frames.append(
            {
                "name": frame_name,
                "data": [
                    {
                        "type": "mesh3d",
                        "x": cube_world[:, 0],
                        "y": cube_world[:, 1],
                        "z": cube_world[:, 2],
                        "i": tri_i,
                        "j": tri_j,
                        "k": tri_k,
                    },
                    {
                        "type": "scatter3d",
                        "x": thr_world[:, 0],
                        "y": thr_world[:, 1],
                        "z": thr_world[:, 2],
                        "mode": "markers",
                        "marker": {"size": sizes, "color": colors},
                    },
                ],
            }
        )
        steps.append(
            {
                "label": f"{time_vals[idx]:.1f}s",
                "method": "animate",
                "args": [[frame_name], {"mode": "immediate", "frame": {"duration": 0}}],
            }
        )

    rot0 = _rotation_matrix_xyz(roll[0], pitch[0], yaw[0])
    pos0 = np.array([x[0], y[0], z[0]])
    cube0 = cube @ rot0.T + pos0
    thr0 = thruster_body @ rot0.T + pos0

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=cube0[:, 0],
                y=cube0[:, 1],
                z=cube0[:, 2],
                i=tri_i,
                j=tri_j,
                k=tri_k,
                color="#5c5c5c",
                opacity=1.0,
                flatshading=True,
                name="Satellite",
            ),
            go.Scatter3d(
                x=thr0[:, 0],
                y=thr0[:, 1],
                z=thr0[:, 2],
                mode="markers",
                marker={"size": 3, "color": inactive_color},
                name="Thrusters",
            ),
        ],
        frames=frames,
    )

    fig.update_layout(
        title="Interactive 3D Replay",
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "bgcolor": "black",
            "aspectmode": "data",
        },
        paper_bgcolor="black",
        plot_bgcolor="black",
        font={"color": "#f5f5f5"},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 30, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "steps": steps,
                "currentvalue": {"prefix": "Time: "},
            }
        ],
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    output_dir = run_dir / "Plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trajectory_3d_replay.html"
    fig.write_html(output_path, include_plotlyjs="cdn")

    print(f"Saved HTML replay viewer to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
