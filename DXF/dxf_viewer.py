#!/usr/bin/env python3
"""
DXF Viewer for Satellite Mission Planning

Visualize and analyze 2D DXF files for satellite shape-following missions.
Displays shape geometry, dimensions, and provides offset/scaling options.

Visualization features:
- 2D shape rendering with matplotlib
- Dimension analysis (width, height, area)
- Center point calculation and visualization
- Path offset generation for safety margins
- Multiple centering options (origin, none)

Supported DXF entities:
- LWPOLYLINE: Lightweight polylines
- LINE: Straight line segments
- ARC: Circular arc segments
- CIRCLE: Complete circles
- POLYLINE: Basic 2D polylines

Path processing:
- Offset modes: Buffer (parallel offset), Scale (proportional)
- Join styles: Round, mitre, bevel
- Configurable resolution for curved segments
- Automatic bounds checking for 3m × 3m workspace

Usage:
    python3 DXF_Viewer.py --file DXF_Files/Shape1.dxf [--offset 0.5] \
    [--mode buffer|scale] [--join round|mitre|bevel] [--res 24] \
    [--center origin|none] [--save]

Requirements:
- ezdxf: DXF file parsing (pip install ezdxf)
- matplotlib: Visualization
- numpy: Numerical operations
"""


import sys
import math
from pathlib import Path
from typing import List, Tuple

import argparse

try:
    import ezdxf
except ImportError:
    print("Missing dependency: ezdxf. Install with: pip install ezdxf")
    sys.exit(1)

import matplotlib.pyplot as plt
import numpy as np


def units_code_to_name_and_scale(insunits: int) -> Tuple[str, float]:
    """Map INSUNITS code to (name, scale_to_meters).

    Returns a human-readable units name and a factor to convert to meters.
    """
    mapping = {
        0: ("unitless", 1.0),
        1: ("inches", 0.0254),
        2: ("feet", 0.3048),
        3: ("miles", 1609.344),
        4: ("millimeters", 0.001),
        5: ("centimeters", 0.01),
        6: ("meters", 1.0),
        7: ("kilometers", 1000.0),
        8: ("microinches", 0.0000254),
        9: ("mils", 0.0000254),
        10: ("yards", 0.9144),
        11: ("angstroms", 1e-10),
        12: ("nanometers", 1e-9),
        13: ("microns", 1e-6),
        14: ("decimeters", 0.1),
        15: ("decameters", 10.0),
        16: ("hectometers", 100.0),
        17: ("gigameters", 1e9),
        18: ("astronomical units", 1.495978707e11),
        19: ("light years", 9.460730472e15),
        20: ("parsecs", 3.085677581e16),
        21: ("US survey feet", 1200.0 / 3937.0),
    }
    return mapping.get(insunits, ("unitless", 1.0))


def sample_circle(cx: float, cy: float, r: float, segments: int = 256) -> List[Tuple[float, float]]:
    pts = []
    for i in range(segments + 1):
        a = 2 * math.pi * i / segments
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def sample_arc(
    cx: float,
    cy: float,
    r: float,
    start_deg: float,
    end_deg: float,
    segments: int = 128,
) -> List[Tuple[float, float]]:
    # Normalize and ensure correct sweep direction following DXF convention
    s = math.radians(start_deg)
    e = math.radians(end_deg)
    # Choose step based on absolute angle span
    span = e - s
    # If negative, wrap to positive by adding 2*pi
    if span < 0:
        span += 2 * math.pi
    n = max(2, int(segments * (span / (2 * math.pi))))
    pts = []
    for i in range(n + 1):
        a = s + span * (i / n)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def collect_paths_3d(msp) -> List[List[Tuple[float, float, float]]]:
    """Collect 3D paths from supported entities as polylines of (x,y,z) points."""
    paths: List[List[Tuple[float, float, float]]] = []

    for e in msp.query("LWPOLYLINE"):
        pts: List[Tuple[float, float]] = []
        try:
            segs = list(e.virtual_entities())
        except Exception:
            segs = []
        tol = 1e-9

        def append_pt(p):
            # nonlocal pts  # Removed unused declaration
            if not pts or math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > tol:
                pts.append(p)

        if segs:
            for seg in segs:
                dxft = seg.dxftype()
                if dxft == "LINE":
                    s = (seg.dxf.start.x, seg.dxf.start.y, getattr(seg.dxf.start, "z", 0.0))
                    ept = (seg.dxf.end.x, seg.dxf.end.y, getattr(seg.dxf.end, "z", 0.0))
                    append_pt(s)
                    append_pt(ept)
                elif dxft == "ARC":
                    c = seg.dxf.center
                    r = float(seg.dxf.radius)
                    sa = float(seg.dxf.start_angle)
                    ea = float(seg.dxf.end_angle)
                    for p in sample_arc(c.x, c.y, r, sa, ea, segments=128):
                        append_pt((p[0], p[1], 0.0))
        if not pts:
            # Fallback to raw vertices
            pts = [
                (v[0], v[1], getattr(v, "z", 0.0)) if hasattr(v, "z") else (v[0], v[1], 0.0)
                for v in e.get_points()
            ]
        if e.closed and pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        if len(pts) >= 2:
            paths.append(pts)

    # 2D POLYLINE
    for e in msp.query("POLYLINE"):
        if e.is_2d_polyline or e.is_polygon_mesh or e.is_poly_face_mesh:
            pts = [
                (v.dxf.location.x, v.dxf.location.y, getattr(v.dxf.location, "z", 0.0))
                for v in e.vertices
            ]
        else:
            pts = [
                (v.dxf.location.x, v.dxf.location.y, getattr(v.dxf.location, "z", 0.0))
                for v in e.vertices
            ]
        if e.is_closed and pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        if len(pts) >= 2:
            paths.append(pts)

    # LINE
    for e in msp.query("LINE"):
        pts = [
            (e.dxf.start.x, e.dxf.start.y, getattr(e.dxf.start, "z", 0.0)),
            (e.dxf.end.x, e.dxf.end.y, getattr(e.dxf.end, "z", 0.0)),
        ]
        paths.append(pts)

    # ARC
    for e in msp.query("ARC"):
        c = e.dxf.center
        r = float(e.dxf.radius)
        z = getattr(c, "z", 0.0)
        pts_2d = sample_arc(c.x, c.y, r, float(e.dxf.start_angle), float(e.dxf.end_angle))
        pts = [(p[0], p[1], z) for p in pts_2d]
        paths.append(pts)

    # CIRCLE
    for e in msp.query("CIRCLE"):
        c = e.dxf.center
        r = float(e.dxf.radius)
        z = getattr(c, "z", 0.0)
        pts_2d = sample_circle(c.x, c.y, r)
        pts = [(p[0], p[1], z) for p in pts_2d]
        paths.append(pts)

    return paths


def _is_closed(path: List[Tuple[float, float, float]], tol: float = 1e-6) -> bool:
    return len(path) > 2 and math.hypot(path[0][0] - path[-1][0], path[0][1] - path[-1][1]) <= tol


def _dedup_consecutive(
    path: List[Tuple[float, float, float]], tol: float = 1e-9
) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for p in path:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > tol:
            out.append(p)
    return out


def _polygon_area(path: List[Tuple[float, float, float]]) -> float:
    if not _is_closed(path):
        return 0.0
    x = np.array([p[0] for p in path])
    y = np.array([p[1] for p in path])
    # Area in XY plane
    return 0.5 * abs(float(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])))


def _centroid(path: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Compute a polygon centroid; prefers shapely if available, else mean
    of vertices."""
    if not path:
        return (0.0, 0.0, 0.0)
    try:
        import importlib
        import importlib.util

        geom_spec = importlib.util.find_spec("shapely.geometry")
        if geom_spec is not None and len(path) >= 3:
            shapely_geometry = importlib.import_module("shapely.geometry")
            Polygon = getattr(shapely_geometry, "Polygon")
            # Use 2D projection for shapely validity check
            pts_2d = [(p[0], p[1]) for p in path]
            poly = Polygon(pts_2d if _is_closed(path) else (pts_2d + [pts_2d[0]]))
            if not poly.is_valid:
                poly = poly.buffer(0)
            c = poly.centroid
            # Return mean Z
            z_mean = sum(p[2] for p in path) / len(path)
            return (float(c.x), float(c.y), z_mean)
    except Exception:
        pass
    pts = path[:-1] if _is_closed(path) and len(path) > 1 else path
    ax = float(np.mean([p[0] for p in pts]))
    ay = float(np.mean([p[1] for p in pts]))
    az = float(np.mean([p[2] for p in pts]))
    return (ax, ay, az)


def _translate_path(
    path: List[Tuple[float, float, float]], dx: float, dy: float
) -> List[Tuple[float, float, float]]:
    return [(p[0] + dx, p[1] + dy, p[2] if len(p) > 2 else 0.0) for p in path]


def _translate_paths(
    paths: List[List[Tuple[float, float, float]]], dx: float, dy: float
) -> List[List[Tuple[float, float, float]]]:
    return [[(p[0] + dx, p[1] + dy, p[2] if len(p) > 2 else 0.0) for p in poly] for poly in paths]


def extract_boundary_polygon(msp) -> List[Tuple[float, float]]:
    """Extract the most likely boundary polygon: choose the largest closed
    loop by area.

    Handles: LWPOLYLINE (with bulges via virtual_entities), POLYLINE,
    CIRCLE, ARC, LINE stitching fallback.
    Returns path in native DXF units.
    """
    closed_polys: List[List[Tuple[float, float]]] = []
    open_paths: List[List[Tuple[float, float]]] = []

    for entity in msp:
        dxft = entity.dxftype()
        if dxft == "LINE":
            open_paths.append(
                [
                    (entity.dxf.start.x, entity.dxf.start.y),
                    (entity.dxf.end.x, entity.dxf.end.y),
                ]
            )
        elif dxft == "LWPOLYLINE":
            pts: List[Tuple[float, float]] = []
            try:
                segs = list(entity.virtual_entities())
            except Exception:
                segs = []

            def ap(p):
                # nonlocal pts  # Removed unused declaration
                if not pts or (abs(p[0] - pts[-1][0]) > 1e-9 or abs(p[1] - pts[-1][1]) > 1e-9):
                    pts.append((float(p[0]), float(p[1])))

            for seg in segs:
                sdx = seg.dxftype()
                if sdx == "LINE":
                    ap((seg.dxf.start.x, seg.dxf.start.y))
                    ap((seg.dxf.end.x, seg.dxf.end.y))
                elif sdx == "ARC":
                    c = seg.dxf.center
                    r = float(seg.dxf.radius)
                    sa = float(seg.dxf.start_angle)
                    ea = float(seg.dxf.end_angle)
                    for p in sample_arc(c.x, c.y, r, sa, ea, segments=90):
                        ap(p)
            if not pts:
                pts = [(v[0], v[1]) for v in entity.get_points()]
            closed_flag = bool(getattr(entity, "closed", False))
            if closed_flag or (pts and _is_closed(pts)):
                if pts and pts[0] != pts[-1]:
                    pts = pts + [pts[0]]
                closed_polys.append(_dedup_consecutive(pts))
            else:
                open_paths.append(_dedup_consecutive(pts))
        elif dxft == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
            if getattr(entity, "is_closed", False) and pts:
                if pts[0] != pts[-1]:
                    pts = pts + [pts[0]]
                closed_polys.append(_dedup_consecutive(pts))
            else:
                open_paths.append(_dedup_consecutive(pts))
        elif dxft == "CIRCLE":
            c = entity.dxf.center
            r = float(entity.dxf.radius)
            pts = sample_circle(c.x, c.y, r, segments=256)
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            closed_polys.append(_dedup_consecutive(pts))
        elif dxft == "ARC":
            c = entity.dxf.center
            r = float(entity.dxf.radius)
            pts = sample_arc(
                c.x,
                c.y,
                r,
                float(entity.dxf.start_angle),
                float(entity.dxf.end_angle),
                segments=128,
            )
            open_paths.append(_dedup_consecutive(pts))

    # If any closed polygons, choose the largest area
    closed_polys = [p if _is_closed(p) else (p + [p[0]] if p else p) for p in closed_polys]
    closed_polys = [_dedup_consecutive(p) for p in closed_polys if len(p) >= 4]
    if closed_polys:
        return max(closed_polys, key=_polygon_area)

    # Fallback: stitch open paths by matching endpoints within tolerance
    paths = [p[:] for p in open_paths if len(p) >= 2]
    tol = 1e-6
    changed = True
    while changed and len(paths) > 1:
        changed = False
        for i in range(len(paths)):
            if changed:
                break
            for j in range(i + 1, len(paths)):
                A = paths[i]
                B = paths[j]
                a0, a1 = np.array(A[0]), np.array(A[-1])
                b0, b1 = np.array(B[0]), np.array(B[-1])
                if np.linalg.norm(a1 - b0) <= tol:
                    merged = A + B[1:]
                elif np.linalg.norm(a0 - b1) <= tol:
                    merged = B + A[1:]
                elif np.linalg.norm(a1 - b1) <= tol:
                    merged = A + list(reversed(B[:-1]))
                elif np.linalg.norm(a0 - b0) <= tol:
                    merged = list(reversed(A[1:])) + B
                else:
                    continue
                paths[i] = _dedup_consecutive(merged)
                paths.pop(j)
                changed = True
                break
    candidates: List[List[Tuple[float, float]]] = []
    for cand_path in paths:
        if len(cand_path) < 3:
            continue
        if (
            not _is_closed(cand_path)
            and math.hypot(cand_path[0][0] - cand_path[-1][0], cand_path[0][1] - cand_path[-1][1])
            <= 1e-6
        ):
            cand_path = cand_path + [cand_path[0]]
        candidates.append(cand_path)
    closed_candidates = [p for p in candidates if _is_closed(p)]
    if closed_candidates:
        return max(closed_candidates, key=_polygon_area)
    return max(candidates, key=lambda p: len(p)) if candidates else []


def sanitize_boundary(
    boundary: List[Tuple[float, float, float]], units_to_m: float
) -> List[Tuple[float, float, float]]:
    """Ensure boundary is a valid closed exterior ring.

    - Closes the ring if needed
    - If Shapely is available: fixes self-intersections via buffer(0)
      and returns the largest exterior ring
    Returns in native DXF units.
    """
    if not boundary:
        return boundary
    ring = boundary[:]
    if not _is_closed(ring):
        ring = ring + [ring[0]]
    to_m = units_to_m if units_to_m > 0 else 1.0
    try:
        import importlib
        import importlib.util

        geom_spec = importlib.util.find_spec("shapely.geometry")
        if geom_spec is None:
            return ring
        shapely_geometry = importlib.import_module("shapely.geometry")
        # shapely_ops = importlib.import_module("shapely.ops")  # Unused
        Polygon = getattr(shapely_geometry, "Polygon")
        MultiPolygon = getattr(shapely_geometry, "MultiPolygon", None)
        # Use X, Y for shapely polygon
        poly = Polygon([(p[0] * to_m, p[1] * to_m) for p in ring])
        if not poly.is_valid:
            poly = poly.buffer(0)
        # If MultiPolygon, pick the largest by area
        if MultiPolygon and isinstance(poly, MultiPolygon):
            parts = list(poly.geoms)
            if not parts:
                return ring
            poly = max(parts, key=lambda g: g.area)
        if poly.is_empty:
            return ring
        exterior = list(poly.exterior.coords)
        from_m = 1.0 / to_m
        return [(float(x) * from_m, float(y) * from_m, 0.0) for x, y in exterior]
    except Exception:
        return ring


def make_offset_path(
    points: List[Tuple[float, float]],
    offset_distance_m: float,
    units_to_m: float,
    join: str = "round",
    resolution: int = 24,
    mode: str = "buffer",
) -> List[Tuple[float, float]]:
    """Create an outward offset path at distance (meters).

    Points are in native DXF units.
    Returns path in native DXF units for plotting overlay.
    """
    if len(points) < 3:
        return points
    # Ensure closed
    pts = points[:]
    if not _is_closed(pts):
        pts = pts + [pts[0]]

    to_m = units_to_m
    if to_m <= 0:
        to_m = 1.0
    pts_m = [(p[0] * to_m, p[1] * to_m) for p in pts]  # Use X,Y for offset logic

    # Mode selection
    if mode.lower() == "scale":
        # Centroid-based uniform scaling keeps the centroid identical
        c = np.mean(np.array(pts_m[:-1]), axis=0)
        radii = [float(np.linalg.norm(np.array(p) - c)) for p in pts_m[:-1]]
        avg_r = max(1e-6, float(np.mean(radii)))
        scale = 1.0 + (offset_distance_m / avg_r)
        out_m = []
        for x, y in pts_m:
            v = np.array([x, y]) - c
            p = c + scale * v
            out_m.append((float(p[0]), float(p[1])))
        if not _is_closed(out_m):
            out_m.append(out_m[0])
    else:
        # Buffer-based true offset
        try:
            import importlib
            import importlib
            import importlib.util

            geom_spec = importlib.util.find_spec("shapely.geometry")
            if geom_spec is None:
                raise ImportError("shapely not installed")
            shapely_geometry = importlib.import_module("shapely.geometry")
            Polygon = getattr(shapely_geometry, "Polygon")
            poly = Polygon(pts_m)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                raise ValueError("empty polygon")
            join_map = {"round": 1, "mitre": 2, "miter": 2, "bevel": 3}
            join_style = join_map.get(join.lower(), 1)
            buffered = poly.buffer(
                offset_distance_m,
                join_style=join_style,
                mitre_limit=5.0,
                resolution=max(4, int(resolution)),
            )
            exterior = list(buffered.exterior.coords)
            out_m = [(float(x), float(y)) for x, y in exterior]
        except Exception:
            c = np.mean(np.array(pts_m[:-1]), axis=0)
            radii = [float(np.linalg.norm(np.array(p) - c)) for p in pts_m[:-1]]
            avg_r = max(1e-6, float(np.mean(radii)))
            scale = 1.0 + (offset_distance_m / avg_r)
            out_m = []
            for x, y in pts_m:
                v = np.array([x, y]) - c
                p = c + scale * v
                out_m.append((float(p[0]), float(p[1])))
            if not _is_closed(out_m):
                out_m.append(out_m[0])

    from_m = 1.0 / to_m
    return [(x * from_m, y * from_m, 0.0) for x, y in out_m]


def compute_bounds(
    paths: List[List[Tuple[float, float]]],
) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for poly in paths:
        for p in poly:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
    if not xs:
        return 0.0, 0.0, 0.0, 0.0
    return min(xs), min(ys), max(xs), max(ys)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "View a DXF and report shape size, matching Mission 5 transform " "+ offset pipeline"
        )
    )
    parser.add_argument("--file", "-f", type=str, help="Path to DXF file")
    parser.add_argument("--save", action="store_true", help="Save a PNG next to the DXF")
    parser.add_argument(
        "--offset",
        type=float,
        default=0.5,
        help="Offset distance in meters for shape path overlay (default: 0.6)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="buffer",
        choices=["buffer", "scale"],
        help=("Offset mode: buffer (true offset) or scale (uniform scaling " "about centroid)"),
    )
    parser.add_argument(
        "--join",
        type=str,
        default="round",
        choices=["round", "mitre", "miter", "bevel"],
        help="Join style for offset path (default: round)",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=24,
        help=(
            "Offset circle approximation resolution (segments per quarter " "circle, default: 24)"
        ),
    )
    parser.add_argument(
        "--center",
        type=str,
        default="none",
        choices=["origin", "none"],
        help=("Optional viewer-only recentering; Mission 5 does not recenter " "(default: none)"),
    )
    parser.add_argument(
        "--shape-cx",
        type=float,
        default=0.0,
        help="Shape center X in meters (Mission 5 style)",
    )
    parser.add_argument(
        "--shape-cy",
        type=float,
        default=0.0,
        help="Shape center Y in meters (Mission 5 style)",
    )
    parser.add_argument(
        "--rot-deg",
        type=float,
        default=0.0,
        help="Shape rotation in degrees (Mission 5 style)",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        help=(
            "Simple mode: scale boundary about origin by this factor; ignores " "mission transforms"
        ),
    )
    args = parser.parse_args()

    dxf_path = args.file
    if not dxf_path:
        try:
            dxf_path = input("Enter path to DXF file: ").strip()
        except KeyboardInterrupt:
            return 1

    path = Path(dxf_path)
    if not path.exists() or path.suffix.lower() != ".dxf":
        print(f"DXF not found or wrong extension: {path}")
        return 1

    # Load DXF
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    # Units
    insunits = int(doc.header.get("$INSUNITS", 0))
    units_name, to_m = units_code_to_name_and_scale(insunits)

    # Collect paths and bounds
    paths = collect_paths_3d(msp)
    if not paths:
        print("No supported 2D entities found (LINE, LWPOLYLINE, POLYLINE, " "ARC, CIRCLE).")
        return 1

    xmin, ymin, xmax, ymax = compute_bounds(paths)
    width = xmax - xmin
    height = ymax - ymin
    width_m = width * to_m
    height_m = height * to_m

    print("DXF:", path)
    print(f"Entities collected: {len(paths)} polylines/curves")
    print(f"Units: {units_name} (INSUNITS={insunits})")
    print(f"Size: width={width:.4f} {units_name}, " f"height={height:.4f} {units_name}")
    if units_name != "meters":
        print(f"      width={width_m:.4f} m, height={height_m:.4f} m")

    # Try to extract a boundary
    boundary = extract_boundary_polygon(msp)
    boundary = sanitize_boundary(boundary, to_m)
    boundary_m = [(x * to_m, y * to_m) for (x, y) in boundary] if boundary else []
    paths_m = [[(p[0] * to_m, p[1] * to_m) for p in poly] for poly in paths]

    # Choose pipeline: simple origin-based scaling vs. Mission-style transform
    if args.scale_factor is not None:
        transformed_shape_m = boundary_m[:]
        transformed_paths_m = [poly[:] for poly in paths_m]
    else:
        # Mission 5 transform: rotate then translate to user center
        rot_rad = math.radians(float(args.rot_deg))
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)

        def do_transform(pts):
            out = []
            for p in pts:
                x = p[0]
                y = p[1]
                z = p[2] if len(p) > 2 else 0.0
                xr = x * cos_r - y * sin_r
                yr = x * sin_r + y * cos_r
                out.append((xr + args.shape_cx, yr + args.shape_cy, z))
            return out

        transformed_shape_m = do_transform(boundary_m) if boundary_m else []
        transformed_paths_m = [do_transform(poly) for poly in paths_m]

    # Optional viewer-only recentering after Mission-style transform
    if args.center == "origin" and transformed_shape_m:
        cx, cy = _centroid(transformed_shape_m)
        transformed_shape_m = _translate_path(transformed_shape_m, -cx, -cy)
        transformed_paths_m = _translate_paths(transformed_paths_m, -cx, -cy)

    offset_pts = []
    # base_len_m = None  # Unused
    # offset_len_m = None  # Unused
    if transformed_shape_m:
        # base_len_m would be calculated here
        pass
        try:
            if args.scale_factor is not None:
                # Simple origin-based scaling
                sf = float(args.scale_factor)
                offset_pts = [(x * sf, y * sf) for (x, y) in transformed_shape_m]
                if len(offset_pts) >= 2 and (
                    abs(offset_pts[0][0] - offset_pts[-1][0]) > 1e-9
                    or abs(offset_pts[0][1] - offset_pts[-1][1]) > 1e-9
                ):
                    offset_pts.append(offset_pts[0])
            else:
                offset_pts = make_offset_path(
                    transformed_shape_m,
                    args.offset,
                    1.0,
                    join=args.join,
                    resolution=args.res,
                    mode=args.mode,
                )
            # offset_len_m would be calculated here
        except Exception as e:
            print(f"Offset generation failed: {e}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for poly in transformed_paths_m:
        if len(poly) >= 2:
            xs_m = [p[0] for p in poly]
            ys_m = [p[1] for p in poly]
            # Handle potential missing Z in poly if transforms stripped it (need checks)
            # But we defined everything as 3 tuples now.
            zs_m = [p[2] if len(p) > 2 else 0.0 for p in poly]
            ax.plot(
                xs_m,
                ys_m,
                zs_m,
                color="#1f77b4",
                linewidth=1.5,
                alpha=0.65,
                label="All entities" if poly is paths[0] else None,
            )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Highlight boundary and centroid
    b_cent = None
    if transformed_shape_m:
        bx = [p[0] for p in transformed_shape_m]
        by = [p[1] for p in transformed_shape_m]
        ax.plot(bx, by, color="#2ca02c", linewidth=2.5, label="Outline")
        try:
            import importlib
            import importlib.util

            geom_spec = importlib.util.find_spec("shapely.geometry")
            if geom_spec is not None:
                shapely_geometry = importlib.import_module("shapely.geometry")
                Polygon = getattr(shapely_geometry, "Polygon")
                bpoly = Polygon(list(zip(bx, by)))
                if not bpoly.is_valid:
                    bpoly = bpoly.buffer(0)
                if not bpoly.is_empty:
                    c = bpoly.centroid
                    b_cent = (float(c.x), float(c.y))
        except Exception:
            pass
        if b_cent is None and bx and by:
            b_cent = (float(np.mean(bx[:-1])), float(np.mean(by[:-1])))
        if b_cent:
            ax.plot(
                [b_cent[0]],
                [b_cent[1]],
                marker="x",
                color="#2ca02c",
                markersize=8,
                label=None,
            )
    # Overlay offset path
    # Offset path and centroid
    o_cent = None
    if offset_pts:
        ox = [p[0] for p in offset_pts]
        oy = [p[1] for p in offset_pts]
        ax.plot(
            ox,
            oy,
            color="#ff7f0e",
            linewidth=2.0,
            linestyle="--",
            label=f"Offset {args.offset:.2f} m",
        )
        try:
            import importlib
            import importlib.util

            geom_spec = importlib.util.find_spec("shapely.geometry")
            if geom_spec is not None:
                shapely_geometry = importlib.import_module("shapely.geometry")
                Polygon = getattr(shapely_geometry, "Polygon")
                opoly = Polygon(list(zip(ox, oy)))
                if not opoly.is_valid:
                    opoly = opoly.buffer(0)
                if not opoly.is_empty:
                    c = opoly.centroid
                    o_cent = (float(c.x), float(c.y))
        except Exception:
            pass
        if o_cent is None and ox and oy:
            o_cent = (float(np.mean(ox[:-1])), float(np.mean(oy[:-1])))
        if o_cent:
            ax.plot(
                [o_cent[0]],
                [o_cent[1]],
                marker="+",
                color="#ff7f0e",
                markersize=8,
                label=None,
            )

    ax.set_aspect("equal", adjustable="box")
    # Fixed view window: -3m to 3m
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if args.save:
        out_png = path.with_suffix(".png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Saved preview: {out_png}")

    try:
        plt.show()
    except Exception:
        # In headless contexts, just save a fallback PNG
        out_png = path.with_suffix(".png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Headless environment detected; saved preview: {out_png}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
