import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

M_TO_IN = 39.37007874015748

Pt = Tuple[float, float]

def get_track_sections_dict() -> dict[str, dict[str, tuple[float, float]]]:
    """
    Builds dictionary of sections of track.

    Returns:
        dict{dict: (float, float)}
    """

    # -----------------------------
    # Original points (meters)
    # -----------------------------
    sections = {
        "T1": {
            "P_LL": (-0.21152, 5.71897),
            "P_LR": ( 0.40079, 5.71897),
            "P_UL": (-1.1005, 6.57861),
            "P_UR": (-1.1005, 7.185155),
        },
        "T2": {
            "P_LL": (-3.532365, 6.57861),
            "P_LR": (-3.532365, 7.185155),
            "P_UL": (-4.558515, 5.71897),
            "P_UR": (-5.16854, 5.71897),
        },
        "T3": {
            "P_LL": (-4.558515, 0.22063),
            "P_LR": (-5.16854, 0.22063),
            "P_UL": (-3.532365, -0.801585),
            "P_UR": (-3.532365, -1.41194),
        },
        "T4": {
            "P_LL": (-1.1005, -0.801585),
            "P_LR": (-1.1005, -1.41194),
            "P_UL": (-0.21152, 0.22063),
            "P_UR": (0.40079, 0.22063),
        },
        "L1": {
            "P_LL": (-0.21152, 0.22063),
            "P_LR": ( 0.40079, 0.22063),
            "P_UL": (-0.21152, 5.71897),
            "P_UR": ( 0.40079, 5.71897),
        },
        "L2": {
            "P_LL": (-4.55749, 5.71897),
            "P_LR": (-5.16854,  5.71897),
            "P_UL": (-4.55749, 0.22063),
            "P_UR": (-5.16854,  0.22063),
        },
        "S1": {
            "P_LL": (-1.1005, 6.57861),
            "P_LR": (-1.1005, 7.185155),
            "P_UL": (-3.53177, 6.57861),
            "P_UR": (-3.53177, 7.185155),
        },
        "S2": {
            "P_LL": (-3.532365, -0.801585),
            "P_LR": (-3.532365, -1.41194),
            "P_UL": (-1.1005,  -0.801585),
            "P_UR": (-1.1005,  -1.41194),
        },
    }

    return sections

# -----------------------------
# Geometry helpers
# -----------------------------

def transform(pt):
    """
    Coordinate transform
        old: x down, y right
        new: x right, y up
    """
    x_old, y_old = pt
    x_new = y_old
    y_new = -x_old
    return (x_new, y_new)

def cartesian_to_polar(v):
    x, y = v
    r = np.linalg.norm(v)
    theta = np.arctan2(y, x)   # radians
    theta = (theta + 2*np.pi) % (2*np.pi) # limit to 0 - 360
    return r, theta

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def _dist_point_to_segment(p, a, b) -> float:
    """Euclidean distance from point p to line segment a-b."""
    
    p = np.asarray(p, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    ab = b - a
    ab_unit_v = ab/np.linalg.norm(ab)
    proj = np.dot(p - a, ab_unit_v)*ab_unit_v
    
    return float(np.linalg.norm(p - a - proj))

def _dist_point_to_arc(P, C, r) -> float:
    """
    Finds the point on the arc that is closest to P.

    Args:
        P (_type_): _description_
        C (_type_): _description_
        r (_type_): _description_

    Returns:
        float: _description_
    """

    P = np.array(P) # Point vector
    C = np.array(C) # Center-of-arc vector

    P_center_to_point = P - C # Vector from arc center to point

    R = r * (P_center_to_point/np.linalg.norm(P_center_to_point)) # Arc radius vector

    D = R - P_center_to_point # Distance vector.

    return np.linalg.norm(D)

def section_polygon(sec):
    """
    Build a consistent quad polygon from your corner naming.
    Order: UL -> UR -> LR -> LL
    """
    return [sec["P_UL"], sec["P_UR"], sec["P_LR"], sec["P_LL"]]


def point_in_poly(pt, poly):
    """
    Ray-casting point-in-polygon.
    pt: (x,y)
    poly: list of (x,y) in order (clockwise or ccw)
    Returns True if inside or on edge.
    """
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)

    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]

        # On-edge check (with tolerance)
        dx, dy = x1 - x0, y1 - y0
        px, py = x - x0, y - y0
        cross = dx * py - dy * px
        if abs(cross) < 1e-12:
            dot = px * dx + py * dy
            if -1e-12 <= dot <= (dx*dx + dy*dy) + 1e-12:
                return True

        # Ray cast
        if (y0 > y) != (y1 > y):
            xinters = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-18) + x0
            if x <= xinters:
                inside = not inside

    return inside

def point_in_turn(Point, turn_dict, angles):
    """
    Checks whether given "Point" is inside the turn defined by Center, r_inner, r_outer, angle1, angle2

    Args:
        Center (Tuple(float, float)): m 
        Point (Tuple(float, float)): m
        r_inner (float): m
        r_outer (float): m
        angle1 (float): degrees
        angle2 (float): degrees

    Returns:
        boolean: True if point is inside, False otherwise
    """

    Center, r_inner, r_outer = get_turn_center_radii(turn_dict)

    inside = False

    Point = np.array(Point)
    Center = np.array(Center)

    angle1, angle2 = angles
    
    angle1 = np.deg2rad(angle1)
    angle2 = np.deg2rad(angle2)

    # Convert Point to Turn's coordinate frame

    Point_in_turn_frame = Point - Center

    r, theta = cartesian_to_polar(Point_in_turn_frame)

    inside = (r_inner <= r <= r_outer) and (angle1 <= theta <= angle2)

    return inside

def line_intersection(p1, p2, p3, p4, eps=1e-12):
    """
    Returns intersection of two infinite lines.

    Line 1: p1 -> p2
    Line 2: p3 -> p4

    All points are (x, y).
    Returns (x, y) or None if lines are parallel.
    """

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Solve using determinant form
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)

    if abs(denom) < eps:
        return None  # parallel

    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

    return (px, py)

def get_turn_center_radii(turn_dict):
    center = line_intersection(turn_dict["P_LL"], turn_dict["P_LR"],
                               turn_dict["P_UL"], turn_dict["P_UR"])
    
    inner_radius = dist(center, turn_dict["P_LL"])
    outer_radius = dist(center, turn_dict["P_LR"])

    return center, inner_radius, outer_radius


def find_section(pt, sections):
    """
    pt: (x,y)
    sections: your dict {"T1": {"P_LL":..., ...}, ...}

    Returns: section name (e.g., "L1") or None if not found.

    Note: some points near boundaries may belong to multiple quads.
          We resolve by checking turns first (T*), then straights.
    """
    # 1) Check turns using sector geometry
    for turn_name, angles in zip(["T1", "T2", "T3", "T4"], [(270, 360), (0, 90), (90, 180), (180, 270)]):
        inside = point_in_turn(pt, sections[turn_name], angles)
        if inside:
            return turn_name

    # 2) Check straights using polygons
    for name in ["L1", "L2", "S1", "S2"]:
        if point_in_poly(pt, section_polygon(sections[name])):
            return name
        
    return None

def get_in_lane_dist(pt, sections):

    value = np.NaN

    section = find_section(pt, sections)

    if section is not None:
        if section in ["T1", "T2", "T3", "T4"]:
            center, _, outer_radius = get_turn_center_radii(sections[section])
            value = _dist_point_to_arc(pt, center, outer_radius)
            
        elif section in ["L1", "L2", "S1", "S2"]:
            a = sections[section]["P_LR"]
            b = sections[section]["P_UR"]
            
            value = _dist_point_to_segment(pt, a, b)

    return value

def get_y_gt(ground_truth):

    sections_dict = get_track_sections_dict()

    sections_tf = {}

    for name, pts in sections_dict.items():
        sections_tf[name] = {k: transform(v) for k, v in pts.items()}

    y_gt = np.full(len(ground_truth), np.nan)

    for i, (gt_x, gt_y) in enumerate(ground_truth):
        y_gt[i] = get_in_lane_dist((gt_x, gt_y), sections_tf)

    return y_gt*M_TO_IN

    # print(find_section((3.75, -0.06), sections_tf)) # L1
    # print(find_section((6.67, 0.86), sections_tf)) # T1
    # print(find_section((6.68, 3.08), sections_tf)) # S1
    # print(find_section((6.71, 4.40), sections_tf)) # T2
    # print(find_section((1.68, 5.11), sections_tf)) # L2
    # print(find_section((-0.83, 4.02), sections_tf)) # T3
    # print(find_section((-1.34, 3.26), sections_tf)) # S2
    # print(find_section((-0.84, 0.36), sections_tf)) # T4









