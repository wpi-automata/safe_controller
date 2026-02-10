import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

M_TO_IN = 39.37007874015748

Pt = Tuple[float, float]

@dataclass(frozen=True)
class Section:
    name: str
    P_UL: Pt  # Upper Left
    P_UR: Pt  # Upper Right
    P_LL: Pt  # Lower Left
    P_LR: Pt  # Lower Right

    def poly(self) -> List[Pt]:
        """
        Polygon corners for point-in-polygon tests.
        Assumes a “rectangle-like” section:
          UL -> UR -> LR -> LL
        """
        return [self.P_UL, self.P_UR, self.P_LR, self.P_LL]


def build_track_sections(
    *,
    L1: Dict[str, Pt],
    L2: Dict[str, Pt],
    S1: Dict[str, Pt],
    S2: Dict[str, Pt],
    T1: Dict[str, Pt],
    T2: Dict[str, Pt],
    T3: Dict[str, Pt],
    T4: Dict[str, Pt],
) -> List[Section]:
    """
    Returns all sections with named corner points (meters).

    Each arg (e.g., L1) must be a dict with keys:
      "P_UL", "P_UR", "P_LL", "P_LR"
    """
    def make(name: str, d: Dict[str, Pt]) -> Section:
        return Section(
            name=name,
            P_UL=d["P_UL"],
            P_UR=d["P_UR"],
            P_LL=d["P_LL"],
            P_LR=d["P_LR"],
        )

    return [
        make("L1", L1),
        make("T1", T1),
        make("S1", S1),
        make("T2", T2),
        make("L2", L2),
        make("T3", T3),
        make("S2", S2),
        make("T4", T4),
    ]

# -----------------------------
# Geometry helpers
# -----------------------------
def _dist_point_to_segment(p, a, b) -> float:
    """Euclidean distance from point p to line segment a-b."""
    p = np.asarray(p, float); a = np.asarray(a, float); b = np.asarray(b, float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))

def _angle_wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def _is_angle_between(theta, a0, a1, ccw=True) -> bool:
    """
    Check if theta lies on arc from a0 to a1 (inclusive).
    If ccw=True: arc follows CCW direction from a0 to a1.
    """
    theta = _angle_wrap(theta); a0 = _angle_wrap(a0); a1 = _angle_wrap(a1)

    if ccw:
        # shift so a0 is 0
        d1 = (a1 - a0) % (2*np.pi)
        dt = (theta - a0) % (2*np.pi)
        return dt <= d1 + 1e-12
    else:
        d1 = (a0 - a1) % (2*np.pi)
        dt = (a0 - theta) % (2*np.pi)
        return dt <= d1 + 1e-12

def _dist_point_to_arc(p, c, r, ang0, ang1, ccw=True) -> float:
    """
    Distance from point p to circular arc centered at c with radius r,
    spanning ang0->ang1 (radians) in CCW or CW direction.
    """
    p = np.asarray(p, float); c = np.asarray(c, float)
    v = p - c
    if np.allclose(v, 0.0):
        # Point at center: closest point is any point on circle
        return float(abs(r))

    theta = float(np.arctan2(v[1], v[0]))
    # candidate closest point: radial projection onto circle
    if _is_angle_between(theta, ang0, ang1, ccw=ccw):
        return float(abs(np.linalg.norm(v) - r))

    # otherwise closest is one of the endpoints
    e0 = c + r * np.array([np.cos(ang0), np.sin(ang0)])
    e1 = c + r * np.array([np.cos(ang1), np.sin(ang1)])
    return float(min(np.linalg.norm(p - e0), np.linalg.norm(p - e1)))

def _point_in_poly(p, poly) -> bool:
    """Ray-casting point-in-polygon for a simple polygon."""
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        # edge crosses horizontal ray?
        if ((y0 > y) != (y1 > y)):
            xinters = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-18) + x0
            if x < xinters:
                inside = not inside
    return inside

# -----------------------------
# Main API
# -----------------------------
def distance_from_right_lane_boundary(xy_m, sections) -> float:
    """
    Compute distance (in inches) from the vehicle position to the *right* lane boundary.

    You provide `sections` describing your track split into 6 parts:
      - 2 long straights
      - 2 short straights
      - 4 turns

    Each section is a dict with:
      - "poly": list[(x,y)] polygon corners (meters) describing the lane *area* for that section
      - "right": one of:
           {"type":"line", "a":(x,y), "b":(x,y)}                     # right boundary is a segment
           {"type":"arc",  "c":(x,y), "r":R, "ang0":a0, "ang1":a1,
                         "ccw":True/False}                           # right boundary is an arc

    The function:
      1) finds which section polygon contains the point,
      2) measures perpendicular distance to that section’s right boundary,
      3) returns inches.

    If the point is not inside any section polygon, it falls back to the minimum
    distance to any section's right boundary.
    """
    p = np.asarray(xy_m, float)

    # first try: pick the section that contains the point
    for sec in sections:
        if _point_in_poly(p, sec["poly"]):
            rb = sec["right"]
            if rb["type"] == "line":
                d_m = _dist_point_to_segment(p, rb["a"], rb["b"])
            elif rb["type"] == "arc":
                d_m = _dist_point_to_arc(p, rb["c"], rb["r"], rb["ang0"], rb["ang1"], ccw=rb.get("ccw", True))
            else:
                raise ValueError(f"Unknown right boundary type: {rb['type']}")
            return d_m * M_TO_IN

    # fallback: min distance to any right boundary
    best = float("inf")
    for sec in sections:
        rb = sec["right"]
        if rb["type"] == "line":
            best = min(best, _dist_point_to_segment(p, rb["a"], rb["b"]))
        elif rb["type"] == "arc":
            best = min(best, _dist_point_to_arc(p, rb["c"], rb["r"], rb["ang0"], rb["ang1"], ccw=rb.get("ccw", True)))
    return best * M_TO_IN
