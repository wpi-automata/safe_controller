import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np

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

# -----------------------------
# Coordinate transform
# old: x down, y right
# new: x right, y up
# -----------------------------
def transform(pt):
    x_old, y_old = pt
    x_new = y_old
    y_new = -x_old
    return (x_new, y_new)

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

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

def get_turn_arc(turn_dict, angles):
    center = line_intersection(turn_dict["P_LL"], turn_dict["P_LR"],
                               turn_dict["P_UL"], turn_dict["P_UR"])
    
    inner_radius = dist(center, turn_dict["P_LL"])
    outer_radius = dist(center, turn_dict["P_LR"])

    theta1, theta2 = angles
    
    inner_arc = Arc(center,      # center (x, y)
                    width = 2*inner_radius,     # diameter in x
                    height = 2*inner_radius,    # diameter in y
                    angle = 0,     # rotation of ellipse
                    theta1 = theta1,    # start angle (deg)
                    theta2 = theta2,   # end angle (deg)
                    linewidth = 2)

    outer_arc = Arc(center,      # center (x, y)
                    width = 2*outer_radius,     # diameter in x
                    height = 2*outer_radius,    # diameter in y
                    angle = 0,     # rotation of ellipse
                    theta1 = theta1,    # start angle (deg)
                    theta2 = theta2,   # end angle (deg)
                    linewidth = 2)

    return inner_arc, outer_arc


# apply transform
sections_tf = {}
for name, pts in sections.items():
    sections_tf[name] = {k: transform(v) for k, v in pts.items()}

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(6,6))

for sec_name, pts in sections_tf.items():
    for label, (x, y) in pts.items():
        ax.scatter(x, y)
        ax.text(x + 0.05, y + 0.05, f"{sec_name}_{label}", fontsize=8)

    order = ["P_UL", "P_UR", "P_LR", "P_LL", "P_UL"]
    poly_x = [pts[k][0] for k in order]
    poly_y = [pts[k][1] for k in order]
    ax.plot(poly_x, poly_y)

# Plot ground truth trajectory
data = np.load("/home/ubuntu/ros_ws/src/safe_controller/safe_controller/last_run.npz", allow_pickle=True)
ground_truth_list = data["ground_truth"]

ground_truth = np.array(ground_truth_list) # in m
gt_x = ground_truth[:, 0] 
gt_y = ground_truth[:, 1]

ax.plot(gt_x, gt_y)

for turn, angles in zip(["T1", "T2", "T3", "T4"], [(-90, 0), (0, 90), (90, 180), (180, 270)]):
    T_inner_arc, T_outer_arc = get_turn_arc(sections_tf[turn], angles)
    ax.add_patch(T_inner_arc)
    ax.add_patch(T_outer_arc)

plt.axis("equal")
plt.xlabel("X (right, m)")
plt.ylabel("Y (up, m)")
plt.title("Transformed Track Points")
plt.grid(True)
plt.show()
