import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np
from gt_to_lane import get_track_sections_dict, transform, line_intersection, dist, get_turn_center_radii

def get_turn_arc(turn_dict, angles):
    center, inner_radius, outer_radius = get_turn_center_radii(turn_dict)

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

sections = get_track_sections_dict()

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

pts = []

def onclick(event):
    if event.inaxes:
        pts.append((event.xdata, event.ydata))
        print(f"Clicked: {event.xdata:.3f}, {event.ydata:.3f}")

        if len(pts) >= 2:
            p1, p2 = pts[-2], pts[-1]
            d = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            print(f"Distance = {d:.4f}")

cid = fig.canvas.mpl_connect("button_press_event", onclick)

plt.axis("equal")
plt.xlabel("X (right, m)")
plt.ylabel("Y (up, m)")
plt.title("Transformed Track Points")
plt.grid(True)
plt.show()
