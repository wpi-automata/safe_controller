# ROS Package Overview

This package contains the following ROS nodes:

## Nodes

### teleop
Implements a nominal controller using keyboard teleop. It publishes `cmd_vel` data to a nominal control topic (`/nominal_control` by default).

### control
Filters the control commands from `teleop` using a Belief CBF. It relies on odometry to update system state. It also relies on the `safety` script (`safety` is NOT a ROS node, but implements the minimally invasive CBF QP).

**Default Topics:**
- `TOPIC_NOM_CTRL = "/nominal_control"`
- `TOPIC_SAFE_CTRL = "/cmd_vel"`
- `TOPIC_ODOM = "/vicon_pose"`

### control_lane
Filters the control commands from a `teleop` topic using a Belief CBF. It updates system state based on lane position values. Meant to be used with DeepRacer experiments to drive it around a track. It also relies on the `safety` script.

### spoofer
This ROS Node takes position data from pure python sim (cbf_lite) and publishes it to `lane_position` topic to run with `control_lane` node.

**Default Topics:**
- `TOPIC_NOM_CTRL = "/nominal_control"`
- `TOPIC_SAFE_CTRL = "/cmd_vel"`
- `TOPIC_ODOM = "/vicon_pose"`
- `TOPIC_LANE_POSE = "/lane_position"`

### clauded (lane-keeping CBF)
Lane-keeping node: a proportional steering controller filtered by a first-order CBF (`dp/dt = -k_lane * omega`). Since the QP has a single variable with box constraints, it is solved in closed form as a clamp of `omega_des` into `[omega_lower, omega_upper]`. Forward speed is scaled down near the lane edges. If `/lane_position` goes stale, the node commands a stop.

```bash
ros2 run safe_controller clauded --ros-args -p p_min:=0.08 -p p_max:=0.92 -p lane_sensitivity:=1.0
```

Key parameters: `control_rate_hz` (50), `forward_speed` (0.8), `k_p` (3.0), `lane_sensitivity` (k_lane, 1.0), `p_min`/`p_max` (0.08/0.92), `cbf_alpha` (4.0), `max_omega` (2.0), `slow_down_near_edge` (true), `min_speed_factor` (0.3).

**Topics:**
- Subscribes `/lane_position` (`std_msgs/Float32`, 0 = left edge, 1 = right edge)
- Publishes `/cmd_vel` (`geometry_msgs/Twist`)
- Publishes `/lane_keeping/cbf_debug` (`std_msgs/Float64MultiArray`), 10 values per tick, in this order:
  `[p, error, h1, h2, omega_desired, omega_lower, omega_upper, omega_safe, cbf_active, speed_cmd]`

## Scripts (NOT ROS NODES!)

### plot_debug
This script plots logged data from `control_lane` node, including:

State (ground_truth_list)

State estimation (P, K, x_hat, z_obs)

Right/Left CBF Data (h, Lfh, Lf_2_h, Lg_Lf_h, u_opt)

### safety
Implements minimally invasive Belief CBF QP.


### plot_cbf_debug
Plots `/lane_keeping/cbf_debug` (from `clauded`) recorded in a ROS 2 bag (mcap or sqlite3). It draws four panels: lane position with the safe set, barrier margins h1/h2, omega_desired vs omega_safe inside the CBF bounds, and commanded speed. Spans where the CBF clipped omega or the safe set was violated are shaded. It also prints a summary (rate, min margins, % CBF active, infeasible count).

```bash
# 1. Record while clauded is running
ros2 bag record /lane_keeping/cbf_debug /lane_position /cmd_vel -o cbf_run

# 2. Plot (after colcon build + source install/setup.bash)
ros2 run safe_controller plot_cbf_debug cbf_run                   # interactive window
ros2 run safe_controller plot_cbf_debug cbf_run -o cbf_run.png    # save to file
ros2 run safe_controller plot_cbf_debug cbf_run --csv cbf_run.csv # also export CSV
ros2 run safe_controller plot_cbf_debug cbf_run --t0 10 --t1 40   # time window [s]

# or without building:
python3 safe_controller/plot_cbf_debug.py cbf_run -o cbf_run.png
```
Run with `--help` for all options. If you change the debug message layout in `clauded.py`, update `LABELS` in both files.
