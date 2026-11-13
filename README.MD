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
Lane-keeping node: a P steering controller filtered by a high-order CBF-QP, built on the `cbf_lite` `DubinsDynamics1D` model and `EKF` estimator (the same pair `control_lane`/`safety.py` use).

- **Model:** state `x = [y, v, theta]`, input `u = [v_cmd, omega_cmd]`. `y` is in `lane_width` units (inches), left-positive, with `y = (1 - p) * lane_width`.
- **Estimator:** EKF with `h(x) = y / lane_width`. It is updated on every `/lane_position` message and predicted on every control tick with the last command sent.
- **Safety filter:** the barriers have relative degree 2, so it uses an HOCBF, `Lf^2 h + LgLf h . u + k1 Lf h + k0 h >= 0`, for both lane boundaries. It is solved with `jaxopt.BoxOSQP` over `[v, omega]`, with Lie derivatives from `jax.grad` of the model's `f`/`g`. If the QP isn't solved, it falls back to the safest input within the actuator limits.
- **Gains:** keep `cbf_k1`/`cbf_k0` small. The model's 0.01 input gains give little control authority, so the `safety.py` values (50 / 37.5) make the QP infeasible near the edge.
- **Path:** `cbf_lite` is imported from `$CBF_LITE_PATH` (default `/home/ubuntu/cbf_lite`).

```bash
ros2 run safe_controller clauded --ros-args -p cbf_k1:=0.1 -p cbf_k0:=0.0025 -p forward_speed:=0.25
```

Key parameters: `control_rate_hz` (50), `forward_speed` (0.25), `k_p` (3.0), `p_min`/`p_max` (0.08/0.92), `lane_width` (24.0), `cbf_k1` (0.1), `cbf_k0` (0.0025), `max_omega` (2.0), `min_speed_factor` (0.3, sets `v_min`), `slow_down_near_edge` (true), `qp_weight_v`/`qp_weight_omega` (1/1), `ekf_q_diag` ([1, 0.001, 0.001]), `ekf_r` (0.01), `ekf_p0` (0.1), `init_v` (0.25), `qp_tol` (1e-6), `qp_maxiter` (1000).

**Topics:**
- Subscribes `/lane_position` (`std_msgs/Float32`, 0 = left edge, 1 = right edge)
- Publishes `/cmd_vel` (`geometry_msgs/Twist`)
- Publishes `/lane_keeping/cbf_debug` (`std_msgs/Float64MultiArray`), 16 values per tick, in this order:
  `[p, p_hat, p_hat_std, h1, h2, y_hat, v_hat, theta_hat, v_desired, omega_desired, v_safe, omega_safe, omega_lower, omega_upper, cbf_active, qp_status]`
  (`h1 = p_hat - p_min`, `h2 = p_max - p_hat`; `omega_lower/upper` is the admissible omega given `v_safe`; `qp_status` 1 = solved)

### occluder
Relays the camera stream and blanks one side of the image (`width_frac` of the width, default 35%) for a single window, to test the pipeline under a momentary occlusion. The window is timed from the **first image received**, i.e. when you press Play in Isaac. It publishes `/occlusion/state` (`std_msgs/Int8`: 0 = none, 1 = left, 2 = right), which `plot_cbf_debug` shades as a hatched band.

```bash
ros2 run safe_controller occluder --ros-args -p side:=right -p start_s:=10.0 -p duration_s:=5.0
ros2 run lane_pos_pipeline image_sub --ros-args -r /rgb:=/rgb_occluded   # network reads the occluded stream
ros2 run safe_controller clauded
ros2 bag record /lane_keeping/cbf_debug /lane_position /cmd_vel /occlusion/state /tf /gray_pub -o occl_right_5s   # /gray_pub = occluded frame the network saw
ros2 run rqt_image_view rqt_image_view /rgb_occluded   # optional: check the mask covers the lane edge
```
Parameters: `side` (left/right), `start_s` (10), `duration_s` (5), `width_frac` (0.35), `fill` (0 = black), `input_topic` (`/rgb`), `output_topic` (`/rgb_occluded`). Restart it before each run, since the window fires once per launch.

## Scripts (NOT ROS NODES!)

### plot_debug
This script plots logged data from `control_lane` node, including:

State (ground_truth_list)

State estimation (P, K, x_hat, z_obs)

Right/Left CBF Data (h, Lfh, Lf_2_h, Lg_Lf_h, u_opt)

### safety
Implements minimally invasive Belief CBF QP.


### plot_cbf_debug
Plots `/lane_keeping/cbf_debug` (from `clauded`) recorded in a ROS 2 bag (mcap or sqlite3). It draws panels for: measured p vs the EKF estimate (±2σ) with the safe set, barrier margins h1/h2, omega_desired vs omega_safe inside the admissible band, v_desired/v_safe/v_hat, and theta_hat. Bags recorded with the older 10-value layout still plot. Spans where the CBF clipped omega or the safe set was violated are shaded. It also prints a summary (rate, min margins, % CBF active, infeasible count).

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
