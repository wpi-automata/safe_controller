#!/usr/bin/env python3
"""
clauded.py  (node name: lane_keeping_cbf_node)

ROS 2 lane-keeping node: a proportional steering controller filtered by a
high-order CBF-QP, using the cbf_lite DubinsDynamics1D model and EKF estimator
(the same model/estimator pair as control_lane.py / safety.py).

Model (cbf_lite DubinsDynamics1D), state x = [y, v, theta], input u = [v_cmd, omega_cmd]:
    y_dot     = v * sin(2 theta)
    v_dot     = 0.01 * v_cmd
    theta_dot = 0.01 * omega_cmd
  y is lateral position in lane_width units (inches by default), LEFT-positive:
    y = (1 - p) * lane_width   (p = /lane_position, 0 = left edge, 1 = right edge)

Estimator: cbf_lite EKF with measurement h(x) = y / lane_width. EKF.update()
takes the raw /lane_position value and flips it internally (z = 1 - p).

Safety filter (relative degree 2 -> HOCBF), for each barrier h:
    Lf^2 h + Lg Lf h . u + k1 * Lf h + k0 * h >= 0
  h_right(x) = y - (1 - p_max) * lane_width
  h_left(x)  = (1 - p_min) * lane_width - y
QP:
    minimize   w_v (v - v_des)^2 + w_w (omega - omega_des)^2
    subject to the two HOCBF rows,  v_min <= v <= v_max,  |omega| <= max_omega
Solved with jaxopt BoxOSQP; Lie derivatives via jax.grad of the model's f, g.

NOTE on gains: DubinsDynamics1D's 0.01 input gains give the model very little
control authority, so k1/k0 must be small (defaults 0.1 / 0.0025) for the QP to
stay feasible. The safety.py gains (k1 = 50, k0 = 37.5) only bind ~0.4 in from
the boundary, where the model can no longer turn in time -> infeasible.

Subscribes:
    /lane_position          (std_msgs/Float32)
Publishes:
    /cmd_vel                (geometry_msgs/Twist)
    /lane_keeping/cbf_debug (std_msgs/Float64MultiArray), layout in LABELS

Set CBF_LITE_PATH if cbf_lite is not at /home/ubuntu/cbf_lite.
"""

import os
import signal
import sys
import time

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)  # float64 for QP accuracy
import jax.numpy as jnp
from jaxopt import BoxOSQP

CBF_LITE_PATH = os.environ.get("CBF_LITE_PATH", "/home/ubuntu/cbf_lite")
if CBF_LITE_PATH not in sys.path:
    sys.path.append(CBF_LITE_PATH)
from dynamics import DubinsDynamics1D  # noqa: E402  (cbf_lite)
from estimators import EKF             # noqa: E402  (cbf_lite)

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32, Float64MultiArray, MultiArrayDimension
from geometry_msgs.msg import Twist


# Order of values published on /lane_keeping/cbf_debug
# (plot_cbf_debug.py keeps a copy of this list -- update both together)
LABELS = [
    "lane_position",    # latest measured p from /lane_position
    "p_hat",            # EKF estimate of p = 1 - y_hat / lane_width
    "p_hat_std",        # 1-sigma of p_hat = sqrt(P_yy) / lane_width
    "h1_left_margin",   # p_hat - p_min   (>= 0 means safe)
    "h2_right_margin",  # p_max - p_hat   (>= 0 means safe)
    "y_hat",            # EKF y [lane_width units, left-positive]
    "v_hat",            # EKF v
    "theta_hat",        # EKF theta [rad], + = heading left
    "v_desired",        # nominal linear speed
    "omega_desired",    # nominal yaw rate (P controller on p_hat)
    "v_safe",           # linear.x actually sent
    "omega_safe",       # angular.z actually sent
    "omega_lower",      # admissible omega interval given v_safe (from the
    "omega_upper",      #   HOCBF rows and actuator limits)
    "cbf_active",       # 1.0 if the QP changed (v_des, omega_des), else 0.0
    "qp_status",        # BoxOSQP status (1 = solved; otherwise fallback used)
]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class DubinsHOCBFQP:
    """
    HOCBF-QP for DubinsDynamics1D, solved with jaxopt BoxOSQP.

    BoxOSQP form:  min 0.5 z^T Q z + c^T z   s.t.  l <= A z <= u,   z = [v, omega]
        Q = diag(2 w_v, 2 w_w),   c = -2 [w_v v_des, w_w omega_des]
        A = [[Lg Lf h_right], [Lg Lf h_left], I_2]      (CBF rows row-normalized)
        l = [-(Lf^2 h + k1 Lf h + k0 h) for each h,  v_min, -max_omega]
        u = [+inf, +inf, v_max, +max_omega]
    """

    # indices into the params vector
    W, P_MIN, P_MAX, K1, K0, V_MIN, V_MAX, MAX_W, W_V, W_W = range(10)

    def __init__(self, dynamics, tol=1e-6, maxiter=1000):
        self.dyn = dynamics
        self.solver = BoxOSQP(tol=tol, maxiter=maxiter)
        self._solve = jax.jit(self._solve_impl)

    def _f(self, x):
        return self.dyn.f(x).ravel()          # (3,)

    def _g(self, x):
        return jnp.asarray(self.dyn.g(x), dtype=jnp.float64)   # (3, 2)

    def _hocbf_row(self, h, x, k1, k0):
        """Returns (a, b, h, Lf h, Lg h) for the row a . u >= b."""
        grad_h = jax.grad(h)
        Lfh_fn = lambda xx: grad_h(xx) @ self._f(xx)
        grad_Lfh = jax.grad(Lfh_fn)
        Lfh = Lfh_fn(x)
        Lf2h = grad_Lfh(x) @ self._f(x)
        LgLfh = grad_Lfh(x) @ self._g(x)
        Lgh = grad_h(x) @ self._g(x)          # should be 0 (relative degree 2)
        b = -(Lf2h + k1 * Lfh + k0 * h(x))
        return LgLfh, b, h(x), Lfh, Lgh

    def _solve_impl(self, x, u_des, prm):
        W, p_min, p_max = prm[self.W], prm[self.P_MIN], prm[self.P_MAX]
        k1, k0 = prm[self.K1], prm[self.K0]
        lo_box = jnp.array([prm[self.V_MIN], -prm[self.MAX_W]])
        hi_box = jnp.array([prm[self.V_MAX], prm[self.MAX_W]])
        w = jnp.array([prm[self.W_V], prm[self.W_W]])

        h_right = lambda xx: xx[0] - (1.0 - p_max) * W
        h_left = lambda xx: (1.0 - p_min) * W - xx[0]

        aR, bR, hR, _, _ = self._hocbf_row(h_right, x, k1, k0)
        aL, bL, hL, _, _ = self._hocbf_row(h_left, x, k1, k0)
        A_cbf = jnp.vstack([aR, aL])                   # (2, 2)
        b_cbf = jnp.array([bR, bL])

        # Row-normalize the CBF rows (same feasible set). LgLfh is ~1e-2 here
        # (0.01 input gains) vs 1 for the box rows; OSQP's ADMM converges
        # poorly with that mismatch.
        nrm = jnp.linalg.norm(A_cbf, axis=1)
        nrm = jnp.where(nrm > 1e-12, nrm, 1.0)
        A_cbf = A_cbf / nrm[:, None]
        b_cbf = b_cbf / nrm

        A = jnp.vstack([A_cbf, jnp.eye(2)])
        l = jnp.concatenate([b_cbf, lo_box])
        u = jnp.concatenate([jnp.full(2, jnp.inf), hi_box])
        Q = jnp.diag(2.0 * w)
        c = -2.0 * w * u_des

        res = self.solver.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u))
        z = res.params.primal[0]
        status = res.state.status

        # Fallback if not solved (infeasible / not converged): take the box
        # corner that maximizes the most-violated CBF row, i.e. the "safest"
        # input the model allows. Coordinates with ~0 coefficient keep u_des.
        margin = A_cbf @ z - b_cbf
        worst = A_cbf[jnp.argmin(margin)]
        u_safest = jnp.where(worst > 0, hi_box, lo_box)
        u_safest = jnp.where(jnp.abs(worst) < 1e-9, jnp.clip(u_des, lo_box, hi_box), u_safest)
        z = jnp.where(status == BoxOSQP.SOLVED, z, u_safest)
        z = jnp.clip(z, lo_box, hi_box)

        # Admissible omega interval given the chosen v (debug only):
        # a0 v + a1 omega >= b  ->  omega >= (b - a0 v)/a1 if a1 > 0, <= if a1 < 0
        a0, a1 = A_cbf[:, 0], A_cbf[:, 1]
        safe_a1 = jnp.where(jnp.abs(a1) > 1e-12, a1, 1.0)
        bound = (b_cbf - a0 * z[0]) / safe_a1
        w_lo = jnp.max(jnp.where(a1 > 1e-12, bound, -jnp.inf))
        w_hi = jnp.min(jnp.where(a1 < -1e-12, bound, jnp.inf))
        w_lo = jnp.maximum(w_lo, lo_box[1])
        w_hi = jnp.minimum(w_hi, hi_box[1])

        return z, status, res.state.iter_num, hR, hL, w_lo, w_hi

    def solve(self, x, u_des, params):
        out = self._solve(jnp.asarray(x, dtype=jnp.float64),
                          jnp.asarray(u_des, dtype=jnp.float64),
                          jnp.asarray(params, dtype=jnp.float64))
        z, status, iters, hR, hL, w_lo, w_hi = jax.device_get(out)
        return (np.asarray(z, dtype=float), int(status), int(iters),
                float(hR), float(hL), float(w_lo), float(w_hi))


class LaneKeepingCBFNode(Node):

    def __init__(self):
        super().__init__('lane_keeping_cbf_node')

        # ------------------------------------------------------------------
        # Parameters (override via a YAML params file or launch arguments)
        # ------------------------------------------------------------------
        dp = self.declare_parameter
        dp('control_rate_hz', 50.0)
        dp('forward_speed', 0.25)          # nominal v (control_lane.py uses 0.25)
        dp('k_p', 3.0)                     # omega_des = k_p * (p_hat - 0.5)
        dp('p_min', 0.08)                  # left safety boundary (normalized)
        dp('p_max', 0.92)                  # right safety boundary (normalized)
        dp('lane_width', 24.0)             # y units of DubinsDynamics1D (inches)
        dp('cbf_k1', 0.1)                  # HOCBF gain on Lf h
        dp('cbf_k0', 0.0025)               # HOCBF gain on h
        dp('max_omega', 2.0)               # rad/s actuator limit
        dp('min_speed_factor', 0.3)        # v_min = min_speed_factor * forward_speed
        dp('slow_down_near_edge', True)    # scale nominal v_des by margin
        dp('qp_weight_v', 1.0)
        dp('qp_weight_omega', 1.0)
        dp('qp_tol', 1e-6)
        dp('qp_maxiter', 1000)
        dp('ekf_q_diag', [1.0, 0.001, 0.001])   # process noise (as in safety.py)
        dp('ekf_r', 0.01)                  # measurement noise on y / lane_width
        dp('ekf_p0', 0.1)                  # initial covariance (diag)
        dp('init_v', 0.25)                 # initial v estimate
        dp('lane_msg_timeout', 0.5)        # s before /lane_position is stale

        p = self.get_parameter
        self.control_period = 1.0 / p('control_rate_hz').value
        self.forward_speed = p('forward_speed').value
        self.k_p = p('k_p').value
        self.p_min = p('p_min').value
        self.p_max = p('p_max').value
        self.W = p('lane_width').value
        self.k1 = p('cbf_k1').value
        self.k0 = p('cbf_k0').value
        self.max_omega = p('max_omega').value
        self.min_speed_factor = p('min_speed_factor').value
        self.slow_down_near_edge = p('slow_down_near_edge').value
        self.qp_tol = p('qp_tol').value
        self.ekf_Q = jnp.diag(jnp.asarray(p('ekf_q_diag').value, dtype=jnp.float64))
        self.ekf_R = p('ekf_r').value * jnp.eye(3)
        self.ekf_P0 = p('ekf_p0').value * jnp.eye(3)
        self.init_v = p('init_v').value
        self.lane_msg_timeout = p('lane_msg_timeout').value

        if not (0.0 <= self.p_min < 0.5 < self.p_max <= 1.0):
            self.get_logger().warn(
                'p_min/p_max look unusual (expected p_min < 0.5 < p_max within [0,1]).')
        if self.k1 <= 0 or self.k0 <= 0 or self.k1 ** 2 < 4 * self.k0:
            self.get_logger().warn(
                'cbf_k1/cbf_k0 should be > 0 with k1^2 >= 4 k0 (real negative '
                'roots) for a valid HOCBF.')

        self.qp_params = np.array([
            self.W, self.p_min, self.p_max, self.k1, self.k0,
            self.min_speed_factor * self.forward_speed, self.forward_speed,
            self.max_omega, p('qp_weight_v').value, p('qp_weight_omega').value,
        ])

        # ------------------------------------------------------------------
        # Model, QP, estimator (JIT-compile up front so the first tick is fast)
        # ------------------------------------------------------------------
        self.dynamics = DubinsDynamics1D()
        self.qp = DubinsHOCBFQP(self.dynamics, tol=self.qp_tol,
                                maxiter=p('qp_maxiter').value)
        t0 = time.perf_counter()
        x_dummy = np.array([0.5 * self.W, self.init_v, 0.0])
        self.qp.solve(x_dummy, np.array([self.forward_speed, 0.0]), self.qp_params)
        warm = self._make_ekf(x_dummy)
        warm.predict(jnp.zeros(2), 0.02)
        warm.update(0.5)
        self.get_logger().info(
            f'JAX HOCBF-QP + EKF compiled in {time.perf_counter() - t0:.2f} s')

        self.ekf = None                 # created on first /lane_position
        self.u_prev = np.zeros(2)       # last applied [v, omega], used for EKF predict
        self.last_predict_time = None

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.lane_position = 0.5
        self.last_lane_msg_time = None
        self._warned_stale = False

        # ------------------------------------------------------------------
        # I/O
        # ------------------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Float32, '/lane_position', self.lane_position_cb, sensor_qos)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(
            Float64MultiArray, '/lane_keeping/cbf_debug', 10)
        self.timer = self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info(
            f'lane_keeping_cbf_node started: rate={p("control_rate_hz").value} Hz, '
            f'safe zone p in [{self.p_min}, {self.p_max}], HOCBF k1={self.k1}, k0={self.k0}')

    # ----------------------------------------------------------------------
    # Estimator
    # ----------------------------------------------------------------------
    def _make_ekf(self, x0):
        W = self.W
        return EKF(self.dynamics,
                   h=lambda x: jnp.array([x[0] / W]),
                   x_init=jnp.asarray(x0, dtype=jnp.float64),
                   P_init=self.ekf_P0,
                   Q=self.ekf_Q,
                   R=self.ekf_R)

    def lane_position_cb(self, msg: Float32):
        p_meas = clamp(float(msg.data), 0.0, 1.0)
        self.lane_position = p_meas
        self.last_lane_msg_time = self.get_clock().now()
        self._warned_stale = False

        if self.ekf is None:
            # y is left-positive: y = (1 - p) * W
            self.ekf = self._make_ekf([(1.0 - p_meas) * self.W, self.init_v, 0.0])
            self.get_logger().info(
                f'EKF initialized from first measurement p={p_meas:.3f}')
        else:
            self.ekf.update(p_meas)  # EKF flips internally: z = 1 - p

    # ----------------------------------------------------------------------
    # Main control loop
    # ----------------------------------------------------------------------
    def control_loop(self):
        now = self.get_clock().now()

        if self.ekf is None or self.last_lane_msg_time is None or \
                (now - self.last_lane_msg_time).nanoseconds * 1e-9 > self.lane_msg_timeout:
            if not self._warned_stale:
                self.get_logger().warn(
                    '/lane_position is stale or has never been received; '
                    'commanding a stop for safety.')
                self._warned_stale = True
            self.publish_stop()
            # Robot is stopped: don't propagate the model, restart timing.
            self.u_prev = np.zeros(2)
            self.last_predict_time = None
            return

        # EKF time update with the input applied since the last tick
        if self.last_predict_time is not None:
            dt = (now - self.last_predict_time).nanoseconds * 1e-9
            if dt > 0.0:
                self.ekf.predict(jnp.asarray(self.u_prev), dt)
        self.last_predict_time = now

        x_hat, P = self.ekf.get_belief()
        x_hat = np.asarray(x_hat, dtype=float).ravel()
        P = np.asarray(P, dtype=float)
        p_hat = 1.0 - x_hat[0] / self.W
        p_hat_std = float(np.sqrt(max(P[0, 0], 0.0))) / self.W
        h1 = p_hat - self.p_min
        h2 = self.p_max - p_hat

        # Nominal controller
        omega_des = self.k_p * (p_hat - 0.5)
        v_des = self.forward_speed
        if self.slow_down_near_edge:
            margin = min(h1, h2)
            worst_case_range = min(0.5 - self.p_min, self.p_max - 0.5)
            if worst_case_range > 1e-6:
                v_des *= clamp(margin / worst_case_range, self.min_speed_factor, 1.0)
        u_des = np.array([v_des, omega_des])

        # Safety filter
        u_safe, status, iters, _, _, w_lo, w_hi = self.qp.solve(x_hat, u_des, self.qp_params)
        if status != BoxOSQP.SOLVED:
            self.get_logger().warn(
                f'HOCBF-QP not solved (status={status}, iters={iters}); using the '
                'safest box input. Check cbf_k1/cbf_k0 and the EKF estimate.',
                throttle_duration_sec=1.0)
        v_safe, omega_safe = float(u_safe[0]), float(u_safe[1])
        # threshold well above solver tolerance so OSQP noise doesn't flag it
        cbf_active = float(np.max(np.abs(u_safe - u_des))) > 100.0 * self.qp_tol

        twist = Twist()
        twist.linear.x = v_safe
        twist.angular.z = omega_safe
        self.cmd_pub.publish(twist)
        self.u_prev = u_safe.copy()

        self.publish_debug([
            self.lane_position, p_hat, p_hat_std, h1, h2,
            x_hat[0], x_hat[1], x_hat[2],
            v_des, omega_des, v_safe, omega_safe, w_lo, w_hi,
            1.0 if cbf_active else 0.0, float(status),
        ])

    def publish_stop(self):
        self.cmd_pub.publish(Twist())  # all zeros

    def publish_debug(self, values):
        msg = Float64MultiArray()
        msg.layout.dim.append(
            MultiArrayDimension(label='cbf_debug', size=len(LABELS), stride=len(LABELS)))
        msg.data = [float(v) for v in values]
        self.debug_pub.publish(msg)


def main(args=None):
    # Handle SIGINT/SIGTERM ourselves: rclpy's default handler invalidates the
    # context before `finally` runs, so the stop command below would never be
    # sent and the robot would keep its last /cmd_vel.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)

    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt  # same clean-stop path as Ctrl-C
    signal.signal(signal.SIGTERM, _on_sigterm)

    node = LaneKeepingCBFNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
