#!/usr/bin/env python3
"""
lane_keeping_cbf_node.py

ROS 2 node that keeps an Ackermann-steered robot inside its lane using a
Control Barrier Function (CBF) safety filter on top of a simple proportional
steering controller.

Subscribes:
    /lane_position   (std_msgs/Float32)  -- normalized lane position in [0, 1]
                                            0.0 = all the way left
                                            1.0 = all the way right

Publishes:
    /cmd_vel                (geometry_msgs/Twist)         -- control command
    /lane_keeping/cbf_debug (std_msgs/Float64MultiArray)  -- live CBF values,
                                                              see LABELS below

Author: (you)
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32, Float64MultiArray, MultiArrayDimension
from geometry_msgs.msg import Twist


# Order of values published on /lane_keeping/cbf_debug
LABELS = [
    "lane_position",   # p
    "error",           # e = p - 0.5
    "h1_left_margin",  # h1(p) = p - p_min      (>=0 means safe)
    "h2_right_margin", # h2(p) = p_max - p      (>=0 means safe)
    "omega_desired",   # unfiltered proportional controller output
    "omega_lower",     # CBF-QP lower bound on omega
    "omega_upper",     # CBF-QP upper bound on omega
    "omega_safe",      # actual omega sent to /cmd_vel
    "cbf_active",      # 1.0 if the CBF actually clipped omega_desired, else 0.0
    "speed_cmd",       # linear.x actually sent
]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class LaneKeepingCBFNode(Node):

    def __init__(self):
        super().__init__('lane_keeping_cbf_node')

        # ------------------------------------------------------------------
        # Parameters (override via a YAML params file or launch arguments)
        # ------------------------------------------------------------------
        self.declare_parameter('control_rate_hz', 50.0)
        self.declare_parameter('forward_speed', 0.8)       # m/s nominal cruise speed
        self.declare_parameter('k_p', 3.0)                 # proportional gain: error -> omega_des
        self.declare_parameter('lane_sensitivity', 1.0)    # k_lane in dp/dt = -k_lane * omega. TUNE THIS.
        self.declare_parameter('p_min', 0.08)               # left safety margin
        self.declare_parameter('p_max', 0.92)               # right safety margin
        self.declare_parameter('cbf_alpha', 4.0)            # class-K gain (higher = more aggressive correction)
        self.declare_parameter('max_omega', 2.0)            # rad/s actuator limit
        self.declare_parameter('lane_msg_timeout', 0.5)     # seconds before we consider /lane_position stale
        self.declare_parameter('slow_down_near_edge', True)
        self.declare_parameter('min_speed_factor', 0.3)     # floor on speed scaling near edges

        p = self.get_parameter
        self.control_period = 1.0 / p('control_rate_hz').value
        self.forward_speed = p('forward_speed').value
        self.k_p = p('k_p').value
        self.k_lane = p('lane_sensitivity').value
        self.p_min = p('p_min').value
        self.p_max = p('p_max').value
        self.cbf_alpha = p('cbf_alpha').value
        self.max_omega = p('max_omega').value
        self.lane_msg_timeout = p('lane_msg_timeout').value
        self.slow_down_near_edge = p('slow_down_near_edge').value
        self.min_speed_factor = p('min_speed_factor').value

        if not (0.0 <= self.p_min < 0.5 < self.p_max <= 1.0):
            self.get_logger().warn(
                'p_min/p_max look unusual (expected p_min < 0.5 < p_max within [0,1]); '
                'double check your parameters.'
            )

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
            Float32, '/lane_position', self.lane_position_cb, sensor_qos
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(
            Float64MultiArray, '/lane_keeping/cbf_debug', 10
        )

        self.timer = self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info(
            f'lane_keeping_cbf_node started: rate={p("control_rate_hz").value} Hz, '
            f'safe zone p in [{self.p_min}, {self.p_max}], alpha={self.cbf_alpha}'
        )

    # ----------------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------------
    def lane_position_cb(self, msg: Float32):
        self.lane_position = clamp(float(msg.data), 0.0, 1.0)
        self.last_lane_msg_time = self.get_clock().now()
        self._warned_stale = False

    # ----------------------------------------------------------------------
    # CBF safety filter
    # ----------------------------------------------------------------------
    def cbf_filter(self, omega_des: float, p: float):
        """
        Scalar CBF-QP:
            minimize   (omega - omega_des)^2
            subject to omega_lower <= omega <= omega_upper

        derived from h1(p) = p - p_min >= 0, h2(p) = p_max - p >= 0 and the
        kinematic model dp/dt = -k_lane * omega:

            h1_dot + alpha*h1 >= 0  ->  omega <= alpha*h1 / k_lane
            h2_dot + alpha*h2 >= 0  ->  omega >= -alpha*h2 / k_lane

        Since this is a single variable constrained to a box, the exact QP
        solution is just clamping omega_des into [omega_lower, omega_upper].
        (Swap this out for a real QP solver, e.g. osqp/qpsolvers, if you add
        more decision variables or constraints later.)

        Args:
            omega_des: Nominal (unfiltered) yaw-rate command in rad/s from the
                proportional controller, omega_des = k_p * (p - 0.5).
                Positive = counter-clockwise (turn left), which moves the
                robot toward p = 0 under dp/dt = -k_lane * omega.
            p: Current normalized lane position in [0, 1] from /lane_position
                (0.0 = left edge, 1.0 = right edge, 0.5 = lane center).

        Returns:
            (omega_safe, h1, h2, omega_lower, omega_upper)
        """
        k = self.k_lane
        alpha = self.cbf_alpha

        h1 = p - self.p_min
        h2 = self.p_max - p

        if abs(k) > 1e-6:
            omega_upper = (alpha * h1) / k
            omega_lower = -(alpha * h2) / k
        else:
            omega_upper = self.max_omega
            omega_lower = -self.max_omega

        # actuator limits
        omega_upper = min(omega_upper, self.max_omega)
        omega_lower = max(omega_lower, -self.max_omega)

        if omega_lower > omega_upper:
            # Infeasible box (can happen if you're already outside the safe
            # set / badly tuned alpha vs k_lane). Fail toward the safe side.
            self.get_logger().warn(
                'CBF constraints infeasible (omega_lower > omega_upper); '
                'falling back to midpoint. Check p_min/p_max/alpha/lane_sensitivity.',
                throttle_duration_sec=1.0,
            )
            omega_safe = 0.5 * (omega_lower + omega_upper)
        else:
            omega_safe = clamp(omega_des, omega_lower, omega_upper)

        return omega_safe, h1, h2, omega_lower, omega_upper

    # ----------------------------------------------------------------------
    # Main control loop (runs at fixed real-time rate, independent of
    # message arrival rate)
    # ----------------------------------------------------------------------
    def control_loop(self):
        now = self.get_clock().now()

        if self.last_lane_msg_time is None or \
                (now - self.last_lane_msg_time).nanoseconds * 1e-9 > self.lane_msg_timeout:
            if not self._warned_stale:
                self.get_logger().warn(
                    '/lane_position is stale or has never been received; '
                    'commanding a stop for safety.'
                )
                self._warned_stale = True
            self.publish_stop()
            return

        p = self.lane_position
        error = p - 0.5
        omega_des = self.k_p * error

        omega_safe, h1, h2, omega_lo, omega_hi = self.cbf_filter(omega_des, p)
        cbf_active = abs(omega_safe - omega_des) > 1e-9

        speed = self.forward_speed
        if self.slow_down_near_edge:
            margin = min(h1, h2)
            worst_case_range = min(0.5 - self.p_min, self.p_max - 0.5)
            if worst_case_range > 1e-6:
                factor = clamp(margin / worst_case_range, self.min_speed_factor, 1.0)
            else:
                factor = 1.0
            speed *= factor

        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = omega_safe
        self.cmd_pub.publish(twist)

        self.publish_debug(p, error, h1, h2, omega_des, omega_lo, omega_hi,
                            omega_safe, cbf_active, speed)

    def publish_stop(self):
        self.cmd_pub.publish(Twist())  # all zeros

    def publish_debug(self, p, error, h1, h2, omega_des, omega_lo, omega_hi,
                       omega_safe, cbf_active, speed):
        msg = Float64MultiArray()
        msg.layout.dim.append(
            MultiArrayDimension(label='cbf_debug', size=len(LABELS), stride=len(LABELS))
        )
        msg.data = [
            float(p), float(error), float(h1), float(h2),
            float(omega_des), float(omega_lo), float(omega_hi),
            float(omega_safe), 1.0 if cbf_active else 0.0, float(speed),
        ]
        self.debug_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneKeepingCBFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()