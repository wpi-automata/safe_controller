#!/usr/bin/env python3
"""
occluder.py

Relays the camera stream and blanks out one side of the image for a single,
fixed window, to test the lane pipeline under a momentary occlusion.

    Isaac --/rgb--> occluder --/rgb_occluded--> image_to_lane_pos --/lane_position--> clauded
                       |
                       +--/occlusion/state (std_msgs/Int8): 0 = none, 1 = left, 2 = right

The occlusion clock starts at the FIRST image received (i.e. when you press
Play in Isaac), so start_s is "seconds after Play", independent of when the
node was launched.

Usage (inside the container):
    # 1. occluder (defaults: right side, starts 10 s after first frame, lasts 5 s)
    ros2 run safe_controller occluder --ros-args -p side:=right -p start_s:=10.0 -p duration_s:=5.0

    # 2. lane network reading the occluded stream instead of /rgb
    ros2 run lane_pos_pipeline image_sub --ros-args -r /rgb:=/rgb_occluded

    # 3. controller as usual, then record (include /occlusion/state so the plot can shade it)
    ros2 bag record /lane_keeping/cbf_debug /lane_position /cmd_vel /occlusion/state /tf /gray_pub -o occl_right_5s

    # check the mask covers the lane edge:
    ros2 run rqt_image_view rqt_image_view /rgb_occluded

Restart the occluder before each run (the window fires once per launch).
"""

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int8

STATE_NONE, STATE_LEFT, STATE_RIGHT = 0, 1, 2


class Occluder(Node):

    def __init__(self):
        super().__init__('occluder')
        self.declare_parameter('side', 'right')          # 'left' or 'right'
        self.declare_parameter('start_s', 10.0)          # s after the first frame
        self.declare_parameter('duration_s', 5.0)        # length of the occlusion
        self.declare_parameter('width_frac', 0.35)       # fraction of image width covered
        self.declare_parameter('fill', 0)                # pixel value (0 = black, 127 = gray)
        self.declare_parameter('input_topic', '/rgb')
        self.declare_parameter('output_topic', '/rgb_occluded')

        gp = lambda n: self.get_parameter(n).value
        self.side = str(gp('side')).lower()
        if self.side not in ('left', 'right'):
            raise ValueError(f"side must be 'left' or 'right', got '{self.side}'")
        self.start_s = float(gp('start_s'))
        self.duration_s = float(gp('duration_s'))
        self.width_frac = float(np.clip(gp('width_frac'), 0.0, 1.0))
        self.fill = int(gp('fill'))
        self.side_state = STATE_LEFT if self.side == 'left' else STATE_RIGHT

        self.bridge = CvBridge()
        self.t_first = None
        self.was_active = False

        # depth 2 matches image_to_lane_pos's subscription
        self.pub = self.create_publisher(Image, gp('output_topic'), 2)
        self.state_pub = self.create_publisher(Int8, '/occlusion/state', 10)
        self.create_subscription(Image, gp('input_topic'), self.image_cb, 2)

        self.get_logger().info(
            f"Occluder ready: {gp('input_topic')} -> {gp('output_topic')}, side={self.side}, "
            f"window=[{self.start_s:.1f}, {self.start_s + self.duration_s:.1f}] s after first "
            f"frame, width={self.width_frac:.0%}, fill={self.fill}. Waiting for first image...")

    def image_cb(self, msg: Image):
        now = self.get_clock().now()
        if self.t_first is None:
            self.t_first = now
            self.get_logger().info('First image received; occlusion clock started.')
        t = (now - self.t_first).nanoseconds * 1e-9
        active = self.start_s <= t < self.start_s + self.duration_s

        if active != self.was_active:
            self.get_logger().info(
                f"{'OCCLUSION ON ' if active else 'OCCLUSION OFF'} ({self.side}) at t={t:.2f} s")
            self.was_active = active

        if active:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').copy()
            w = img.shape[1]
            cut = int(round(self.width_frac * w))
            if self.side == 'left':
                img[:, :cut] = self.fill
            else:
                img[:, w - cut:] = self.fill
            out = self.bridge.cv2_to_imgmsg(img, encoding=msg.encoding)
            out.header = msg.header
            self.pub.publish(out)
        else:
            self.pub.publish(msg)  # pass through untouched

        self.state_pub.publish(Int8(data=self.side_state if active else STATE_NONE))


def main(args=None):
    # Default rclpy signal handling: it wakes the executor on Ctrl-C/SIGTERM.
    # (Disabling it, as clauded does to send a final stop, makes this node hang
    # on Ctrl-C when no images are arriving, since it has no timer to wake it.)
    rclpy.init(args=args)
    node = Occluder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
