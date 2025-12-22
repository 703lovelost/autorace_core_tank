import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from typing import Optional, Tuple

from ament_index_python.packages import get_package_share_directory
import os

from .sign_classifier import SignClassifier, TURN_LEFT_CLASS_ID, TURN_RIGHT_CLASS_ID


class SimpleController(Node):
    def __init__(self) -> None:
        super().__init__("simple_controller")

        self._bridge = CvBridge()
        self.saved = False
        self.state = 0
        self.turn_sign = None
        self.last_lane_left = 0
        self.last_lane_right = 0
        self.last_sign_center = 0
        self.sign_found = False
        self._turn_ticks = 0

        self.declare_parameter("topics.color_image", "/color/image")
        self.declare_parameter("topics.depth_image", "/depth/image")
        self.declare_parameter("topics.cmd_vel", "/cmd_vel")

        self.declare_parameter("sign.model_path", "")
        self.declare_parameter("sign.conf_threshold", 0.65)
        self.declare_parameter("sign.turn_ticks", 18)
        self.declare_parameter("sign.turn_angular_speed", 2.5)
        self.declare_parameter("sign.turn_linear_speed", 0.05)

        color_topic = self.get_parameter("topics.color_image").value
        depth_topic = self.get_parameter("topics.depth_image").value
        cmd_vel_topic = self.get_parameter("topics.cmd_vel").value

        model_path = self.get_parameter("sign.model_path").value
        if not model_path:
            pkg_share = get_package_share_directory('autorace_core_tank')
            model_path = os.path.join(pkg_share, 'models', 'pytorch_model.bin')
        self._sign_classifier = SignClassifier(model_path=model_path, device="cpu", input_size=32)

        self._color_sub = self.create_subscription(Image, color_topic, self._on_color, 10)
        self._depth_sub = self.create_subscription(Image, depth_topic, self._on_depth, 10)
        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None

        self.get_logger().info("SimpleController started")

    def _on_color(self, msg: Image) -> None:
        try:
            self._latest_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._try_process()
        except Exception as e:
            self.get_logger().error(f"color conversion error: {e}")

    def _on_depth(self, msg: Image) -> None:
        try:
            if msg.encoding in ("32FC1", "32FC"):
                depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            elif msg.encoding in ("16UC1", "16UC"):
                depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1").astype(np.float32) / 1000.0
            else:
                depth = self._bridge.imgmsg_to_cv2(msg).astype(np.float32)
            depth = np.where(np.isfinite(depth), depth, 0.0)
            self._latest_depth = depth
            self._try_process()
        except Exception as e:
            self.get_logger().error(f"depth conversion error: {e}")

    def _try_process(self) -> None:
        if self._latest_bgr is not None and self._latest_depth is not None:
            v, w = self.process_image(self._latest_bgr, self._latest_depth)
            twist = Twist()
            twist.linear.x = float(v)
            twist.angular.z = float(w)
            self._cmd_pub.publish(twist)

    def process_image(self, bgr: np.ndarray, depth: np.ndarray) -> Tuple[float, float]:
        hsv_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        img_width = hsv_image.shape[1]
        center_x = img_width // 2

        def lane_detector(demonstration):
            yellow_mask = cv2.inRange(hsv_image, np.array([20, 100, 100]), np.array([30, 255, 255]))
            white_mask = cv2.inRange(hsv_image, np.array([0, 0, 200]), np.array([180, 30, 255]))
            combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
            row = combined_mask[-10, :]
            white_indices = np.where(row == 255)[0]

            if len(white_indices[white_indices < center_x]):
                left_index = int(white_indices[white_indices < center_x][-1])
            else:
                left_index = 0

            if len(white_indices[white_indices > center_x]):
                right_index = int(white_indices[white_indices > center_x][0])
            else:
                right_index = img_width

            if demonstration is not None:
                pid_center = (left_index + right_index) // 2
                cv2.circle(demonstration, (max(left_index, center_x - 300), combined_mask.shape[0] - 10), radius=20, color=[0, 0, 255], thickness=10)
                cv2.circle(demonstration, (min(right_index, center_x + 300), combined_mask.shape[0] - 10), radius=20, color=[0, 0, 255], thickness=10)
                cv2.circle(demonstration, (pid_center, combined_mask.shape[0] - 10), radius=10, color=[0, 255, 0], thickness=10)
                cv2.circle(demonstration, (center_x, combined_mask.shape[0] - 10), radius=10, color=[255, 0, 0], thickness=10)

            return left_index, right_index

        def look_for_sign(demonstration):
            self.sign_found = False
            frame = bgr

            lower_blue = np.array([100, 100, 50])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

            kernel = np.ones((5, 5), np.uint8)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = float(cv2.contourArea(largest))

                if area > 2000:
                    perimeter = float(cv2.arcLength(largest, True))
                    if perimeter > 0.0:
                        circularity = float(4 * np.pi * area / (perimeter ** 2))
                    else:
                        circularity = 0.0

                    if 0.6 <= circularity <= 1.2:
                        x, y, w, h = cv2.boundingRect(largest)
                        roi = frame[y:y + h, x:x + w]
                        center = (y + h // 2, x + w // 2)

                        cls, label, score = self._sign_classifier.predict(roi)
                        conf_thr = float(self.get_parameter("sign.conf_threshold").value)
                        if score >= conf_thr:
                            self.sign_found = True

                        if demonstration is not None:
                            cv2.line(demonstration, (center[1], 0), (center[1], demonstration.shape[0]), [255, 0, 0], 5)
                            cv2.rectangle(demonstration, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(demonstration, f"{label} {score:.2f}", (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        return roi, center, area, cls, label, score

            return None, None, None, None, None, 0.0

        demonstration = bgr.copy()

        if self.state == 0:
            threshold = 100
            drive_speed = 0
            rotation_speed = 0
            green_mask = cv2.inRange(hsv_image, np.array([45, 50, 50]), np.array([75, 255, 255]))
            if (green_mask.sum() / 255) > threshold:
                self.state = 1
                self.get_logger().info("start")

        elif self.state == 1:
            drive_speed = 0.1
            rotation_speed = 4

            left_index, right_index = lane_detector(demonstration)
            left_index = max(left_index, center_x - 300)
            right_index = min(right_index, center_x + 300)
            pid_center = (left_index + right_index) // 2
            diff = (center_x - pid_center) / max(1, abs(right_index - left_index))
            rotation_speed *= diff
            drive_speed *= 1 - (abs(diff) + 0.2) * 2

            if self.turn_sign is not None:
                if right_index == img_width and left_index == 0:
                    if self.turn_sign == 'right':
                        rotation_speed = -0.3
                    else:
                        rotation_speed = 0.3
                    drive_speed = 0

            crop, center, area, cls, label, score = look_for_sign(demonstration)

            if self.sign_found:
                self.get_logger().info(f"sign found: {label} {score:.2f}")
                self.state = 2

        elif self.state == 2:
            rotation_speed = 1
            drive_speed = 0.05

            crop, center, area, cls, label, score = look_for_sign(demonstration)

            if center is None:
                center = self.last_sign_center
            else:
                self.last_sign_center = center

                if area is not None and area > 20000:
                    self.get_logger().info("classify sign")
                    rotation_speed = 0
                    drive_speed = 0
                    self.state = 3

            diff = (center_x - center[1]) / img_width if center is not None else 0.0
            rotation_speed *= diff

        elif self.state == 3:
            drive_speed = 0
            rotation_speed = 0
            crop, center, area, cls, label, score = look_for_sign(demonstration)

            if self.sign_found and cls in (TURN_LEFT_CLASS_ID, TURN_RIGHT_CLASS_ID):
                self.turn_sign = 'left' if cls == TURN_LEFT_CLASS_ID else 'right'
                self._turn_ticks = 0
                self.get_logger().info(f"turn now: {self.turn_sign}")
                self.state = 4

        elif self.state == 4:
            drive_speed = float(self.get_parameter("sign.turn_linear_speed").value)
            ang = float(self.get_parameter("sign.turn_angular_speed").value)
            ticks_total = int(self.get_parameter("sign.turn_ticks").value)

            if self.turn_sign == 'left':
                rotation_speed = abs(ang)
            else:
                rotation_speed = -abs(ang)

            self._turn_ticks += 1
            if self._turn_ticks >= max(1, ticks_total):
                self.state = 2

        else:
            drive_speed = 0
            rotation_speed = 0

        cv2.imshow('demonstration', demonstration)
        cv2.waitKey(1)

        return drive_speed, rotation_speed

    def destroy_node(self) -> None:
        cv2.destroyAllWindows()
        super().destroy_node()


def main() -> None:
    pkg_share = get_package_share_directory('autorace_core_tank')
    left_img_path = os.path.join(pkg_share, 'signs', 'left_sign.png')
    right_img_path = os.path.join(pkg_share, 'signs', 'right_sign.png')

    global template_left
    global template_right
    template_left = cv2.imread(left_img_path)
    template_right = cv2.imread(right_img_path)

    rclpy.init()
    node = SimpleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
