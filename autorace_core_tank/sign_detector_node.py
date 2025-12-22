import json
import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from .sign_classifier import SignClassifier, TURN_LEFT_CLASS_ID, TURN_RIGHT_CLASS_ID


class SignDetectorNode(Node):
    def __init__(self):
        super().__init__("sign_detector")
        self._bridge = CvBridge()

        self.declare_parameter("topics.color_image", "/color/image")
        self.declare_parameter("topics.sign_detections", "/autorace_core_tank/sign_detections")
        self.declare_parameter("topics.sign_debug_image", "/autorace_core_tank/sign_debug_image")

        self.declare_parameter("sign.conf_threshold", 0.65)
        self.declare_parameter("sign.min_area", 2000)
        self.declare_parameter("sign.circularity_min", 0.6)
        self.declare_parameter("sign.circularity_max", 1.2)
        self.declare_parameter("sign.input_size", 32)
        self.declare_parameter("sign.model_path", "")

        color_topic = self.get_parameter("topics.color_image").value
        detections_topic = self.get_parameter("topics.sign_detections").value
        debug_topic = self.get_parameter("topics.sign_debug_image").value

        model_path = self.get_parameter("sign.model_path").value
        input_size = int(self.get_parameter("sign.input_size").value)

        self._classifier = SignClassifier(model_path=model_path or None, device="cpu", input_size=input_size)

        self._sub = self.create_subscription(Image, color_topic, self._on_image, 10)
        self._pub_det = self.create_publisher(String, detections_topic, 10)
        self._pub_dbg = self.create_publisher(Image, debug_topic, 10)

        self.get_logger().info("SignDetectorNode started")

    def _find_sign_roi(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < float(self.get_parameter("sign.min_area").value):
            return None

        perimeter = float(cv2.arcLength(largest, True))
        if perimeter <= 0.0:
            return None

        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        cmin = float(self.get_parameter("sign.circularity_min").value)
        cmax = float(self.get_parameter("sign.circularity_max").value)
        if not (cmin <= circularity <= cmax):
            return None

        x, y, w, h = cv2.boundingRect(largest)
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(bgr.shape[1], x + w)
        y1 = min(bgr.shape[0], y + h)
        roi = bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        center = (int(x0 + w // 2), int(y0 + h // 2))
        return {
            "bbox": (int(x0), int(y0), int(w), int(h)),
            "center": center,
            "area": area,
            "circularity": circularity,
            "roi": roi,
        }

    def _on_image(self, msg):
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        det = self._find_sign_roi(bgr)
        out = {
            "found": False,
            "class_id": None,
            "label": None,
            "score": 0.0,
            "bbox": None,
            "center": None,
        }

        dbg = bgr.copy()

        if det is not None:
            cls, label, score = self._classifier.predict(det["roi"])
            out.update({
                "found": True,
                "class_id": cls,
                "label": label,
                "score": float(score),
                "bbox": list(det["bbox"]),
                "center": [int(det["center"][0]), int(det["center"][1])],
            })

            x, y, w, h = det["bbox"]
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            txt = f"{label} {score:.2f}"
            cv2.putText(dbg, txt, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if cls in (TURN_LEFT_CLASS_ID, TURN_RIGHT_CLASS_ID):
                cv2.putText(dbg, "TURN", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if out["found"] and out["score"] < float(self.get_parameter("sign.conf_threshold").value):
            out["found"] = False

        self._pub_det.publish(String(data=json.dumps(out, ensure_ascii=False)))

        try:
            dbg_msg = self._bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
            dbg_msg.header = msg.header
            self._pub_dbg.publish(dbg_msg)
        except Exception as e:
            self.get_logger().error(f"debug publish error: {e}")


def main():
    rclpy.init()
    node = SignDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
