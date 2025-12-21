#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from typing import Optional, Tuple

class SimpleController(Node):
    def __init__(self) -> None:
        super().__init__("simple_controller")

        self._bridge = CvBridge()
        self.saved = False

        # Параметры топиков (можно оставить по умолчанию)
        self.declare_parameter("topics.color_image", "/color/image")
        self.declare_parameter("topics.depth_image", "/depth/image")
        self.declare_parameter("topics.cmd_vel", "/cmd_vel")

        color_topic = self.get_parameter("topics.color_image").value
        depth_topic = self.get_parameter("topics.depth_image").value
        cmd_vel_topic = self.get_parameter("topics.cmd_vel").value

        # Подписки
        self._color_sub = self.create_subscription(Image, color_topic, self._on_color, 10)
        self._depth_sub = self.create_subscription(Image, depth_topic, self._on_depth, 10)
        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        # Хранилище последних изображений
        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._got_color = False
        self._got_depth = False

        self.get_logger().info("SimpleController запущен. Показываю изображение в окне 'Color' и 'Depth'.")

    def _on_color(self, msg: Image) -> None:
        try:
            self._latest_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._got_color = True
            self._try_process()
        except Exception as e:
            self.get_logger().error(f"Ошибка конвертации color: {e}")

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
            self._got_depth = True
            self._try_process()
        except Exception as e:
            self.get_logger().error(f"Ошибка конвертации depth: {e}")

    def _try_process(self) -> None:
        """Вызывается при получении хотя бы одного из изображений. Для синхронной обработки — оба должны быть."""
        if self._latest_bgr is not None and self._latest_depth is not None:
            # Отображаем изображения
            #cv2.imshow("Color", self._latest_bgr)
            # Визуализируем depth как изображение (нормализуем для отображения)
            depth_vis = self._latest_depth.copy()
            depth_vis = np.clip(depth_vis, 0, 5.0) / 5.0 * 255  # до 5 метров
            depth_vis = depth_vis.astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            #cv2.imshow("Depth", depth_vis)
            cv2.waitKey(1)  # важно для обновления окон OpenCV

            # Вызываем пользовательскую логику
            v, w = self.process_image(self._latest_bgr, self._latest_depth)

            # Публикуем команду
            twist = Twist()
            twist.linear.x = float(v)
            twist.angular.z = float(w)
            self._cmd_pub.publish(twist)

    def process_image(self, bgr: np.ndarray, depth: np.ndarray) -> Tuple[float, float]:
        hsv_image = cv2.cvtColor(self._latest_bgr, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(hsv_image, np.array([20, 100, 100]), np.array([30, 255, 255]))

        white_mask = cv2.inRange(hsv_image, np.array([0, 0, 200]), np.array([180, 30, 255]))

        combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
        #combined_mask = yellow_mask
        #cv2.imshow('HSV', combined_mask)
        
        #Настройки 
        rect_w = 400
        rect_h = 50
        left_center = (224, 370)    
        right_center = (624, 370)   

        reference_rotation_speed = 1
        drive_speed = 0.1
       

        
        def get_roi(center_x, center_y, width, height):
            half_w = width // 2
            half_h = height // 2
            x1 = center_x - half_w
            x2 = center_x + half_w
            y1 = center_y - half_h
            y2 = center_y + half_h
            return y1, y2, x1, x2  # (y_top, y_bottom, x_left, x_right)

     
        left_roi = get_roi(left_center[0], left_center[1], rect_w, rect_h)
        right_roi = get_roi(right_center[0], right_center[1], rect_w, rect_h)

  
        area = rect_w * rect_h
        left_sum = combined_mask[left_roi[0]:left_roi[1], left_roi[2]:left_roi[3]].sum() / (area * 255)
        right_sum = combined_mask[right_roi[0]:right_roi[1], right_roi[2]:right_roi[3]].sum() / (area * 255)

 
        demonstration = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)

        cv2.putText(demonstration, f"{left_sum:.2f}",
                    org=(left_center[0] - rect_w // 2, left_center[1] - rect_h // 2 - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
                    color=(0, 0, 255), thickness=2)
        cv2.putText(demonstration, f"{right_sum:.2f}",
                    org=(right_center[0] - rect_w // 2, right_center[1] - rect_h // 2 - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
                    color=(0, 0, 255), thickness=2)

        cv2.rectangle(demonstration,
                    (left_center[0] - rect_w // 2, left_center[1] - rect_h // 2),
                    (left_center[0] + rect_w // 2, left_center[1] + rect_h // 2),
                    (0, 0, 255), thickness=5)
        cv2.rectangle(demonstration,
                    (right_center[0] - rect_w // 2, right_center[1] - rect_h // 2),
                    (right_center[0] + rect_w // 2, right_center[1] + rect_h // 2),
                    (0, 0, 255), thickness=5)

        cv2.imshow('PID', demonstration)
        
        rotation = (-left_sum + right_sum)
        drive_speed *= 1 - abs(rotation)


        return drive_speed, rotation * reference_rotation_speed


    def destroy_node(self) -> None:
        cv2.destroyAllWindows()
        super().destroy_node()


def main() -> None:
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
