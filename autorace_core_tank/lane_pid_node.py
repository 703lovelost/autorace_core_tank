#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from typing import Optional, Tuple

from ament_index_python.packages import get_package_share_directory
import os



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
        img_width = hsv_image.shape[1]
        center_x = hsv_image.shape[1] // 2
        
        def lane_detector(demonstration):
            yellow_mask = cv2.inRange(hsv_image, np.array([20, 100, 100]), np.array([30, 255, 255]))

            white_mask = cv2.inRange(hsv_image, np.array([0, 0, 200]), np.array([180, 30, 255]))

            combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
            row = combined_mask[-10, :]
            
            white_indices = np.where(row==255)[0]

            if len(white_indices[white_indices < center_x]):
                left_index = white_indices[white_indices < center_x][-1]
            else:
                left_index = 0
            
            if len(white_indices[white_indices > center_x]):
                right_index = white_indices[white_indices > center_x][0]
            else:
                right_index = img_width

            
            if not (demonstration is None):
                pid_center = (left_index + right_index)//2
                cv2.circle(demonstration, (max(left_index, center_x - 300), combined_mask.shape[0]-10), radius = 20, color=[0, 0, 255], thickness=10)
                cv2.circle(demonstration, (min(right_index, center_x + 300), combined_mask.shape[0]-10), radius = 20, color=[0, 0, 255],thickness=10)
                cv2.circle(demonstration, (pid_center, combined_mask.shape[0]-10), radius = 10, color=[0, 255, 0],thickness=10)
                cv2.circle(demonstration, (center_x, combined_mask.shape[0]-10), radius = 10, color=[255, 0, 0],thickness=10)
                cv2.putText(
                    demonstration,
                    text=f"{right_index - left_index}",
                    org=(center_x, 450),         
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(255, 0, 0),
                    thickness=2
                )

            
            return left_index, right_index
        

        def look_for_sign(demonstration):
            frame = self._latest_bgr
            hsv = hsv_image

            # Маска для синего (дорожные знаки)
            lower_blue = np.array([100, 100, 50])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # Очистка
            kernel = np.ones((5, 5), np.uint8)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

            # Найти контуры
            contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Самый большой контур
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                
                if area > 2000:
                    # === ПРОВЕРКА НА КРУГЛОСТЬ ===
                    perimeter = cv2.arcLength(largest, True)
                    if perimeter == 0:
                        circularity = 0
                    else:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    # Допустимый диапазон круглости
                    if 0.6 <= circularity <= 1.2:
                        x, y, w, h = cv2.boundingRect(largest)
                        roi = frame[y:y+h, x:x+w]
                        center = (y+h//2, x+w//2)
                        cv2.imshow('sign', roi)

                        #self.get_logger().info(f"area {area}, circularity: {circularity:.3f}")

                        if not (demonstration is None):
                            cv2.line(demonstration, (center[1], 0) ,(center[1], demonstration.shape[0]), [255, 0, 0], 5)

                        return roi, center, area
                    
            return None, None, None
            
                                    
        demonstration = self._latest_bgr.copy()

        if self.state == 0:
            #терпим на светофоре
            threshold = 100

            drive_speed = 0
            rotation_speed = 0
            green_mask = cv2.inRange(hsv_image, np.array([45, 50, 50]), np.array([75, 255, 255]))
            if (green_mask.sum() / 255) > threshold:
                self.state = 1
                self.get_logger().info("стартуем!")

        elif self.state == 1:
            #хорошо работает на резких поворотах, не работает на перекрёстке

            #Настройки 
            drive_speed = 0.1
            rotation_speed = 4

            left_index, right_index = lane_detector(demonstration)

            left_index = max(left_index, center_x - 300)
            right_index = min(right_index, center_x + 300)

            pid_center = (left_index + right_index)//2
            diff = (center_x - pid_center) / abs(right_index - left_index)
            rotation_speed *= diff
            drive_speed *= 1 - (abs(diff) + 0.2) * 2
            if not (self.turn_sign is None):
                if right_index == img_width and left_index == 0:
                    if self.turn_sign == 'right':
                        rotation_speed = -0.3
                    else:
                        rotation_speed = 0.3
                    drive_speed = 0
            
            crop, center, area = look_for_sign(demonstration)

            if not (center is None):
                self.get_logger().info("знак найден, ровняемся на него")
                self.state = 2
        
        elif self.state == 2:
            #едем на знак
            rotation_speed = 1
            drive_speed = 0.05

            crop, center, area = look_for_sign(demonstration)

            if center is None:
                center = self.last_sign_center
            else:
                self.last_sign_center = center

                if area > 20000:
                    self.get_logger().info("определяем направление знака")
                    rotation_speed = 0
                    drive_speed = 0
                    self.state = 3

            diff = (center_x - center[1]) / img_width
            rotation_speed *= diff
        
        elif self.state == 3:
            drive_speed = 0
            rotation_speed = 0
            crop, center, area = look_for_sign(demonstration)
            #картинки знаков уже загружены в template_left template_right (картинки лежат в папке signs)
            #в этой стейте надо только определить, в какую сторону поворачивать и запустить другую стейту


        cv2.imshow('demonstration', demonstration)
    
        return drive_speed, rotation_speed


    def destroy_node(self) -> None:
        cv2.destroyAllWindows()
        super().destroy_node()


def main() -> None:
    # Получаем путь к папке пакета
    pkg_share = get_package_share_directory('autorace_core_tank')
    left_img_path = os.path.join(pkg_share, 'signs', 'left_sign.png')
    right_img_path = os.path.join(pkg_share, 'signs', 'right_sign.png')

    # Загружаем
    global template_left
    global template_right 
    template_left = cv2.imread(left_img_path)
    template_right = cv2.imread(right_img_path)
    if template_left is None or template_right is None:
        print('cringe didnt find templates')
        return
    else:
        print('yipee found signs')
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
