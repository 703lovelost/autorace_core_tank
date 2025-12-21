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
        
        def lane_detector():
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
                
                demonstration = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)

                depth_image = self._latest_depth
                depth_demonstration = cv2.cvtColor(depth_image.copy(), cv2.COLOR_GRAY2BGR)
                
                region = depth_image[400:480, left_index:right_index]
                if len(region[region > 0]):
                    minvobl = region[region > 0].min()
                else:
                    minvobl = 1

                pid_center = (left_index + right_index)//2
                cv2.circle(demonstration, (left_index, combined_mask.shape[0]-10), radius = 20, color=[0, 0, 255], thickness=10)
                cv2.circle(demonstration, (right_index, combined_mask.shape[0]-10), radius = 20, color=[0, 0, 255],thickness=10)
                cv2.circle(demonstration, (pid_center, combined_mask.shape[0]-10), radius = 10, color=[0, 255, 0],thickness=10)
                cv2.circle(demonstration, (center_x, combined_mask.shape[0]-10), radius = 10, color=[255, 0, 0],thickness=10)

                mask = np.any(demonstration != [0, 0, 0], axis=-1)
                depth_demonstration[mask] = demonstration[mask]

                # === ОБНАРУЖЕНИЕ И КЛАССИФИКАЦИЯ ЗНАКА ПОВОРОТА ===
                sign_label = None
                sign_bbox = None  # (x, y, w, h)

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

                if self.turn_sign is None:
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

                                try:
                                    if template_left is None or template_right is None:
                                        self.get_logger().error("Шаблоны left_sign.png или right_sign.png не загружены!")
                                    else:
                                        # Приведём ROI и шаблоны к одинаковому размеру
                                        roi_resized = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
                                        tmpl_l = cv2.resize(template_left, (64, 64), interpolation=cv2.INTER_AREA)
                                        tmpl_r = cv2.resize(template_right, (64, 64), interpolation=cv2.INTER_AREA)

                                        # В grayscale
                                        roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                                        tmpl_l_gray = cv2.cvtColor(tmpl_l, cv2.COLOR_BGR2GRAY)
                                        tmpl_r_gray = cv2.cvtColor(tmpl_r, cv2.COLOR_BGR2GRAY)

                                        # Сравнение
                                        res_l = cv2.matchTemplate(roi_gray, tmpl_l_gray, cv2.TM_CCOEFF_NORMED)
                                        res_r = cv2.matchTemplate(roi_gray, tmpl_r_gray, cv2.TM_CCOEFF_NORMED)

                                        score_l = float(res_l[0, 0])
                                        score_r = float(res_r[0, 0])

                                        self.get_logger().info(f"score left {score_l:.3f}, score right {score_r:.3f}, circularity: {circularity:.3f}")
                                        
                                        if score_l > score_r:
                                            self.get_logger().info("Обнаружен знак: НАЛЕВО")
                                            self.turn_sign = 'left'
                                            sign_color = (255, 0, 0)  # синий
                                            sign_bbox = (x, y, w, h)
                                        else:
                                            self.get_logger().info("Обнаружен знак: НАПРАВО")
                                            self.turn_sign = 'right'
                                            sign_color = (0, 255, 0)  # зелёный
                                            sign_bbox = (x, y, w, h)

                                except Exception as e:
                                    self.get_logger().error(f"Ошибка при сравнении: {e}")
                            else:
                                self.get_logger().debug(f"Контур отклонён: circularity = {circularity:.2f} (должно быть 0.6–1.2)")
                        else:
                            self.get_logger().debug("Контур слишком мал (area < 800)")
                # =============================================


                cv2.line(depth_demonstration, (left_index, 400), (right_index, 400), [0, 255, 0], 2)
                cv2.line(depth_demonstration, (left_index, 480), (right_index, 480), [0, 255, 0], 2)
                cv2.line(depth_demonstration, (left_index, 480), (left_index, 400), [0, 255, 0], 2)
                cv2.line(depth_demonstration, (right_index, 480), (right_index, 400), [0, 255 , 0], 2)
                cv2.putText(
                    depth_demonstration,
                    text=f"{minvobl:.2f}",
                    org=(400, 390),         
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 255, 0),
                    thickness=2
                )
                
                cv2.imshow('PID', depth_demonstration)

                
                return left_index, right_index, minvobl

        if self.state == 0:
            #терпим на светофоре
            threshold = 100

            drive_speed = 0
            rotation_speed = 0
            green_mask = cv2.inRange(hsv_image, np.array([45, 50, 50]), np.array([75, 255, 255]))
            if (green_mask.sum() / 255) > threshold:
                self.state = 1
                self.get_logger().info("стартуем!")

            cv2.imshow('PID', green_mask)
        elif self.state == 1:
            #не пересекаем разметку

            #Настройки 
            drive_speed = 0.1
            rotation_speed = 4

            left_index, right_index, minvobl = lane_detector()

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
            

            if minvobl < 0.10:
                self.state = 2
                self.get_logger().info("смотрим право")
 

        elif self.state == 2:
            left_index, right_index, minvobl = lane_detector()

            drive_speed = 0
            rotation_speed = -0.5
            if minvobl > 0.15:
                if right_index != img_width:
                    self.state = 3
                    self.get_logger().info("смотрим на препятствие")
                else:
                    self.state = 1
                    self.get_logger().info("поехали!")

        elif self.state == 3:
            left_index, right_index, minvobl = lane_detector()

            drive_speed = 0
            rotation_speed = 0.5
            if minvobl < 0.15:
                self.state = 4
                self.get_logger().info("смотрим лево")
        
        elif self.state == 4:
            left_index, right_index, minvobl = lane_detector()

            drive_speed = 0
            rotation_speed = 0.5
            if minvobl > 0.15:
                self.state = 1
                self.get_logger().info("поехали!")

            
    

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