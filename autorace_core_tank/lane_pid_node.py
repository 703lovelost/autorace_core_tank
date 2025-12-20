#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
import message_filters

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


class LanePidController(Node):
    """
    Узел управления роботом по разметке (жёлтая слева, белая справа) с PID-регулятором.

    Входы:
      - /color/image (RGB)  : обнаружение разметки
      - /depth/image (Depth): перевод положения разметки из пикселей в метры для ограничения по габаритам
      - /depth/camera_info  : матрица K для восстановления 3D точки (u,v,Z) -> X,Y,Z

    Выход:
      - /cmd_vel : управление роботом (diff-drive)

    Идея:
      1) В нижней части кадра выделяем маски жёлтого и белого.
      2) На нескольких строках (y) ищем внутренние границы разметки:
           - жёлтая: правый край жёлтого в левой половине
           - белая : левый край белого в правой половине
      3) По depth + camera_info переводим x-координаты этих границ в латеральные метры X_left, X_right.
      4) Ошибка = центр полосы (X_center) относительно робота (0.0).
      5) PID по ошибке даёт угловую скорость, линейную скорость выбираем максимально возможную,
         но снижаем при больших ошибках/плохой уверенности/узкой полосе.
    """

    def __init__(self) -> None:
        super().__init__("autorace_core_tank")

        self._bridge = CvBridge()
        self._intrinsics: Optional[CameraIntrinsics] = None

        # -----------------------------
        # Topics
        # -----------------------------
        self.declare_parameter("topics.color_image", "/color/image")
        self.declare_parameter("topics.depth_image", "/depth/image")
        self.declare_parameter("topics.depth_camera_info", "/depth/camera_info")
        self.declare_parameter("topics.cmd_vel", "/cmd_vel")
        self.declare_parameter("topics.debug_image", "/autorace_core_tank/debug_image")

        # -----------------------------
        # PID параметры
        # -----------------------------
        self.declare_parameter("pid.kp", 6.5)
        self.declare_parameter("pid.ki", 0.35)
        self.declare_parameter("pid.kd", 0.20)
        self.declare_parameter("pid.integral_limit", 1.0)

        # -----------------------------
        # Управление скоростью
        # -----------------------------
        self.declare_parameter("control.linear_velocity_max", 1.2)
        self.declare_parameter("control.linear_velocity_min", 0.20)
        self.declare_parameter("control.angular_velocity_max", 2.2)
        self.declare_parameter("control.speed_drop_by_error", 0.85)
        self.declare_parameter("control.speed_drop_by_turn", 0.55)

        # -----------------------------
        # Безопасность по габаритам
        # -----------------------------
        self.declare_parameter("safety.robot_half_width_m", 0.070)
        self.declare_parameter("safety.margin_m", 0.020)
        self.declare_parameter("safety.min_lane_width_m", 0.22)
        self.declare_parameter("safety.stop_on_detection_loss", True)

        # -----------------------------
        # Обработка изображения
        # -----------------------------
        self.declare_parameter("vision.roi_y_start_ratio", 0.55)
        self.declare_parameter("vision.num_sample_rows", 18)
        self.declare_parameter("vision.row_band_height", 2)

        # HSV пороги (настроены под типичные изображения симуляции)
        self.declare_parameter("vision.yellow_hsv_lower", [18, 80, 80])
        self.declare_parameter("vision.yellow_hsv_upper", [40, 255, 255])
        self.declare_parameter("vision.white_hsv_lower", [0, 0, 210])
        self.declare_parameter("vision.white_hsv_upper", [179, 45, 255])

        self.declare_parameter("vision.morph_kernel", 5)
        self.declare_parameter("vision.debug_publish", True)

        self._load_params()

        self._cmd_pub = self.create_publisher(Twist, self._topics_cmd_vel, 10)
        self._dbg_pub = self.create_publisher(Image, self._topics_debug_image, 2)

        self._camera_info_sub = self.create_subscription(
            CameraInfo, self._topics_depth_camera_info, self._on_camera_info, 10
        )

        # Синхронизация color+depth
        self._color_sub = message_filters.Subscriber(self, Image, self._topics_color_image)
        self._depth_sub = message_filters.Subscriber(self, Image, self._topics_depth_image)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub],
            queue_size=20,
            slop=0.15,
            allow_headerless=False
        )
        self._sync.registerCallback(self._on_frames)

        self.add_on_set_parameters_callback(self._on_params)

        # Состояние PID
        self._prev_error: float = 0.0
        self._integral: float = 0.0
        self._prev_time: Optional[float] = None

        self.get_logger().info("autorace_core_tank: LanePidController started")

    # -----------------------------
    # Параметры
    # -----------------------------
    def _load_params(self) -> None:
        self._topics_color_image = self.get_parameter("topics.color_image").get_parameter_value().string_value
        self._topics_depth_image = self.get_parameter("topics.depth_image").get_parameter_value().string_value
        self._topics_depth_camera_info = self.get_parameter("topics.depth_camera_info").get_parameter_value().string_value
        self._topics_cmd_vel = self.get_parameter("topics.cmd_vel").get_parameter_value().string_value
        self._topics_debug_image = self.get_parameter("topics.debug_image").get_parameter_value().string_value

        self._kp = float(self.get_parameter("pid.kp").value)
        self._ki = float(self.get_parameter("pid.ki").value)
        self._kd = float(self.get_parameter("pid.kd").value)
        self._integral_limit = float(self.get_parameter("pid.integral_limit").value)

        self._v_max = float(self.get_parameter("control.linear_velocity_max").value)
        self._v_min = float(self.get_parameter("control.linear_velocity_min").value)
        self._w_max = float(self.get_parameter("control.angular_velocity_max").value)
        self._speed_drop_by_error = float(self.get_parameter("control.speed_drop_by_error").value)
        self._speed_drop_by_turn = float(self.get_parameter("control.speed_drop_by_turn").value)

        self._robot_half_width = float(self.get_parameter("safety.robot_half_width_m").value)
        self._margin = float(self.get_parameter("safety.margin_m").value)
        self._min_lane_width = float(self.get_parameter("safety.min_lane_width_m").value)
        self._stop_on_detection_loss = bool(self.get_parameter("safety.stop_on_detection_loss").value)

        self._roi_y_start_ratio = float(self.get_parameter("vision.roi_y_start_ratio").value)
        self._num_sample_rows = int(self.get_parameter("vision.num_sample_rows").value)
        self._row_band_height = int(self.get_parameter("vision.row_band_height").value)

        self._yellow_lower = np.array(self.get_parameter("vision.yellow_hsv_lower").value, dtype=np.uint8)
        self._yellow_upper = np.array(self.get_parameter("vision.yellow_hsv_upper").value, dtype=np.uint8)
        self._white_lower = np.array(self.get_parameter("vision.white_hsv_lower").value, dtype=np.uint8)
        self._white_upper = np.array(self.get_parameter("vision.white_hsv_upper").value, dtype=np.uint8)

        self._morph_kernel = int(self.get_parameter("vision.morph_kernel").value)
        self._debug_publish = bool(self.get_parameter("vision.debug_publish").value)

    def _on_params(self, params) -> SetParametersResult:
        """
        Callback для "живого" обновления параметров.
        rqt_reconfigure работает с параметрами ROS2: нужно объявить параметры и реагировать на изменения.
        """
        try:
            for p in params:
                if p.name in ("pid.kp", "pid.ki", "pid.kd") and p.type_ != p.Type.DOUBLE:
                    return SetParametersResult(successful=False, reason=f"{p.name} must be float")
                if p.name in ("control.linear_velocity_max", "control.linear_velocity_min", "control.angular_velocity_max") and p.type_ != p.Type.DOUBLE:
                    return SetParametersResult(successful=False, reason=f"{p.name} must be float")
            self._load_params()
            return SetParametersResult(successful=True)
        except Exception as e:
            return SetParametersResult(successful=False, reason=str(e))

    # -----------------------------
    # CameraInfo
    # -----------------------------
    def _on_camera_info(self, msg: CameraInfo) -> None:
        """
        Сохраняем интринсики камеры (матрица K).
        Для глубины (Depth) в Gazebo обычно достаточно K из depth/camera_info.
        """
        if len(msg.k) != 9:
            return
        fx = float(msg.k[0])
        fy = float(msg.k[4])
        cx = float(msg.k[2])
        cy = float(msg.k[5])
        if fx <= 1.0 or fy <= 1.0:
            return
        self._intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)

    # -----------------------------
    # Основной callback: color+depth
    # -----------------------------
    def _on_frames(self, color_msg: Image, depth_msg: Image) -> None:
        """
        Получаем синхронные кадры и считаем команду управления.
        """
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._prev_time is None:
            dt = 0.0
        else:
            dt = max(1e-3, now - self._prev_time)
        self._prev_time = now

        if self._intrinsics is None:
            self._publish_stop()
            return

        bgr = self._to_bgr(color_msg)
        depth = self._to_depth_meters(depth_msg)

        if bgr is None or depth is None:
            self._publish_stop()
            return

        lane = self._estimate_lane_geometry(bgr, depth, self._intrinsics)

        if lane is None:
            if self._stop_on_detection_loss:
                self._publish_stop(reset_integrator=True)
            return

        x_center_m, lane_width_m, conf = lane

        # Проверка минимальной ширины полосы
        if lane_width_m < max(self._min_lane_width, 2.0 * (self._robot_half_width + self._margin)):
            self._publish_stop(reset_integrator=True)
            return

        # Максимально допустимое смещение центра робота относительно центра полосы
        max_offset = (lane_width_m * 0.5) - self._robot_half_width - self._margin
        max_offset = max(1e-3, max_offset)

        # Нормализованная ошибка в метрах (центр полосы в системе камеры: X>0 справа)
        # Чтобы ехать к центру, поворачиваем в сторону знака -X_center.
        error = float(np.clip(x_center_m, -max_offset, max_offset))

        w_cmd = self._pid_step(error, dt)

        # Адаптивная скорость: пытаемся ехать максимально быстро, но снижаем на поворотах/ошибке
        err_ratio = min(1.0, abs(error) / max_offset)
        speed_factor_error = (1.0 - self._speed_drop_by_error * err_ratio)
        speed_factor_turn = 1.0 / (1.0 + self._speed_drop_by_turn * abs(w_cmd))
        speed_factor_conf = min(1.0, max(0.25, conf))

        v_cmd = self._v_max * speed_factor_error * speed_factor_turn * speed_factor_conf
        v_cmd = float(np.clip(v_cmd, self._v_min, self._v_max))

        twist = Twist()
        twist.linear.x = v_cmd
        twist.angular.z = float(np.clip(w_cmd, -self._w_max, self._w_max))
        self._cmd_pub.publish(twist)

        if self._debug_publish:
            dbg = self._make_debug_image(bgr, depth, x_center_m, lane_width_m, error, twist)
            self._dbg_pub.publish(self._bridge.cv2_to_imgmsg(dbg, encoding="bgr8"))

    # -----------------------------
    # PID
    # -----------------------------
    def _pid_step(self, error: float, dt: float) -> float:
        """
        PID: P + I + D.
        На выходе - угловая скорость (rad/s).

        Важно:
          - anti-windup через ограничение интеграла
          - dt защищаем от нуля
        """
        if dt <= 0.0:
            dt = 1e-3

        self._integral += error * dt
        self._integral = float(np.clip(self._integral, -self._integral_limit, self._integral_limit))

        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        u = self._kp * error + self._ki * self._integral + self._kd * derivative

        # Знак: X_center>0 (полоса справа) => надо повернуть направо => angular.z < 0
        return -u

    # -----------------------------
    # Vision utilities
    # -----------------------------
    def _to_bgr(self, msg: Image) -> Optional[np.ndarray]:
        """
        Конвертация ROS Image -> OpenCV BGR.
        """
        try:
            return self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return None

    def _to_depth_meters(self, msg: Image) -> Optional[np.ndarray]:
        """
        Конвертация depth в метры (float32).
        Gazebo depth обычно приходит как 32FC1 (метры).
        """
        try:
            if msg.encoding in ("32FC1", "32FC"):
                depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1").astype(np.float32)
            elif msg.encoding in ("16UC1", "16UC"):
                depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1").astype(np.float32) / 1000.0
            else:
                depth = self._bridge.imgmsg_to_cv2(msg).astype(np.float32)
            depth = np.where(np.isfinite(depth), depth, 0.0)
            return depth
        except Exception:
            return None

    def _estimate_lane_geometry(
        self,
        bgr: np.ndarray,
        depth: np.ndarray,
        intr: CameraIntrinsics
    ) -> Optional[Tuple[float, float, float]]:
        """
        Возвращает:
          - x_center_m  : латеральное смещение центра полосы (метры) относительно камеры (0 - центр)
          - lane_width_m: ширина полосы (метры)
          - conf        : эвристическая уверенность [0..1]
        """
        h, w = bgr.shape[:2]
        roi_y0 = int(h * self._roi_y_start_ratio)
        roi_y0 = max(0, min(h - 2, roi_y0))

        roi = bgr[roi_y0:h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask_y = cv2.inRange(hsv, self._yellow_lower, self._yellow_upper)
        mask_w = cv2.inRange(hsv, self._white_lower, self._white_upper)

        k = max(1, self._morph_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, kernel)
        mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE, kernel)
        mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_OPEN, kernel)
        mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_CLOSE, kernel)

        # Выбор строк для анализа: ближе к нижней части ROI (это ближе к роботу)
        roi_h = roi.shape[0]
        rows = np.linspace(int(roi_h * 0.35), int(roi_h * 0.95), self._num_sample_rows).astype(int)
        rows = np.unique(np.clip(rows, 0, roi_h - 1))

        left_samples: List[Tuple[int, int]] = []
        right_samples: List[Tuple[int, int]] = []

        half = w // 2

        for ry in rows:
            band_y0 = max(0, ry - self._row_band_height)
            band_y1 = min(roi_h, ry + self._row_band_height + 1)

            band_yellow = mask_y[band_y0:band_y1, :half]
            band_white = mask_w[band_y0:band_y1, half:]

            ys = np.where(band_yellow > 0)
            ws = np.where(band_white > 0)

            if ys[1].size > 30:
                # внутренний край жёлтой разметки: максимальный x среди жёлтых
                lx = int(np.max(ys[1]))
                left_samples.append((roi_y0 + ry, lx))
            if ws[1].size > 30:
                # внутренний край белой разметки: минимальный x среди белых (в правой половине)
                rx = int(np.min(ws[1]) + half)
                right_samples.append((roi_y0 + ry, rx))

        if len(left_samples) < max(3, self._num_sample_rows // 5) or len(right_samples) < max(3, self._num_sample_rows // 5):
            return None

        # Берём медиану по координатам для устойчивости
        left_y = int(np.median([p[0] for p in left_samples]))
        left_x = int(np.median([p[1] for p in left_samples]))
        right_y = int(np.median([p[0] for p in right_samples]))
        right_x = int(np.median([p[1] for p in right_samples]))

        # Берём глубину в окрестности точки
        z_left = self._depth_median(depth, left_x, left_y)
        z_right = self._depth_median(depth, right_x, right_y)

        if z_left <= 0.05 or z_right <= 0.05:
            return None

        x_left = (left_x - intr.cx) * z_left / intr.fx
        x_right = (right_x - intr.cx) * z_right / intr.fx

        lane_width = float(x_right - x_left)
        x_center = float((x_left + x_right) * 0.5)

        # Уверенность: больше найденных строк => выше, плюс штраф за сильную асимметрию глубины
        conf = min(1.0, (len(left_samples) + len(right_samples)) / (2.0 * self._num_sample_rows))
        conf *= float(np.clip(1.0 - 0.5 * abs(z_left - z_right), 0.3, 1.0))

        return x_center, abs(lane_width), float(conf)

    def _depth_median(self, depth: np.ndarray, x: int, y: int, r: int = 3) -> float:
        """
        Медианная глубина в квадрате (2r+1)x(2r+1) вокруг (x,y).
        """
        h, w = depth.shape[:2]
        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        patch = depth[y0:y1, x0:x1].reshape(-1)
        patch = patch[(patch > 0.05) & np.isfinite(patch)]
        if patch.size == 0:
            return 0.0
        return float(np.median(patch))

    def _make_debug_image(
        self,
        bgr: np.ndarray,
        depth: np.ndarray,
        x_center_m: float,
        lane_width_m: float,
        error_m: float,
        cmd: Twist
    ) -> np.ndarray:
        """
        Рисует отладочную информацию поверх изображения.
        """
        dbg = bgr.copy()
        h, w = dbg.shape[:2]

        txt = [
            f"x_center_m: {x_center_m:+.3f}",
            f"lane_width_m: {lane_width_m:.3f}",
            f"error_m: {error_m:+.3f}",
            f"cmd: v={cmd.linear.x:.2f}  w={cmd.angular.z:+.2f}",
            f"PID: kp={self._kp:.2f} ki={self._ki:.2f} kd={self._kd:.2f}"
        ]
        y = 25
        for t in txt:
            cv2.putText(dbg, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(dbg, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22

        # Линия "центра кадра"
        cv2.line(dbg, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)

        return dbg

    # -----------------------------
    # Safety / stop
    # -----------------------------
    def _publish_stop(self, reset_integrator: bool = False) -> None:
        """
        Публикует нулевую команду управления.
        """
        if reset_integrator:
            self._integral = 0.0
            self._prev_error = 0.0
        t = Twist()
        self._cmd_pub.publish(t)


def main() -> None:
    rclpy.init()
    node = LanePidController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
