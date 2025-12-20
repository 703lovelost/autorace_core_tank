# autorace_core_tank

ROS 2 Humble пакет управления дифф-приводным роботом по разметке AutoRace:
- левый край дороги: **жёлтая** разметка
- правый край дороги: **белая** разметка
- управление по RGB+Depth камере через PID (P + I + D)
- коэффициенты PID и параметры скорости настраиваются в рантайме через `ros2 param` и GUI `rqt_reconfigure`.

## Что делает узел

Узел `lane_pid_controller`:
1. Получает синхронные кадры:
   - `/color/image`
   - `/depth/image`
2. По HSV выделяет жёлтую и белую разметку.
3. На наборе строк в нижней части кадра находит внутренние границы разметки.
4. Используя `/depth/camera_info` восстанавливает метры X_left/X_right.
5. Считает центр полосы `x_center_m` и ширину `lane_width_m`.
6. PID по `x_center_m` формирует `angular.z`, а `linear.x` пытается держать максимально высоким,
   но снижает его на поворотах/при большой ошибке/при низкой уверенности.

## Установка зависимостей

Стандартные ROS-пакеты (Ubuntu / apt):
- `ros-humble-cv-bridge`
- `ros-humble-message-filters`
- `ros-humble-rqt-reconfigure`

Также нужен OpenCV для Python (`python3-opencv`) и numpy (`python3-numpy`), обычно уже установлены.

## Сборка

Внутри вашего workspace:

```bash
cd ~/ros2_ws/src
# поместите сюда папку autorace_core_tank
cd ..
colcon build --symlink-install
source install/setup.bash
```

## Запуск

Команда по требованию задания:

```bash
ros2 launch autorace_core_tank autorace_core.launch
```

(Дополнительно оставлен классический вариант)
```bash
ros2 launch autorace_core_tank autorace_core.launch.py
```

## Настройка PID в рантайме

### Через rqt_reconfigure

```bash
ros2 run rqt_reconfigure rqt_reconfigure
```

В дереве выберите узел `autorace_core_tank` и меняйте параметры:
- `pid.kp`, `pid.ki`, `pid.kd`
- `control.linear_velocity_max`, `control.angular_velocity_max`, ...
- HSV пороги в `vision.*`

### Через CLI

```bash
ros2 param set /autorace_core_tank pid.kp 7.0
```

## Отладка

Публикуется отладочное изображение:
- `/autorace_core_tank/debug_image` (`sensor_msgs/Image`, BGR)

Можно смотреть через `rqt_image_view`.
