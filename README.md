# autorace_core_tank

ROS2-пакет управления дифф-приводным роботом для соревнования AutoRace 2025.
Авторство команды ТАНК.

## Что делает узел

С принципами работы узла вы можете ознакомиться в презентации проекта здесь.

## Установка зависимостей

Стандартные ROS-пакеты (Ubuntu / apt):
- `ros-jazzy-cv-bridge`
- `ros-jazzy-message-filters`
- `ros-jazzy-rqt-reconfigure`

## Сборка

Создайте workspace `ros2_ws` и выполните следующие команды:

```bash
cd ~/ros2_ws/src
# поместите сюда папку autorace_core_tank
cd ..
colcon build --symlink-install
source install/setup.bash
```

## Запуск

Рекомендуем предварительно запустить `robot_bringup`.

```bash
ros2 launch autorace_core_tank autorace_core.launch
```
