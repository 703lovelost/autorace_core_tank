# autorace_core_tank

ROS2-пакет управления дифф-приводным роботом для соревнования AutoRace 2025.
Авторство команды ТАНК.

## Что делает узел

С принципами работы узла вы можете ознакомиться в презентации проекта <a href="https://www.figma.com/deck/vfCCsnMUq5keUA0DoJjxFc/%D0%A0%D0%B0%D0%B7%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%BA%D0%B0-%D0%BA%D0%BE%D0%BD%D1%82%D1%80%D0%BE%D0%BB%D0%BB%D0%B5%D1%80%D0%B0?node-id=9-442&t=ovnMBEFYeMgAZIvw-1">здесь</a>.

## Установка зависимостей

Стандартные ROS-пакеты (Ubuntu / apt):
- `ros-jazzy-cv-bridge`
- `ros-jazzy-message-filters`
- `ros-jazzy-rqt-reconfigure`

## Сборка

Создайте workspace `ros2_ws` и выполните следующие команды:

```bash
cd ~/ros2_ws/src
git clone https://github.com/703lovelost/autorace_core_tank.git
cd ..
colcon build --symlink-install
```

## Запуск

Рекомендуем предварительно запустить `robot_bringup`.

```bash
source install/setup.bash
ros2 launch autorace_core_tank autorace_core.launch
```
