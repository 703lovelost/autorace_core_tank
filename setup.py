from setuptools import setup
import os
from glob import glob

package_name = "autorace_core_tank"

setup(
    name=package_name,
    version="0.0.2",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "signs"), glob("signs/*")),
        (os.path.join("share", package_name, "models"), glob("models/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="PID lane following controller for AutoRace (RGB+Depth) + traffic sign classifier.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "lane_pid_controller = autorace_core_tank.lane_pid_node:main",
            "sign_detector = autorace_core_tank.sign_detector_node:main",
        ],
    },
)
