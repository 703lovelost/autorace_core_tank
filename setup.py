from glob import glob
import os
from setuptools import setup

package_name = "autorace_core_tank"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "models"), glob("models/*")),  # вот эта строчка
        (os.path.join("share", package_name, "signs"), glob("signs/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="AutoRace User",
    maintainer_email="user@local",
    description="PID lane following controller for AutoRace (RGB+Depth).",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "lane_pid_controller = autorace_core_tank.lane_pid_node:main",
        ],
    },
)
