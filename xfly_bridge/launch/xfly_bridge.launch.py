from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("ble_address", default_value="00:80:E1:22:BA:6C"),
        DeclareLaunchArgument("send_rate",   default_value="20.0"),

        Node(
            package="xfly_bridge",
            executable="xfly_bridge.py",
            name="xfly_bridge",
            parameters=[{
                "ble_address":       LaunchConfiguration("ble_address"),
                "send_rate":         LaunchConfiguration("send_rate"),
                "straight_assist":   True,
                "straight_strength": 3,
                "steer_assist":      True,
                "steer_strength":    3,
            }],
            output="screen",
        ),
    ])
