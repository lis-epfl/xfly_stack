cmake_minimum_required(VERSION 3.8)
project(xfly_bridge)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/xfly_bridge.py
  scripts/xfly_teleop.py
  scripts/xfly_phone_teleop.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
