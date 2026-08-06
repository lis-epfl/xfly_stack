cmake_minimum_required(VERSION 3.8)
project(xfly_control)

find_package(ament_cmake REQUIRED)

# Install the Python node scripts
install(PROGRAMS
  scripts/mpcc_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# external_trajectory.py is imported by mpcc_node.py, and the track CSVs
# are read at runtime, so both must land next to the node.
install(FILES
  scripts/external_trajectory.py
  scripts/track_1.csv
  scripts/track_2.csv
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
