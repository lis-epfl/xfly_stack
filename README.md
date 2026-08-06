# XFly Stack

ROS 2 software stack for Model Predictive Contouring Control (MPCC) of
the XFly bird-scale flapping-wing micro aerial vehicle. The stack
provides the communication link to the aircraft, the real-time
controller, an offline trajectory generator, and the two racing
trajectories used in the accompanying publication.

[![XFly tracking the two racing trajectories](docs/tracking_demo.gif)](https://youtu.be/pVb3DoUntWI)

## Repository structure

```
xfly_stack/
├── xfly_bridge/     Bluetooth Low Energy link to the aircraft, manual teleoperation
├── xfly_control/    MPCC controller, trajectory generator, racing trajectories
└── docs/            Media
```

## Requirements

| Component | Requirement |
|---|---|
| ROS 2 | Humble (`rclpy`, `std_msgs`, `geometry_msgs`, `nav_msgs`) |
| Motion capture | OptiTrack, publishing the aircraft rigid body |
| Message definitions | [`optitrack_multiplexer_ros2_msgs`](https://github.com/lis-epfl/optitrack_packages_ros2) (provides `RigidBodyStamped`) |
| Python | `numpy`, `casadi`; `scipy` for the generator, `matplotlib` for plots |
| Bluetooth | `python3-bleak` |

The controller requires a nonlinear programming solver. IPOPT is
distributed with CasADi and is the default. KNITRO is faster and was
used for the timing results reported in the publication (mean 6.7 ms,
95th percentile 8.2 ms, against a 10 ms control period).

## Installation

Place both packages on the `src` path of a colcon workspace and build:

```bash
mkdir -p ~/xfly_ws/src
ln -s ~/xfly_stack/xfly_bridge  ~/xfly_ws/src/
ln -s ~/xfly_stack/xfly_control ~/xfly_ws/src/
cd ~/xfly_ws
colcon build --symlink-install
source install/setup.bash
```

Both packages are pure Python, so `--symlink-install` allows script
changes to take effect without rebuilding.

## Usage

The components must be started in the following order. Each step
assumes the preceding ones are running.

### 1. Trajectory generation (optional)

Required only for a new circuit. The two trajectories from the
publication are provided as `xfly_control/scripts/track_1.csv` and
`track_2.csv`.

```bash
cd ~/xfly_stack/xfly_control/scripts
python3 generate_track.py --track oval --output my_track.csv --plot
```

The generator is standalone: it requires neither ROS 2 nor the
aircraft. It refines the method described in Appendix B of the
publication; `xfly_control/README_generate_track.md` documents the
differences and the complete formulation.

### 2. Motion capture

Start the [OptiTrack multiplexer](https://github.com/lis-epfl/optitrack_packages_ros2) so that the aircraft rigid body is
published. The controller subscribes by default to:

```
/optitrack_multiplexer_node/rigid_body/XFly2
```

Set the `optitrack_topic` parameter if the rigid body is named
differently.

### 3. Communication bridge

```bash
ros2 launch xfly_bridge xfly_bridge.launch.py
```

The bridge establishes the Bluetooth Low Energy link and relays control
commands to the aircraft. The aircraft MAC address must be supplied
through the `ble_address` parameter; `xfly_bridge/README.md` describes
how to obtain it.

Verify the link before proceeding:

```bash
ros2 topic echo /xfly_bridge/connected      # expected: data: true
ros2 topic echo /xfly_bridge/battery_level
```

Manual teleoperation is available to confirm that the airframe responds
to commands:

```bash
ros2 launch xfly_bridge xfly_teleop.launch.py
```

### 4. Controller

Validation in simulation requires neither the aircraft, motion capture,
nor the bridge:

```bash
cd ~/xfly_stack/xfly_control/scripts
python3 mpcc_node.py --sim --trajectory external \
    --trajectory-csv track_1.csv --n-loops 3 --duration 60
```

Deployment on the aircraft:

```bash
ros2 run xfly_control mpcc_node.py --real --solver knitro \
    --trajectory external \
    --trajectory-csv ~/xfly_stack/xfly_control/scripts/track_1.csv \
    --n-loops 3
```

The trajectory file is resolved relative to the working directory
rather than to the installed node; supply an absolute path when using
`ros2 run`.

## Citation

If you use this software in your research, please cite:

```bibtex
@article{toumieh2026mpcc,
  title   = {Accurate Trajectory Tracking with Model Predictive
             Contouring Control for Bird-Scale Flapping-Wing MAVs},
  author  = {Toumieh, Charbel and Zeng, Jack and Mistry, Niel and
             Floreano, Dario},
  journal = {TODO},
  year    = {TODO},
  doi     = {TODO}
}
```

## License

Released under the MIT License. See the `package.xml` of each package.

## Acknowledgements

Developed at the Laboratory of Intelligent Systems (LIS), École
Polytechnique Fédérale de Lausanne (EPFL).
