# XFly Stack

ROS 2 software stack for Model Predictive Contouring Control (MPCC) of
the XFly bird-scale flapping-wing micro aerial vehicle. The stack
provides the communication link to the aircraft, the real-time
controller, an offline trajectory generator, and the two racing
trajectories used in the accompanying publication.

[![XFly tracking the two racing trajectories](docs/tracking_demo.gif)](https://youtu.be/pVb3DoUntWI)

The ornithopter tracking both racing trajectories in real time. White
and cyan curves denote the reference trajectory; red frames denote the
gates. Track 1 consists of two overlapping loops, Track 2 of a
three-lobe clover. Both are three-gate closed circuits with altitude
varying between 0.3 m and 1.25 m, flown over three laps.
[Click the animation](https://youtu.be/pVb3DoUntWI) for the full video.

MPCC tracks an arc-length-parameterized reference while optimizing
progress online, which removes the need for a predefined speed profile.
On the XFly platform the method achieves a mean deviation from the
reference between 6.5 cm and 9 cm at airspeeds up to 3 m/s, an 8.5×
improvement over the previous state of the art on the same airframe.

## Repository structure

```
xfly_stack/
├── xfly_bridge/     Bluetooth Low Energy link to the aircraft, manual teleoperation
├── xfly_control/    MPCC controller, trajectory generator, racing trajectories
└── docs/            Media
```

Both directories are ROS 2 `ament_cmake` packages contained in this
single repository; they are not submodules.

## Requirements

| Component | Requirement |
|---|---|
| ROS 2 | Humble (`rclpy`, `std_msgs`, `geometry_msgs`, `nav_msgs`) |
| Motion capture | OptiTrack, publishing the aircraft rigid body |
| Message definitions | `optitrack_multiplexer_ros2_msgs` (provides `RigidBodyStamped`) |
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
aircraft. Refer to `xfly_control/README_generate_track.md` for the
complete documentation.

### 2. Motion capture

Start the OptiTrack multiplexer so that the aircraft rigid body is
published. The controller subscribes by default to:

```
/optitrack_multiplexer_node/rigid_body/XFly2
```

Set the `optitrack_topic` parameter if the rigid body is named
differently. The controller does not arm without a valid pose stream
and disarms after 0.5 s without one (`tracking_timeout`).

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

> **The controller arms itself on startup.** The `auto_arm` parameter
> defaults to `true`, so the aircraft begins flying as soon as a valid
> pose is received. Launch with `-p auto_arm:=false` to require
> explicit arming, which is then issued on `/mpcc_node/arm`:
>
> ```bash
> ros2 topic pub --once /mpcc_node/arm std_msgs/msg/Bool "{data: true}"
> ```

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
