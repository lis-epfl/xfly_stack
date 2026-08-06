# xfly_control

Model Predictive Contouring Control (MPCC) for trajectory tracking on
the XFly bird-scale ornithopter, together with the offline trajectory
generator.

The controller tracks an arc-length-parameterized reference and
optimizes progress along it online, which removes the need for a
predefined speed profile. Refer to the [repository
README](../README.md) for system requirements, installation, and the
order in which the components must be started.

## Contents

```
scripts/
├── mpcc_node.py                 Controller: ROS 2 node and simulator
├── external_trajectory.py       Converts a trajectory CSV into an
│                                arc-length-parameterized C² spline
├── generate_track.py            Offline trajectory generator
├── README_generate_track.md     Generator documentation
├── track_1.csv                  Track 1: two overlapping loops
└── track_2.csv                  Track 2: three-lobe clover
```

`mpcc_node.py` depends only on `external_trajectory.py`.
`generate_track.py` is standalone and requires neither ROS 2 nor the
controller.

## Running the controller

In simulation, without ROS 2 or the aircraft:

```bash
cd scripts
python3 mpcc_node.py --sim --trajectory external \
    --trajectory-csv track_1.csv --n-loops 3 --duration 60
```

On the aircraft:

```bash
ros2 run xfly_control mpcc_node.py --real --solver knitro \
    --trajectory external \
    --trajectory-csv ~/xfly_stack/xfly_control/scripts/track_1.csv \
    --n-loops 3
```

The trajectory file is resolved relative to the working directory
rather than to the installed node; supply an absolute path when using
`ros2 run`.

The controller arms itself on startup, since `auto_arm` defaults to
`true`; the aircraft begins flying as soon as a valid pose is received.
Launch with `-p auto_arm:=false` to require explicit arming, which is
then issued on `/mpcc_node/arm`:

```bash
ros2 topic pub --once /mpcc_node/arm std_msgs/msg/Bool "{data: true}"
```

### Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--sim` / `--real` | — | Simulation or flight |
| `--trajectory` | `straight` | `straight`, `helix`, `helix_reverse`, `external` |
| `--trajectory-csv` | `track_1.csv` | Trajectory file, used with `--trajectory external` |
| `--n-loops` | 6 | Number of laps of the racing circuit |
| `--duration` | 20.0 | Simulation duration in seconds |
| `--solver` | `ipopt` | `ipopt` (distributed with CasADi) or `knitro` |
| `--disturbance` | `none` | Disturbance model applied in simulation |

`--help` lists the remaining arguments, including the switches used for
the ablation study reported in the publication.

### Node parameters

Declared on the node and configurable through `-p` or a launch file:

| Parameter | Default |
|---|---|
| `optitrack_topic` | `/optitrack_multiplexer_node/rigid_body/XFly2` |
| `cmd_topic` | `/xfly_bridge/cmd` (`Vector3Stamped`; x: flapping, y: rudder) |
| `enable_topic` | `/xfly_bridge/enable` |
| `control_rate` | 100.0 Hz |
| `mpc_horizon` / `mpc_dt` | 15 steps / 0.1 s |
| `tracking_timeout` | 0.5 s; the controller disarms if motion capture is lost |
| `auto_arm` | `true`; the controller arms itself at startup |
| `startup_hold_time` | 0.5 s |

## Generating a trajectory

```bash
cd scripts
python3 generate_track.py --track oval --output my_track.csv --plot
python3 generate_track.py --gates my_gates.json --auto-yaw --output my_track.csv
```

The generator fits quintic Bézier segments through a gate sequence,
minimizing the curvature integral subject to the platform constraints:
turn radius ≥ 1.8 m, climb angle ≤ 20°, 80 × 60 cm gates, and a
7.5 × 7.5 × 4 m flight volume. It exits with status 0 only if the
generated trajectory satisfies every constraint, and can therefore be
used as a verification step.

`README_generate_track.md` documents the formulation, the seven preset
circuits, the gate design rules, and the verification results.

## Racing trajectories

`track_1.csv` and `track_2.csv` are the flight-tested trajectories from
the publication. Both are three-gate closed circuits with altitude
varying between 0.3 m and 1.25 m, flown over three laps, and yielded
mean contouring errors of 8.5 cm and 8.8 cm respectively.

The CSV format is:

```
seg,t,x,y,z,curvature,climb_rate,climb_angle_deg
```

`external_trajectory.py` reads only the `x`, `y` and `z` columns; the
remaining fields are provided for inspection.
