# xfly_control

MPCC (Model Predictive Contouring Control) trajectory tracking for the
XFly ornithopter, plus the offline track generator.

See `../README.md` for how this fits with `xfly_bridge`, the order to
start things in, and the topic wiring.

## Contents

```
scripts/
├── mpcc_node.py                 the controller (ROS 2 node + simulator)
├── external_trajectory.py       loads a track CSV into an arc-length
│                                parameterized C² spline for the MPCC
├── generate_track.py            offline track generator
├── README_generate_track.md     generator documentation
├── track_1.csv                  paper Track 1 (two overlapping loops)
└── track_2.csv                  paper Track 2 (three-lobe clover)
```

`mpcc_node.py` imports only `external_trajectory.py`. `generate_track.py`
is standalone — it needs neither ROS nor the controller.

## Running the controller

Simulation (no ROS, no aircraft):

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
ros2 topic pub --once /mpcc_node/arm std_msgs/msg/Bool "{data: true}"
```

The CSV is opened relative to your current directory, not to the
installed node — use an absolute path with `ros2 run`, or `cd scripts/`
first.

### Useful arguments

| Flag | Default | Meaning |
|---|---|---|
| `--sim` / `--real` | — | simulate, or fly |
| `--trajectory` | `straight` | `straight`, `helix`, `helix_reverse`, `external` |
| `--trajectory-csv` | `track_1.csv` | track file (path relative to cwd), with `--trajectory external` |
| `--n-loops` | 6 | laps of the racing loop |
| `--duration` | 20.0 | simulation length in seconds |
| `--solver` | `ipopt` | `ipopt` (ships with CasADi) or `knitro` (faster) |
| `--disturbance` | `none` | inject a disturbance in simulation |

`--help` lists the rest, including the ablation switches used for the
paper's ablation study.

### Key parameters

Declared on the node, so settable with `-p` or from a launch file:

| Parameter | Default |
|---|---|
| `optitrack_topic` | `/optitrack_multiplexer_node/rigid_body/XFly2` |
| `cmd_topic` | `/xfly_bridge/cmd` (`Vector3Stamped`: x=flapping, y=rudder) |
| `enable_topic` | `/xfly_bridge/enable` |
| `control_rate` | 100.0 Hz |
| `mpc_horizon` / `mpc_dt` | 15 / 0.1 s |
| `tracking_timeout` | 0.5 s — disarms if mocap drops |

## Generating a new track

```bash
cd scripts
python3 generate_track.py --track oval --output my_track.csv --plot
python3 generate_track.py --gates my_gates.json --auto-yaw --output my_track.csv
```

Exit status is 0 only when the generated track meets the turn-radius and
climb limits, so it can gate a build step. Full documentation, the seven
preset layouts, gate-design rules and verified results are in
`README_generate_track.md`.

Note the default constraints match the airframe: turn radius ≥ 1.8 m,
climb ≤ 20°, 80 × 60 cm gates, 7.5 × 7.5 × 4 m flight volume.

## The two tracks

`track_1.csv` and `track_2.csv` are the flight-tested racing
trajectories from the Flapping-MPCC paper, copied unchanged from
`trajectory_v10.csv` and `trajectory_clov_inverted.csv`. Three-gate
closed loops, altitude 0.3–1.25 m, flown over 3 laps for mean contouring
errors of 8.5 cm and 8.8 cm respectively.
