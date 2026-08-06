# xfly_stack

Everything needed to fly the XFly ornithopter under MPCC trajectory
tracking, in one place: the BLE link to the aircraft, the controller,
the offline track generator, and the two racing tracks from the
Flapping-MPCC paper.

```
xfly_stack/
├── xfly_bridge/     BLE link to the aircraft + manual teleop
└── xfly_control/    MPCC controller + offline track generation
```

`xfly_stack` is a single self-contained git repository. The two
directories are plain subdirectories of it, not submodules and not
separate repos.

![XFly tracking the two racing tracks](docs/tracking_demo.gif)

The ornithopter flying the two paper tracks under MPCC, in real time.
The white and cyan curves are the reference trajectory; the red squares
are the gates. Track 1 is a pair of overlapping loops, Track 2 a
three-lobe clover — both three-gate closed loops with the altitude
varying between 0.3 m and 1.25 m.

---

## 1. Prerequisites

| Need | Notes |
|---|---|
| ROS 2 (Humble) | `rclpy`, `std_msgs`, `geometry_msgs`, `nav_msgs` |
| `optitrack_multiplexer_ros2_msgs` | provides `RigidBodyStamped`; must be on the workspace path |
| Python | `numpy`, `casadi`, `scipy` (generator), `matplotlib` (plots) |
| `python3-bleak` | BLE client used by the bridge |
| A motion-capture stream | OptiTrack, publishing the rigid body for the aircraft |

An NLP solver is needed by the controller. IPOPT ships with CasADi and
is the default; KNITRO is faster and is what the paper used for the
timing numbers (mean 6.7 ms, P95 8.2 ms).

## 2. Build

These are ROS 2 `ament_cmake` packages. Put them on a workspace's
`src/` path — either by making `xfly_stack` itself the `src` directory,
or by symlinking:

```bash
mkdir -p ~/xfly_ws/src
ln -s ~/xfly_stack/xfly_bridge  ~/xfly_ws/src/
ln -s ~/xfly_stack/xfly_control ~/xfly_ws/src/
cd ~/xfly_ws
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` is worth it here: both packages are pure Python, so
edits to the scripts take effect without rebuilding.

---

## 3. Order of operations

Bring things up in this order. Each step assumes the previous one is
running.

### Step 0 — (optional) generate a track

Only needed for a **new** track. The two tracks from the paper are
already in `xfly_control/scripts/` as `track_1.csv` and `track_2.csv`.

```bash
cd ~/xfly_stack/xfly_control/scripts
python3 generate_track.py --track oval --output my_track.csv --plot
```

Full documentation: `xfly_control/README_generate_track.md`. The
generator is standalone — it needs no ROS and does not talk to the
aircraft.

### Step 1 — motion capture

Start your OptiTrack multiplexer so the aircraft's rigid body is being
published. The controller expects, by default:

```
/optitrack_multiplexer_node/rigid_body/XFly2
```

Override with the `optitrack_topic` parameter if your rigid body is
named differently. **The controller will not arm without tracking** —
it times out after 0.5 s of no pose (`tracking_timeout`).

### Step 2 — the BLE bridge

```bash
ros2 launch xfly_bridge xfly_bridge.launch.py
```

The bridge connects over Bluetooth LE and relays commands to the
aircraft. Before the first run you need the aircraft's MAC address —
see `xfly_bridge/README.md` for how to find it with `bluetoothctl`, then
set it via the `ble_address` parameter.

Confirm the link is up before continuing:

```bash
ros2 topic echo /xfly_bridge/connected      # expect: data: true
ros2 topic echo /xfly_bridge/battery_level
```

To check the airframe responds at all, fly it by hand first:

```bash
ros2 launch xfly_bridge xfly_teleop.launch.py
```

### Step 3 — the controller

Simulation first — no aircraft, no mocap, no bridge needed:

```bash
cd ~/xfly_stack/xfly_control/scripts
python3 mpcc_node.py --sim --trajectory external \
    --trajectory-csv track_1.csv --n-loops 3 --duration 60
```

Then on the real aircraft:

```bash
ros2 run xfly_control mpcc_node.py --real --solver knitro \
    --trajectory external \
    --trajectory-csv ~/xfly_stack/xfly_control/scripts/track_1.csv \
    --n-loops 3
```

The track path is opened **relative to the directory you launch from**,
not to the installed node — so give an absolute path with `ros2 run`, or
`cd` into `scripts/` first and use the bare filename.

Arm when you are ready for it to fly:

```bash
ros2 topic pub --once /mpcc_node/arm std_msgs/msg/Bool "{data: true}"
```
