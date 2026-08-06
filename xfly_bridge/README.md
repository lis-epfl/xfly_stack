# xfly_bridge

Bluetooth Low Energy communication bridge between ROS 2 and the XFly
ornithopter, together with manual teleoperation nodes.

The bridge subscribes to control commands, maintains the BLE connection
to the aircraft, and publishes link status and battery level. Refer to
the [repository README](../README.md) for system requirements and the
order in which the components must be started.

## Contents

```
scripts/
├── xfly_bridge.py        BLE bridge node
├── xfly_teleop.py        Keyboard teleoperation
└── xfly_phone_teleop.py  Teleoperation from a mobile device
launch/
├── xfly_bridge.launch.py Bridge alone
└── xfly_teleop.launch.py Bridge and keyboard teleoperation
```

## Usage

```bash
ros2 launch xfly_bridge xfly_bridge.launch.py     # bridge only
ros2 launch xfly_bridge xfly_teleop.launch.py     # bridge and teleoperation
```

## Interface

| Topic | Type | Direction |
|---|---|---|
| `~/cmd` | `geometry_msgs/Vector3Stamped` | Subscribed; x: flapping [0, 1], y: rudder [−1, 1] |
| `~/enable` | `std_msgs/Bool` | Subscribed; flight enable |
| `~/connected` | `std_msgs/Bool` | Published; BLE link status |
| `~/battery_level` | `std_msgs/UInt8` | Published; state of charge in percent |

Both channels are carried in a single timestamped message, so the
flapping and rudder commands applied by the bridge are always
consistent with one another. Commands are clamped on receipt.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `ble_address` | — | MAC address of the aircraft |
| `send_rate` | 100.0 | Command transmission rate in Hz |
| `continuous_send` | `true` | Retransmit the last command continuously |
| `straight_assist` | `false` | Enable straight-flight assistance |
| `straight_strength` | 0 | Straight-flight assistance gain |
| `steer_assist` | `false` | Enable steering assistance |
| `steer_strength` | 0 | Steering assistance gain |

## Obtaining the aircraft MAC address

The `ble_address` parameter must be set to the MAC address of the
aircraft. To determine it:

1. Run `bluetoothctl` in a terminal.
2. Enter `scan on` and look for the `XFly` device.
3. If the device does not appear, open the Ubuntu Bluetooth settings
   and select it there. It will then be listed in the terminal running
   `bluetoothctl`.
