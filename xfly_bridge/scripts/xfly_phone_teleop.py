#!/usr/bin/env python3
"""
Phone-passthrough teleop for the X-Fly drone.

Runs a BLE peripheral (using bless) that mimics the real X-Fly drone.
The official phone app connects to this fake drone and sends flight
commands. This node decodes them and publishes to the xfly_bridge,
which forwards them to the real drone over its own BLE connection.

    Phone app  ──BLE──>  This node (fake drone)  ──ROS 2──>  xfly_bridge  ──BLE──>  Real drone

Flight writes arrive on characteristic ACC1 as 2 bytes:
    byte[0] = rudder   (0–100, centre = 50)
    byte[1] = flapping (0–255)

This node maps them to:
    /xfly_bridge/cmd    (Vector3Stamped, .x = flapping 0–1, .y = rudder -1–1)

Usage:
    # Terminal 1: run the bridge (connects to real drone)
    ros2 launch xfly_bridge xfly_bridge.launch.py

    # Terminal 2: run this passthrough (phone connects here)
    ros2 run xfly_bridge xfly_phone_teleop.py

    # Terminal 3 (optional): record a bag
    ros2 bag record /mocap/pose /drone/velocity /xfly_bridge/cmd
"""

import asyncio
import threading
import time
from collections import defaultdict

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Bool

from bless import (
    BlessServer,
    GATTCharacteristicProperties,
    GATTAttributePermissions,
)


# ── BLE UUIDs (same as the real X-Fly drone) ─────────────────────────
S_ACC0 = "0000acc0-0000-1000-8000-00805f9b34fb"
C_ACC1 = "0000acc1-0000-1000-8000-00805f9b34fb"   # flight: [rudder, flapping]
C_FE11 = "0000fe11-0000-1000-8000-00805f9b34fb"   # config packets

S_7770 = "00007770-0000-1000-8000-00805f9b34fb"
C_7771 = "00007771-8e22-4541-9d4c-21edae82ed19"
C_7772 = "00007772-8e22-4541-9d4c-21edae82ed19"

S_DDD0 = "0000ddd0-cc7a-482a-984a-7f2ed5b3e58f"
C_DDD1 = "0000ddd1-8e22-4541-9d4c-21edae82ed19"

# ── Frequency tracking ────────────────────────────────────────────────
FREQ_WINDOW = 3.0  # seconds


class PhoneTeleopNode(Node):
    """
    ROS 2 node that receives X-Fly commands from the phone app via BLE
    and publishes them to the xfly_bridge.
    """

    def __init__(self):
        super().__init__("xfly_phone_teleop")

        self.declare_parameter("bridge_ns", "/xfly_bridge")
        self.declare_parameter("ble_name", "XFLY")
        ns = self.get_parameter("bridge_ns").value
        self._ble_name = self.get_parameter("ble_name").value

        # Publishers
        self._pub_cmd = self.create_publisher(
            Vector3Stamped, f"{ns}/cmd", 10)
        self._pub_enable = self.create_publisher(
            Bool, f"{ns}/enable", 10)

        # Bridge connection status
        self._bridge_connected = False
        self.create_subscription(
            Bool, f"{ns}/connected", self._cb_bridge_connected, 10)

        # State
        self._flapping = 0.0       # 0.0 – 1.0
        self._rudder = 0.0         # -1.0 – 1.0
        self._phone_connected = False
        self._last_write_time = 0.0
        self._write_timestamps = defaultdict(list)

        # Publish at fixed rate
        self._timer = self.create_timer(1.0 / 20.0, self._publish)

        # Log timer
        self._log_timer = self.create_timer(3.0, self._log_status)

        # Start BLE server in background
        self._ble_thread = threading.Thread(
            target=self._run_ble, daemon=True)
        self._ble_thread.start()

        self.get_logger().info(
            f"Phone passthrough teleop started. "
            f"BLE name: '{self._ble_name}', publishing to: {ns}")

    # ── ROS callbacks ─────────────────────────────────────────────────

    def _cb_bridge_connected(self, msg: Bool):
        self._bridge_connected = msg.data

    def _publish(self):
        """Publish current flapping/rudder with timestamp."""
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = float(self._flapping)
        msg.vector.y = float(self._rudder)
        msg.vector.z = 0.0
        self._pub_cmd.publish(msg)

    def _log_status(self):
        """Periodic status log."""
        phone = "CONNECTED" if self._phone_connected else "waiting..."
        bridge = "CONNECTED" if self._bridge_connected else "disconnected"
        freq = self._get_freq(C_ACC1)
        self.get_logger().info(
            f"Phone: {phone} | Bridge: {bridge} | "
            f"Flap: {self._flapping:.2f} | Rud: {self._rudder:+.2f} | "
            f"Cmd rate: {freq:.1f} Hz")

    # ── Frequency tracking ────────────────────────────────────────────

    def _record_write(self, uuid):
        now = time.monotonic()
        ts = self._write_timestamps[uuid]
        ts.append(now)
        while ts and (now - ts[0]) > FREQ_WINDOW:
            ts.pop(0)

    def _get_freq(self, uuid):
        ts = self._write_timestamps.get(uuid, [])
        if len(ts) < 2:
            return 0.0
        return (len(ts) - 1) / (ts[-1] - ts[0])

    # ── BLE command decoding ──────────────────────────────────────────

    def _decode_flight_cmd(self, data: bytearray):
        """
        Decode a 2-byte flight command from the phone app.
        byte[0] = rudder:   0–100, centre = 50
        byte[1] = flapping: 0–255
        """
        if len(data) < 2:
            return

        rud_byte = data[0]
        flap_byte = data[1]

        self._flapping = max(0.0, min(1.0, flap_byte / 255.0))
        self._rudder = max(-1.0, min(1.0, (rud_byte - 50.0) / 50.0))

        self._last_write_time = time.monotonic()

    # ── BLE server (runs in background thread) ────────────────────────

    def _run_ble(self):
        asyncio.run(self._ble_main())

    async def _ble_main(self):
        server = BlessServer(name=self._ble_name, adapter="hci1")
        server.write_request_func = self._on_ble_write
        server.read_request_func = self._on_ble_read

        # Service ACC0: flight + config
        await server.add_new_service(S_ACC0)
        await server.add_new_characteristic(
            S_ACC0, C_ACC1,
            GATTCharacteristicProperties.read
            | GATTCharacteristicProperties.write
            | GATTCharacteristicProperties.write_without_response,
            bytearray([0x00]),
            GATTAttributePermissions.readable
            | GATTAttributePermissions.writeable,
        )
        await server.add_new_characteristic(
            S_ACC0, C_FE11,
            GATTCharacteristicProperties.write_without_response,
            bytearray([0x00]),
            GATTAttributePermissions.writeable,
        )

        # Service 7770
        await server.add_new_service(S_7770)
        await server.add_new_characteristic(
            S_7770, C_7771,
            GATTCharacteristicProperties.read
            | GATTCharacteristicProperties.write
            | GATTCharacteristicProperties.write_without_response,
            bytearray([0x00]),
            GATTAttributePermissions.readable
            | GATTAttributePermissions.writeable,
        )
        await server.add_new_characteristic(
            S_7770, C_7772,
            GATTCharacteristicProperties.read
            | GATTCharacteristicProperties.notify,
            bytearray([0x00]),
            GATTAttributePermissions.readable,
        )

        # Service DDD0
        await server.add_new_service(S_DDD0)
        await server.add_new_characteristic(
            S_DDD0, C_DDD1,
            GATTCharacteristicProperties.read
            | GATTCharacteristicProperties.notify,
            bytearray([0x00]),
            GATTAttributePermissions.readable,
        )

        await server.start()
        self.get_logger().info(
            f"BLE server started as '{self._ble_name}' — "
            f"waiting for phone app to connect...")

        while rclpy.ok():
            await asyncio.sleep(1.0)

    def _on_ble_write(self, characteristic, value):
        uuid = str(characteristic.uuid).lower()
        self._record_write(uuid)

        if uuid == C_ACC1.lower():
            self._decode_flight_cmd(value)
            if not self._phone_connected:
                self._phone_connected = True
                self.get_logger().info("Phone app connected (receiving commands)")
                msg = Bool()
                msg.data = True
                self._pub_enable.publish(msg)

        elif uuid == C_FE11.lower():
            self.get_logger().debug(f"Config write: {value.hex().upper()}")
        else:
            self.get_logger().debug(f"Write to {uuid}: {value.hex().upper()}")

    def _on_ble_read(self, characteristic):
        return characteristic.value


def main(args=None):
    rclpy.init(args=args)
    node = PhoneTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        msg = Vector3Stamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.vector.x = 0.0
        msg.vector.y = 0.0
        node._pub_cmd.publish(msg)

        enable_msg = Bool()
        enable_msg.data = False
        node._pub_enable.publish(enable_msg)

        node.get_logger().info("Phone teleop shutting down.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()