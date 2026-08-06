#!/usr/bin/env python3
"""
ROS 2 node that bridges normalized flight commands to an X-Fly drone over BLE.

Subscriptions:
    ~/cmd           (geometry_msgs/Vector3Stamped)  — .vector.x = flapping (0–1), .vector.y = rudder (-1–1)
    ~/enable        (std_msgs/Bool)                 — toggle flight on/off

Publishes:
    ~/connected     (std_msgs/Bool)     — BLE connection status

Parameters:
    ble_address        — BLE MAC address of the drone
    send_rate          — BLE command send rate in Hz (default 12.5)
    continuous_send    — send commands every tick regardless of change (default false)
    straight_assist    — enable straight-line assist on startup (default true)
    straight_strength  — straight assist strength 0-3 index (default 0)
    steer_assist       — enable steering assist on startup (default true)
    steer_strength     — steer assist strength 0-3 index (default 0)
"""

import asyncio
import subprocess
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Bool, UInt8

from bleak import BleakClient, BleakError


# —— BLE characteristics ——————————————————————————————————————————————
FLIGHT_UUID  = "0000acc1-0000-1000-8000-00805f9b34fb"
CONFIG_UUID  = "0000fe11-0000-1000-8000-00805f9b34fb"
NOTIFY_UUID  = "00007772-8e22-4541-9d4c-21edae82ed19"
BATTERY_UUID = "00002a19-0000-1000-8000-00805f9b34fb"

# Fixed init packet — app sends F3 05 51 52 53 55 56
INIT_PACKET  = bytearray.fromhex("F3055152535556")

# Strength values observed from the app (index 0–3)
STEER_STRENGTHS    = [0x00, 0x19, 0x32, 0x64]   # 0, 25, 50, 100
STRAIGHT_STRENGTHS = [0x00, 0x19, 0x32, 0x64]   # 0, 25, 50, 100

HANDSHAKE_DELAY = 0.2
CONFIG_GAP      = 0.2


class XFlyBridgeNode(Node):
    def __init__(self):
        super().__init__("xfly_bridge")

        # —— Parameters ————————————————————————————————————————————————
        self.declare_parameter("ble_address", "00:80:E1:22:BA:6C") # 00:80:E1:22:BA:6C  00:80:E1:22:B8:22
        self.declare_parameter("send_rate", 100.0)
        self.declare_parameter("continuous_send", True)
        self.declare_parameter("straight_assist", False)
        self.declare_parameter("straight_strength", 0)
        self.declare_parameter("steer_assist", False)
        self.declare_parameter("steer_strength", 0)

        self.ble_address       = self.get_parameter("ble_address").value
        self.send_rate         = self.get_parameter("send_rate").value
        self.continuous_send   = self.get_parameter("continuous_send").value
        self.straight_assist   = self.get_parameter("straight_assist").value
        self.straight_strength = self.get_parameter("straight_strength").value
        self.steer_assist      = self.get_parameter("steer_assist").value
        self.steer_strength    = self.get_parameter("steer_strength").value

        # —— Shared state (written by ROS callbacks, read by BLE loop) —
        self._lock = threading.Lock()
        self._flapping = 0.0
        self._rudder   = 0.0
        self._flight_enabled = True
        self._user_active = False
        self._pending_configs: list[bytearray] = []
        self._connected = False
        self._last_rud_byte = -1
        self._last_flp_byte = -1

        # —— ROS subscriptions —————————————————————————————————————————
        self.create_subscription(Vector3Stamped, "~/cmd", self._cb_cmd, 10)
        self.create_subscription(Bool, "~/enable", self._cb_enable, 10)

        # —— Publisher for connection status ————————————————————————————
        self._pub_connected = self.create_publisher(Bool, "~/connected", 10)
        self._pub_battery   = self.create_publisher(UInt8, "~/battery_level", 10)
        self._status_timer = self.create_timer(1.0, self._publish_status)

        # —— Start BLE loop in a background thread —————————————————————
        self._ble_thread = threading.Thread(target=self._run_ble_loop, daemon=True)
        self._ble_thread.start()

        self.get_logger().info(
            f"X-Fly bridge started — BLE address: {self.ble_address}, "
            f"rate: {self.send_rate} Hz, continuous: {self.continuous_send}"
        )

    # —— ROS callbacks —————————————————————————————————————————————————

    def _cb_cmd(self, msg: Vector3Stamped):
        with self._lock:
            self._flapping = max(0.0, min(1.0, msg.vector.x))
            self._rudder = max(-1.0, min(1.0, msg.vector.y))
            self._user_active = True

    def _cb_enable(self, msg: Bool):
        with self._lock:
            self._flight_enabled = msg.data
            self._user_active = True
            self._pending_configs.append(INIT_PACKET)
            if not msg.data:
                self._flapping = 0.0
                self._rudder = 0.0
        self.get_logger().info(f"Flight {'ENABLED' if msg.data else 'DISABLED'}")

    def _publish_status(self):
        msg = Bool()
        msg.data = self._connected
        self._pub_connected.publish(msg)

    def _on_battery_notify(self, _sender, data: bytearray):
        """BLE Battery Level characteristic returns a single byte (0–100 %)."""
        if data:
            level = data[0]
            msg = UInt8()
            msg.data = level
            self._pub_battery.publish(msg)
            self.get_logger().debug(f"Battery level: {level}%")

    # —— Mapping helpers ———————————————————————————————————————————————

    @staticmethod
    def _map_flapping(value: float) -> int:
        """Map [0, 1] → [0, 255]."""
        return int(round(value * 255))

    @staticmethod
    def _map_rudder(value: float) -> int:
        """Map [-1, 1] → [0, 100], centre = 50."""
        return int(round((value + 1.0) * 0.5 * 100))

    # —— Config packet builders ————————————————————————————————————————

    @staticmethod
    def _build_startup_config(straight_on, straight_strength,
                              steer_on, steer_strength):
        """
        Build the F9 batch config the app sends on startup.
        Format: F9 <count> <subcmd1> <val1> <subcmd2> <val2> ...

        Observed from app:
            F9 04 51 00 56 64 52 00 53 00
              = 4 params: straight_enable=0, straight_str=0x64,
                          steer_enable=0, steer_str=0
        """
        sa_val = 0x01 if straight_on else 0x00
        ss_val = STRAIGHT_STRENGTHS[min(straight_strength, 3)] if straight_on else 0x00
        st_on  = 0x01 if steer_on else 0x00
        st_val = STEER_STRENGTHS[min(steer_strength, 3)] if steer_on else 0x00

        return bytearray([
            0xF9, 0x04,
            0x51, sa_val,
            0x56, ss_val,
            0x52, st_on,
            0x53, st_val,
        ])

    @staticmethod
    def _build_steer_config(steer_on, steer_strength):
        """
        Build the F9 batch steer config.
        Observed from app:
            OFF: F9 02 52 00 53 00
            ON:  F9 02 52 01 53 19
        """
        st_on  = 0x01 if steer_on else 0x00
        st_val = STEER_STRENGTHS[min(steer_strength, 3)] if steer_on else 0x00
        return bytearray([0xF9, 0x02, 0x52, st_on, 0x53, st_val])

    @staticmethod
    def _build_straight_enable(on: bool):
        """
        Straight assist uses individual F8 commands.
        Observed from app:
            ON:  F8 51 01
            OFF: F8 51 00
        """
        return bytearray([0xF8, 0x51, 0x01 if on else 0x00])

    # —— BLE background loop ——————————————————————————————————————————

    def _run_ble_loop(self):
        asyncio.run(self._ble_main())

    def _apply_low_latency_ble(self):
        """
        Run hcitool lecup to set aggressive BLE connection parameters.
        Equivalent to:
            sudo hcitool lecup --handle 2048 --min 6 --max 6 --latency 0 --timeout 200
        """
        cmd = [
            "sudo", "hcitool", "lecup",
            "--handle", "2048",
            "--min", "6",
            "--max", "6",
            "--latency", "0",
            "--timeout", "200",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.get_logger().info("hcitool lecup applied (low-latency BLE params).")
            else:
                self.get_logger().warn(
                    f"hcitool lecup failed (rc={result.returncode}): {result.stderr.strip()}"
                )
        except FileNotFoundError:
            self.get_logger().error("hcitool not found — install bluez or run as root.")
        except subprocess.TimeoutExpired:
            self.get_logger().warn("hcitool lecup timed out.")

    async def _ble_main(self):
        period = 1.0 / self.send_rate

        while rclpy.ok():
            try:
                self.get_logger().info(f"Connecting to {self.ble_address} …")
                async with BleakClient(self.ble_address) as client:
                    self.get_logger().info("BLE connected.")
                    with self._lock:
                        self._connected = True
                        self._last_rud_byte = -1
                        self._last_flp_byte = -1

                    # Apply low-latency connection parameters
                    self._apply_low_latency_ble()

                    await client.start_notify(
                        NOTIFY_UUID,
                        lambda _s, d: self.get_logger().debug(f"BLE notify: {d.hex()}")
                    )

                    # Subscribe to battery level notifications (fall back to polling)
                    _battery_notify_ok = False
                    try:
                        await client.start_notify(BATTERY_UUID, self._on_battery_notify)
                        _battery_notify_ok = True
                        self.get_logger().info("Battery notifications enabled.")
                    except (BleakError, Exception) as e:
                        self.get_logger().warn(
                            f"Battery notify not supported ({e}), will poll instead."
                        )
                    _battery_poll_counter = 0

                    # ---- Replicate the app's exact startup sequence ----
                    # Step 1: Send F9 config FIRST (app does this before init)
                    startup_pkt = self._build_startup_config(
                        self.straight_assist, self.straight_strength,
                        self.steer_assist,    self.steer_strength,
                    )
                    await client.write_gatt_char(CONFIG_UUID, startup_pkt, response=False)
                    self.get_logger().info(f"[CFG] STARTUP {startup_pkt.hex().upper()}")
                    await asyncio.sleep(CONFIG_GAP)

                    # Step 2: Send a few idle flight commands (app does this)
                    for _ in range(5):
                        await client.write_gatt_char(
                            FLIGHT_UUID, bytearray([0x32, 0x00]), response=False
                        )
                        await asyncio.sleep(0.08)

                    # Step 3: Send F3 init handshake
                    await client.write_gatt_char(CONFIG_UUID, INIT_PACKET, response=False)
                    self.get_logger().info(f"[CFG] INIT {INIT_PACKET.hex().upper()}")
                    await asyncio.sleep(HANDSHAKE_DELAY)

                    # Step 4: Send config again after init (belt and suspenders)
                    await client.write_gatt_char(CONFIG_UUID, startup_pkt, response=False)
                    self.get_logger().info(f"[CFG] STARTUP2 {startup_pkt.hex().upper()}")
                    await asyncio.sleep(CONFIG_GAP)

                    next_tick = asyncio.get_event_loop().time()

                    while client.is_connected and rclpy.ok():
                        next_tick += period

                        await self._process_pending_configs(client)

                        with self._lock:
                            enabled     = self._flight_enabled
                            flapping    = self._flapping
                            rudder      = self._rudder
                            user_active = self._user_active

                        if enabled and user_active:
                            rud_byte = self._map_rudder(rudder)
                            flp_byte = self._map_flapping(flapping)
                            should_send = (
                                self.continuous_send
                                or rud_byte != self._last_rud_byte
                                or flp_byte != self._last_flp_byte
                            )
                            if should_send:
                                try:
                                    await client.write_gatt_char(
                                        FLIGHT_UUID,
                                        bytearray([rud_byte, flp_byte]),
                                        response=False,
                                    )
                                    self._last_rud_byte = rud_byte
                                    self._last_flp_byte = flp_byte
                                except BleakError as e:
                                    self.get_logger().warn(f"Flight write failed: {e}")
                                    break

                        now = asyncio.get_event_loop().time()
                        await asyncio.sleep(max(0.0, next_tick - now))

                        # Poll battery ~once per second if notifications unavailable
                        if not _battery_notify_ok:
                            _battery_poll_counter += 1
                            if _battery_poll_counter >= int(self.send_rate):
                                _battery_poll_counter = 0
                                try:
                                    batt_data = await client.read_gatt_char(BATTERY_UUID)
                                    self._on_battery_notify(None, batt_data)
                                except BleakError as e:
                                    self.get_logger().debug(f"Battery read failed: {e}")

            except BleakError as e:
                self.get_logger().warn(f"BLE error: {e} — retrying in 3 s")
            except Exception as e:
                self.get_logger().error(f"Unexpected BLE error: {e}")
            finally:
                with self._lock:
                    self._connected = False

            await asyncio.sleep(3.0)

    async def _send_startup_config(self, client: BleakClient):
        """
        Send the startup config as the app does:
        1) F3 handshake (already sent)
        2) F9 batch with all 4 settings
        """
        pkt = self._build_startup_config(
            self.straight_assist, self.straight_strength,
            self.steer_assist,    self.steer_strength,
        )
        await client.write_gatt_char(CONFIG_UUID, pkt, response=False)
        self.get_logger().info(f"[CFG] STARTUP {pkt.hex().upper()}")
        await asyncio.sleep(CONFIG_GAP)

    async def _process_pending_configs(self, client: BleakClient):
        with self._lock:
            pending = list(self._pending_configs)
            self._pending_configs.clear()

        for pkt in pending:
            delay = HANDSHAKE_DELAY if pkt == INIT_PACKET else CONFIG_GAP
            await client.write_gatt_char(CONFIG_UUID, pkt, response=False)
            self.get_logger().info(f"[CFG] {pkt.hex().upper()}")
            await asyncio.sleep(delay)


def main(args=None):
    rclpy.init(args=args)
    node = XFlyBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down X-Fly bridge.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()