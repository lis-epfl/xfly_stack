#!/usr/bin/env python3
"""
Keyboard teleop node for the X-Fly drone bridge.

Publishes normalised commands on:
    /xfly_bridge/cmd    (Vector3Stamped, .x = flapping 0–1, .y = rudder -1–1)
    /xfly_bridge/enable (Bool)

Key bindings:
    W / S          – increase / decrease flapping
    A / D          – steer left / right
    SPACE          – emergency stop (flapping → 0, rudder → 0)
    E              – toggle flight enable / disable
    R              – centre rudder
    UP / DOWN      – fine flapping adjustment
    LEFT / RIGHT   – fine rudder adjustment
    Q / ESC        – quit
"""

import sys
import curses
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Bool


# ── Tuning constants ──────────────────────────────────────────────────
FLAP_STEP_COARSE = 0.42
FLAP_STEP_FINE   = 0.01
RUDDER_STEP_COARSE = 0.5
RUDDER_STEP_FINE   = 0.05
PUBLISH_RATE_HZ  = 100


class TeleopNode(Node):
    def __init__(self):
        super().__init__("xfly_teleop")

        self.declare_parameter("bridge_ns", "/xfly_bridge")
        ns = self.get_parameter("bridge_ns").value

        # Publishers
        self._pub_cmd    = self.create_publisher(Vector3Stamped, f"{ns}/cmd", 10)
        self._pub_enable = self.create_publisher(Bool, f"{ns}/enable", 10)

        # Subscriber – connection status from the bridge
        self._connected = False
        self.create_subscription(Bool, f"{ns}/connected", self._cb_connected, 10)

        # State
        self._flapping = 0.0
        self._rudder   = 0.0
        self._enabled  = True
        self._quit     = False

        # Publish at a fixed rate
        self._timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self._publish)

    # ── Callbacks ─────────────────────────────────────────────────────

    def _cb_connected(self, msg: Bool):
        self._connected = msg.data

    def _publish(self):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = self._flapping
        msg.vector.y = self._rudder
        msg.vector.z = 0.0
        self._pub_cmd.publish(msg)

    # ── Key handling ──────────────────────────────────────────────────

    def handle_key(self, key: int) -> bool:
        """Process a curses key code. Returns False when the user wants to quit."""

        # Quit
        if key in (ord('q'), ord('Q'), 27):
            self._flapping = 0.0
            self._rudder = 0.0
            self._enabled = False
            self._publish_enable(False)
            self._publish()
            self._quit = True
            return False

        # Emergency stop
        if key == ord(' '):
            self._flapping = 0.0
            self._rudder = 0.0

        # Flapping coarse
        elif key in (ord('w'), ord('W')):
            self._flapping = min(1.0, self._flapping + FLAP_STEP_COARSE)
        elif key in (ord('s'), ord('S')):
            self._flapping = max(0.0, self._flapping - FLAP_STEP_COARSE)

        # Rudder coarse
        elif key in (ord('a'), ord('A')):
            self._rudder = max(-1.0, self._rudder - RUDDER_STEP_COARSE)
        elif key in (ord('d'), ord('D')):
            self._rudder = min(1.0, self._rudder + RUDDER_STEP_COARSE)

        # Flapping fine (arrow up/down)
        elif key == curses.KEY_UP:
            self._flapping = min(1.0, self._flapping + FLAP_STEP_FINE)
        elif key == curses.KEY_DOWN:
            self._flapping = max(0.0, self._flapping - FLAP_STEP_FINE)

        # Rudder fine (arrow left/right)
        elif key == curses.KEY_LEFT:
            self._rudder = max(-1.0, self._rudder - RUDDER_STEP_FINE)
        elif key == curses.KEY_RIGHT:
            self._rudder = min(1.0, self._rudder + RUDDER_STEP_FINE)

        # Centre rudder
        elif key in (ord('r'), ord('R')):
            self._rudder = 0.0

        # Toggle enable
        elif key in (ord('e'), ord('E')):
            self._enabled = not self._enabled
            self._publish_enable(self._enabled)
            if not self._enabled:
                self._flapping = 0.0
                self._rudder = 0.0

        return True

    def _publish_enable(self, val: bool):
        msg = Bool()
        msg.data = val
        self._pub_enable.publish(msg)

    # ── TUI rendering ─────────────────────────────────────────────────

    def draw(self, stdscr):
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        if h < 20 or w < 52:
            stdscr.addstr(0, 0, "Terminal too small – resize to at least 52x20")
            stdscr.refresh()
            return

        col = 2

        stdscr.addstr(1, col, "================================================")
        stdscr.addstr(2, col, "          X-FLY  KEYBOARD  TELEOP              ")
        stdscr.addstr(3, col, "================================================")

        # Connection / enable status
        ble_str = "* CONNECTED" if self._connected else "  DISCONNECTED"
        mode_str = "FLIGHT" if self._enabled else "DISABLED"
        ble_color = curses.color_pair(2) if self._connected else curses.color_pair(1)
        mode_color = curses.color_pair(2) if self._enabled else curses.color_pair(1)

        stdscr.addstr(5, col, "  BLE: ")
        stdscr.addstr(5, col + 7, f"{ble_str:<16}", ble_color | curses.A_BOLD)
        stdscr.addstr(5, col + 25, "Mode: ")
        stdscr.addstr(5, col + 31, f"{mode_str:<10}", mode_color | curses.A_BOLD)

        # Flapping bar
        flap_pct = self._flapping * 100
        bar_len = 30
        filled = int(round(self._flapping * bar_len))
        bar = "#" * filled + "-" * (bar_len - filled)
        stdscr.addstr(7, col, f"  Flapping: {flap_pct:5.1f}%  [{bar}]")

        # Rudder bar (centred)
        rud_pct = self._rudder * 100
        half = bar_len // 2
        rud_pos = int(round((self._rudder + 1.0) * 0.5 * bar_len))
        rud_bar = list("-" * bar_len)
        rud_bar[half] = "|"
        rud_pos = max(0, min(bar_len - 1, rud_pos))
        rud_bar[rud_pos] = "#"
        stdscr.addstr(8, col, f"  Rudder:  {rud_pct:+6.1f}%  [{''.join(rud_bar)}]")

        # Controls help
        stdscr.addstr(10, col, "  --- Controls ------------------------------------")
        stdscr.addstr(11, col, "  W / S          Flapping  +/-  (coarse)")
        stdscr.addstr(12, col, "  UP / DOWN      Flapping  +/-  (fine)")
        stdscr.addstr(13, col, "  A / D          Rudder left / right (coarse)")
        stdscr.addstr(14, col, "  LEFT / RIGHT   Rudder left / right (fine)")
        stdscr.addstr(15, col, "  R              Centre rudder")
        stdscr.addstr(16, col, "  SPACE          Emergency stop")
        stdscr.addstr(17, col, "  E              Toggle flight enable")
        stdscr.addstr(18, col, "  Q / ESC        Quit")

        stdscr.refresh()


def curses_main(stdscr, node: TeleopNode):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(50)

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_RED, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)

    while rclpy.ok() and not node._quit:
        key = stdscr.getch()
        if key != -1:
            if not node.handle_key(key):
                break
        node.draw(stdscr)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        curses.wrapper(lambda stdscr: curses_main(stdscr, node))
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Teleop shutting down.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()