#!/usr/bin/env python3
"""
MPCC Controller for XFly — 9-state, CasADi NLP (IPOPT / Knitro)
================================================================

9-state model: [px, py, pz, ψ, v, vz, az, ψ̇, ψ̈]
  2-control: [u_flap, u_rud]
  θ (path parameter) and vθ (progress speed) integrated into OCP
  ψ derived from velocity direction atan2(vy,vx), KF-filtered
  ψ̇ estimated via 2-state KF on [ψ, ψ̇] fed by velocity yaw at 240 Hz
  ψ̈ is now a STATE (1st-order lag from rudder command) — captures yaw
      inertia / coasting instability, controlled by tau_psi_ddot
  Delay compensation: ψ,ψ̇ projected forward by tunable lookahead (default 200ms)

Solver: Knitro via CasADi plugin (~5-10ms solve)
        CasADi/IPOPT fallback (~10-15ms)

Trajectory: sigmoid-smoothed piecewise line (C² for SQP Hessian).

Usage:
  python mpcc_node.py --sim --solver ipopt
  python mpcc_node.py --sim --solver knitro
  python mpcc_node.py --real --solver knitro
"""

import sys
import os
import argparse
import time
import threading

import numpy as np
import casadi as ca
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

ROS_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from std_msgs.msg import Float32, Bool, UInt8
    from geometry_msgs.msg import PoseStamped, Vector3Stamped
    from nav_msgs.msg import Path, Odometry
    from optitrack_multiplexer_ros2_msgs.msg import RigidBodyStamped
    ROS_AVAILABLE = True
except ImportError:
    pass

KNITRO_AVAILABLE = False
try:
    _test_x = ca.MX.sym('x')
    _test_nlp = {'x': _test_x, 'f': _test_x**2}
    _test_solver = ca.nlpsol('test', 'knitro', _test_nlp,
                             {'print_time': 0, 'knitro': {'outlev': 0}})
    KNITRO_AVAILABLE = True
    del _test_x, _test_nlp, _test_solver
except Exception:
    pass

# External trajectory support
try:
    from external_trajectory import ExternalTrajectory
    EXTERNAL_TRAJ_AVAILABLE = True
except ImportError:
    EXTERNAL_TRAJ_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
#  MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class XFlyParams:
    k_thrust: float = 4.07
    k_drag: float = 0.227
    v_max: float = 2.96
    thrust_eps: float = 0.02   # smoothing width for the C-inf thrust clamp (Eq. 5)
    g: float = 9.81
    gamma_slope: float = 1.0
    u_level: float = 0.91070379
    k_rudder_gamma: float = 0.0752
    k_z: float = 1.6
    k_psi_z: float = 0.075
    tau_z: float = 0.3
    wn_z: float = 4.5
    zeta_z: float = 0.2
    tau_gamma: float = 0.1
    zeta_gamma: float = 0.0
    zeta_gamma_down: float = 0.0
    zeta_blend_eps: float = 0.5
    gamma_offset_deg: float = -11.5

    k_yaw: float = -20.0          # rudder→ψ̈ gain: ψ̈_cmd = k_yaw·u_rud·v
    tau_psi_ddot: float = 0.15  # yaw-acceleration lag (s) — controls coasting
    rud_trim: float = 0.0
    k_sat: float = 0.0
    yaw_delay_comp: float = 0.0

    u_flap_min: float = 0.0
    u_flap_max: float = 1.0
    u_rud_min: float = -1.0
    u_rud_max: float = 1.0

    command_scale_gain: float = 2.5
    command_scale_min: float = 0.8
    command_scale_max: float = 1.2

    @property
    def gamma_min(self):
        return np.radians(-40.0)

    @property
    def gamma_max(self):
        return np.radians(25.0)


# ── Shared MPC weights ──
DEFAULT_MPC_WEIGHTS = {
    'contour_xy':    250.0,
    'contour_z':     250.0,
    'lag':            50.0,
    'heading':        0.0,
    'progress':        0.1,
    'control':        50.0,
    'flap_target':    30.0,
    'du_flap':         0.0,
    'du_rud':          0.0,
    'psi_dot':         0.0,
    'psi_ddot':        0.0,
}


# ═══════════════════════════════════════════════════════════════════════
#  ABLATION FLAGS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AblationFlags:
    """
    Flags to disable individual model or cost components for ablation study.
    Each flag removes exactly one design choice; all others remain identical.

    Dynamics ablations
    ------------------
    no_altitude_coupling : bool
        (i)  Remove the turn-induced altitude coupling  k_{ψz} ψ̇²  from
             the vz dynamics.  Sets the term to zero in both the OCP model
             and the RK4 simulation plant.

    no_yaw_accel_state : bool
        (ii) Collapse the 3rd-order yaw chain to 2nd order by bypassing
             the ψ̈ lag state.  The rudder command maps directly to ψ̇
             (d(ψ̇)/dt = ψ̈_cmd, ψ̈ state is frozen).

    fixed_yaw_airspeed : bool
        (iii) Replace the speed-dependent yaw command  ψ̈_cmd = k·u_rud·v
              with a fixed nominal airspeed  v̄ = v_eq(u_level).

    Cost-function ablations
    -----------------------
    no_flap_feedforward : bool
        (iv) Remove the flapping feedforward term
             q_f (u_flap,k − u_flap,k*)² from the running cost.

    no_heading_penalty : bool
        (v)  Remove the heading alignment penalty  q_ψ e_{ψ,k}²  from
             the running cost.

    Additional ablations
    --------------------
    no_lag_penalty : bool
        (vi)  Remove the lag (along-track) error penalty from the cost.

    no_contour_z_split : bool
        (vii) Merge the separate XY and Z contouring weights into a single
              3D contouring term  (use contour_xy weight for the full 3D
              contouring error, set contour_z weight to zero).

    euler_integration : bool
        (viii) Use forward-Euler instead of RK4 in the simulation plant
               (the OCP always uses Euler; this tests plant–model mismatch).
    """
    no_altitude_coupling:  bool = False   # (i)
    no_yaw_accel_state:    bool = False   # (ii)
    fixed_yaw_airspeed:    bool = False   # (iii)
    no_flap_feedforward:   bool = False   # (iv)
    no_heading_penalty:    bool = False   # (v)
    no_lag_penalty:        bool = False   # (vi)
    no_contour_z_split:    bool = False   # (vii)
    euler_integration:     bool = False   # (viii)

    def active_labels(self):
        """Return list of human-readable labels for active ablations."""
        labels = []
        if self.no_altitude_coupling:  labels.append('(i) no altitude coupling')
        if self.no_yaw_accel_state:    labels.append('(ii) 2nd-order yaw')
        if self.fixed_yaw_airspeed:    labels.append('(iii) fixed yaw airspeed')
        if self.no_flap_feedforward:   labels.append('(iv) no flap feedforward')
        if self.no_heading_penalty:    labels.append('(v) no heading penalty')
        if self.no_lag_penalty:        labels.append('(vi) no lag penalty')
        if self.no_contour_z_split:    labels.append('(vii) merged 3D contour')
        if self.euler_integration:     labels.append('(viii) Euler sim plant')
        return labels

    def tag(self):
        """Short tag string for filenames / plot titles."""
        parts = []
        if self.no_altitude_coupling:  parts.append('noAltCoup')
        if self.no_yaw_accel_state:    parts.append('2ndYaw')
        if self.fixed_yaw_airspeed:    parts.append('fixVyaw')
        if self.no_flap_feedforward:   parts.append('noFlapFF')
        if self.no_heading_penalty:    parts.append('noHdg')
        if self.no_lag_penalty:        parts.append('noLag')
        if self.no_contour_z_split:    parts.append('merged3D')
        if self.euler_integration:     parts.append('eulerSim')
        return '+'.join(parts) if parts else 'baseline'


# ═══════════════════════════════════════════════════════════════════════
#  STRAIGHT LINE TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════

class StraightLineTrajectory:
    def __init__(self,
                 start=(-3.6, 2.7, 0.1),
                 mid=(0.0, 0.0, 0.5),
                 end=(3.6, -2.7, 0.9)):

        self.P0 = np.array(start)
        self.P1 = np.array(mid)
        self.P2 = np.array(end)

        d1 = self.P1 - self.P0
        self.L1 = np.linalg.norm(d1)
        self.dir1 = d1 / self.L1
        self.gamma1 = np.arctan2(d1[2], np.sqrt(d1[0]**2 + d1[1]**2))

        d2 = self.P2 - self.P1
        self.L2 = np.linalg.norm(d2)
        self.dir2 = d2 / self.L2
        self.gamma2 = np.arctan2(d2[2], np.sqrt(d2[0]**2 + d2[1]**2))

        self.s1 = self.L1
        self.total_arc = self.L1 + self.L2

        print(f"Straight line trajectory:")
        print(f"  1. Climb:  {self.L1:.2f}m, γ={np.degrees(self.gamma1):.1f}°")
        print(f"     {start} → {mid}")
        print(f"  2. Level:  {self.L2:.2f}m, γ={np.degrees(self.gamma2):.1f}°")
        print(f"     {mid} → {end}")
        print(f"  Total: {self.total_arc:.2f}m")

    def position(self, s):
        frac1 = s / self.L1
        x1 = self.P0[0] + frac1 * (self.P1[0] - self.P0[0])
        y1 = self.P0[1] + frac1 * (self.P1[1] - self.P0[1])
        z1 = self.P0[2] + frac1 * (self.P1[2] - self.P0[2])
        frac2 = (s - self.s1) / self.L2
        x2 = self.P1[0] + frac2 * (self.P2[0] - self.P1[0])
        y2 = self.P1[1] + frac2 * (self.P2[1] - self.P1[1])
        z2 = self.P1[2] + frac2 * (self.P2[2] - self.P1[2])
        return (ca.if_else(s < self.s1, x1, x2),
                ca.if_else(s < self.s1, y1, y2),
                ca.if_else(s < self.s1, z1, z2))

    def tangent(self, s):
        return (ca.if_else(s < self.s1, self.dir1[0], self.dir2[0]),
                ca.if_else(s < self.s1, self.dir1[1], self.dir2[1]),
                ca.if_else(s < self.s1, self.dir1[2], self.dir2[2]))

    def climb_angle_ca(self, s):
        return ca.if_else(s < self.s1, self.gamma1, self.gamma2)

    def position_np(self, s):
        s = float(s)
        if s < self.s1:
            return self.P0 + (s / self.L1) * (self.P1 - self.P0)
        else:
            return self.P1 + ((s - self.s1) / self.L2) * (self.P2 - self.P1)

    def tangent_np(self, s):
        return self.dir1.copy() if float(s) < self.s1 else self.dir2.copy()

    def heading(self, s):
        t = self.tangent_np(s)
        return np.arctan2(t[1], t[0])

    def climb_angle_np(self, s):
        return self.gamma1 if float(s) < self.s1 else self.gamma2

    def positions_array(self, s_arr):
        return np.array([self.position_np(s) for s in s_arr])

    def project_to_path(self, pos, theta_guess, search_radius=2.0,
                        n_samples=50):
        s_lo = max(0.0, theta_guess - search_radius)
        s_hi = min(self.total_arc, theta_guess + search_radius)
        best_s, best_d2 = theta_guess, np.inf
        for s in np.linspace(s_lo, s_hi, n_samples):
            d2 = np.sum((pos - self.position_np(s))**2)
            if d2 < best_d2:
                best_d2 = d2
                best_s = s
        return best_s, np.sqrt(best_d2)


class CompositeTrajectory:
    """
    Three-segment trajectory parameterized by arc length θ:
      Seg 1: straight line  (approach + climb)
      Seg 2: helical climb  (CW, constant climb angle)
      Seg 3: level circles  (CW, γ=0)
    """

    def __init__(self,
                 start=(-3.7, 1.5, -0.11),
                 join=(0.0, 1.5, 0.3),
                 center=(0.0, 0.0),
                 z_top=1.5,
                 radius=1.5,
                 n_climb_rev=2.0,
                 n_level_rev=3.0,
                 direction=1):

        self.P0 = np.array(start)
        self.P1 = np.array(join)
        self.R = radius
        self.d = direction
        self.z_join = join[2]
        self.z_top = z_top
        self.n_climb_rev = n_climb_rev
        self.n_level_rev = n_level_rev

        d = self.P1 - self.P0
        self.L1 = np.linalg.norm(d)
        self.dir1 = d / self.L1
        self.gamma1 = np.arctan2(d[2], np.sqrt(d[0]**2 + d[1]**2))
        self.gamma_approach = self.gamma1

        dz_climb = z_top - join[2]
        phi_climb = 2 * np.pi * n_climb_rev
        self.dz_per_rad = dz_climb / phi_climb
        self.arc_per_rad_climb = np.sqrt(radius**2 + self.dz_per_rad**2)
        self.L2 = phi_climb * self.arc_per_rad_climb
        self.gamma_helix = np.arctan2(self.dz_per_rad, radius)
        self.phi_end_climb = phi_climb

        phi_level = 2 * np.pi * n_level_rev
        self.L3 = phi_level * radius
        self.phi_end_total = self.phi_end_climb + phi_level

        self.total_arc = self.L1 + self.L2 + self.L3
        self.s1 = self.L1
        self.s2 = self.L1 + self.L2

        dir_label = "CW" if direction == 1 else "CCW"
        print(f"Composite trajectory ({dir_label}):")
        print(f"  1. Approach: {self.L1:.2f}m, γ={np.degrees(self.gamma1):.1f}°")
        print(f"     {tuple(self.P0)} → {tuple(self.P1)}")
        print(f"  2. Helix:   {self.L2:.2f}m, {n_climb_rev} rev, "
              f"γ={np.degrees(self.gamma_helix):.1f}°, R={radius}m")
        print(f"  3. Circles: {self.L3:.2f}m, {n_level_rev} rev, γ=0°")
        print(f"  Total: {self.total_arc:.2f}m")

    def position(self, s):
        frac = s / self.L1
        x1 = self.P0[0] + frac * (self.P1[0] - self.P0[0])
        y1 = self.P0[1] + frac * (self.P1[1] - self.P0[1])
        z1 = self.P0[2] + frac * (self.P1[2] - self.P0[2])

        phi2 = (s - self.s1) / self.arc_per_rad_climb
        x2 = self.R * ca.sin(phi2)
        y2 = self.d * self.R * ca.cos(phi2)
        z2 = self.z_join + phi2 * self.dz_per_rad

        phi3 = self.phi_end_climb + (s - self.s2) / self.R
        x3 = self.R * ca.sin(phi3)
        y3 = self.d * self.R * ca.cos(phi3)
        z3 = self.z_top

        x = ca.if_else(s < self.s1, x1, ca.if_else(s < self.s2, x2, x3))
        y = ca.if_else(s < self.s1, y1, ca.if_else(s < self.s2, y2, y3))
        z = ca.if_else(s < self.s1, z1, ca.if_else(s < self.s2, z2, z3))
        return (x, y, z)

    def tangent(self, s):
        tx1, ty1, tz1 = self.dir1[0], self.dir1[1], self.dir1[2]

        phi2 = (s - self.s1) / self.arc_per_rad_climb
        raw_x2 = self.R * ca.cos(phi2)
        raw_y2 = -self.d * self.R * ca.sin(phi2)
        raw_z2 = self.dz_per_rad
        norm2 = self.arc_per_rad_climb
        tx2, ty2, tz2 = raw_x2/norm2, raw_y2/norm2, raw_z2/norm2

        phi3 = self.phi_end_climb + (s - self.s2) / self.R
        tx3 = ca.cos(phi3)
        ty3 = -self.d * ca.sin(phi3)
        tz3 = 0.0

        tx = ca.if_else(s < self.s1, tx1, ca.if_else(s < self.s2, tx2, tx3))
        ty = ca.if_else(s < self.s1, ty1, ca.if_else(s < self.s2, ty2, ty3))
        tz = ca.if_else(s < self.s1, tz1, ca.if_else(s < self.s2, tz2, tz3))
        return (tx, ty, tz)

    def climb_angle_ca(self, s):
        return ca.if_else(s < self.s1, self.gamma1,
                   ca.if_else(s < self.s2, self.gamma_helix, 0.0))

    def position_np(self, s):
        s = float(s)
        if s < self.s1:
            return self.P0 + (s / self.L1) * (self.P1 - self.P0)
        elif s < self.s2:
            phi = (s - self.s1) / self.arc_per_rad_climb
            return np.array([self.R * np.sin(phi),
                             self.d * self.R * np.cos(phi),
                             self.z_join + phi * self.dz_per_rad])
        else:
            phi = self.phi_end_climb + (s - self.s2) / self.R
            return np.array([self.R * np.sin(phi),
                             self.d * self.R * np.cos(phi),
                             self.z_top])

    def tangent_np(self, s):
        s = float(s)
        if s < self.s1:
            return self.dir1.copy()
        elif s < self.s2:
            phi = (s - self.s1) / self.arc_per_rad_climb
            raw = np.array([self.R * np.cos(phi),
                           -self.d * self.R * np.sin(phi),
                            self.dz_per_rad])
            return raw / np.linalg.norm(raw)
        else:
            phi = self.phi_end_climb + (s - self.s2) / self.R
            return np.array([np.cos(phi), -self.d * np.sin(phi), 0.0])

    def heading(self, s):
        t = self.tangent_np(s)
        return np.arctan2(t[1], t[0])

    def climb_angle_np(self, s):
        s = float(s)
        if s < self.s1:
            return self.gamma1
        elif s < self.s2:
            return self.gamma_helix
        else:
            return 0.0

    def positions_array(self, s_arr):
        return np.array([self.position_np(s) for s in s_arr])

    def project_to_path(self, pos, theta_guess, search_radius=2.0,
                        n_samples=50):
        s_lo = max(0.0, theta_guess - search_radius)
        s_hi = min(self.total_arc, theta_guess + search_radius)
        best_s, best_d2 = theta_guess, np.inf
        for s in np.linspace(s_lo, s_hi, n_samples):
            d2 = np.sum((pos - self.position_np(s))**2)
            if d2 < best_d2:
                best_d2 = d2
                best_s = s
        return best_s, np.sqrt(best_d2)


# ═══════════════════════════════════════════════════════════════════════
#  MPCC CONTROLLER — 9-state (CasADi NLP: IPOPT or Knitro)
# ═══════════════════════════════════════════════════════════════════════

class MPCController:
    """
    CasADi-based MPCC using either IPOPT or Knitro as the NLP backend.

    9-state model: [px, py, pz, ψ, v, vz, az, ψ̇, ψ̈]
    2-control: [u_flap, u_rud]

    ψ̈ is a state driven by a first-order lag from the rudder command:
        d(ψ̈)/dt = (ψ̈_cmd − ψ̈) / τ_psi_ddot
    where ψ̈_cmd = k_eff · u_rud · v / τ_yaw  (old algebraic expression).
    τ_psi_ddot controls how slowly yaw acceleration settles — captures
    the real-world inertia / coasting instability after a rudder kick.
    """

    KNITRO_ALG_NAMES = {
        0: 'Auto',
        1: 'Interior/Direct',
        2: 'Interior/CG',
        3: 'Active Set',
        4: 'SQP',
        5: 'Multi',
    }

    def __init__(self, xfly: XFlyParams, traj: StraightLineTrajectory,
                 horizon: int = 10, dt: float = 0.10,
                 solver_name: str = 'ipopt', knitro_alg: int = 1,
                 ablation: AblationFlags = None):
        self.params = xfly
        self.traj = traj
        self.N = horizon
        self.dt = dt
        self.nx = 9   # [px, py, pz, ψ, v, vz, az, ψ̇, ψ̈]
        self.nu = 2
        self.solver_name = solver_name.lower()
        self.knitro_alg = knitro_alg
        self.ablation = ablation or AblationFlags()

        if self.solver_name not in ('ipopt', 'knitro'):
            raise ValueError(
                f"MPCController supports 'ipopt' or 'knitro', "
                f"got '{solver_name}'")

        self.weights = dict(DEFAULT_MPC_WEIGHTS)
        self._build_ocp()

    def _build_ocp(self):
        N = self.N
        nx, nu = self.nx, self.nu
        dt = self.dt
        p = self.params
        traj = self.traj
        w = self.weights
        abl = self.ablation

        # ── Precompute fixed nominal airspeed for ablation (iii) ──
        v_bar = compute_v_eq(p, p.u_level)  # equilibrium speed at level flight

        X = ca.MX.sym('X', nx, N + 1)
        U = ca.MX.sym('U', nu, N)
        Theta = ca.MX.sym('Theta', N + 1)
        Vtheta = ca.MX.sym('Vtheta', N)
        # Param: [x0 (9), theta0 (1), u_flap_prev (1), u_rud_prev (1), u_level (1)] = 13
        Param = ca.MX.sym('Param', nx + 1 + 2 + 1)
        u_level_p = Param[nx + 3]  # symbolic u_level from param vector

        obj = 0
        g = []

        g.append(X[:, 0] - Param[:nx])
        g.append(Theta[0] - Param[nx])

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            thetak = Theta[k]
            vthetak = Vtheta[k]

            # ── Unpack 9-state ──
            px, py, pz   = xk[0], xk[1], xk[2]
            psi          = xk[3]
            v            = xk[4]
            vz           = xk[5]
            az           = xk[6]
            psi_dot      = xk[7]
            psi_ddot     = xk[8]   # ψ̈ is now a STATE

            u_flap, u_rud = uk[0], uk[1]

            ref_x, ref_y, ref_z = traj.position(thetak)
            tx, ty, tz = traj.tangent(thetak)
            gamma_traj = traj.climb_angle_ca(thetak)

            # 3D contouring error decomposition
            ex, ey, ez = px - ref_x, py - ref_y, pz - ref_z
            e_lag = ex * tx + ey * ty + ez * tz
            ecx = ex - e_lag * tx
            ecy = ey - e_lag * ty
            ecz = ez - e_lag * tz
            e_cont_xy_sq = ecx**2 + ecy**2 + 1e-6
            e_cont_z_sq  = ecz**2 + 1e-6
            psi_ref = ca.atan2(ty, tx)
            e_psi = ca.atan2(ca.sin(psi - psi_ref),
                             ca.cos(psi - psi_ref))

            # ── Cost — with ablation switches ──
            # Ablation (vii): merged 3D contour → use contour_xy weight for
            #   full 3D error, skip separate Z weight
            if abl.no_contour_z_split:
                e_cont_3d_sq = ecx**2 + ecy**2 + ecz**2 + 1e-6
                obj += w['contour_xy'] * e_cont_3d_sq
            else:
                obj += w['contour_xy'] * e_cont_xy_sq
                obj += w['contour_z']  * e_cont_z_sq

            # Ablation (vi): no lag penalty
            if not abl.no_lag_penalty:
                obj += w['lag'] * e_lag**2

            # Ablation (v): no heading penalty
            if not abl.no_heading_penalty:
                obj += w['heading'] * e_psi**2

            obj -= w['progress'] * vthetak

            # Ablation (iv): no flap feedforward
            if not abl.no_flap_feedforward:
                u_flap_target = u_level_p + gamma_traj / p.gamma_slope
                obj += w['flap_target'] * (u_flap - u_flap_target)**2
            obj += w['control'] * u_rud**2

            # Control rate penalty (du)
            if k == 0:
                du_flap = u_flap - Param[nx + 1]
                du_rud  = u_rud  - Param[nx + 2]
            else:
                du_flap = U[0, k] - U[0, k - 1]
                du_rud  = U[1, k] - U[1, k - 1]
            obj += w['du_flap'] * du_flap**2
            obj += w['du_rud']  * du_rud**2

            # ψ̇ and ψ̈ penalty (ψ̈ is now the state, not the command)
            obj += w['psi_dot']  * psi_dot**2
            obj += w['psi_ddot'] * psi_ddot**2

            # ── Dynamics — 9-state [px, py, pz, ψ, v, vz, az, ψ̇, ψ̈] ──
            # C-inf smooth clamp of max(0, 1 - v/v_max): softplus-type rectifier
            # s_eps(x) = 0.5(x + sqrt(x^2 + eps^2)) -> max(0,x) as eps->0. Keeps
            # the forward-speed dynamics continuously differentiable at v=v_max
            # for the gradient-based solver.
            # Original (hard, C0 kink at v_max):
            #     thrust_factor = ca.fmax(0, 1 - v / p.v_max)
            _tf = 1 - v / p.v_max
            thrust_factor = 0.5 * (_tf + ca.sqrt(_tf * _tf + p.thrust_eps**2))
            v_dot = p.k_thrust * u_flap * thrust_factor - p.k_drag * v

            # Ablation (iii): fixed airspeed in yaw command
            v_yaw = v_bar if abl.fixed_yaw_airspeed else v

            # ψ̈_cmd: what the rudder is commanding (old algebraic expression)
            k_eff        = p.k_yaw / (1 + p.k_sat * psi_dot**2)
            psi_ddot_cmd = k_eff * (u_rud + p.rud_trim) * v_yaw

            # Ablation (ii): 2nd-order yaw — bypass ψ̈ lag, map rudder→ψ̇
            if abl.no_yaw_accel_state:
                # d(ψ̇)/dt = ψ̈_cmd directly;  ψ̈ state frozen at 0
                psi_dot_dot  = psi_ddot_cmd
                psi_ddot_dot = -psi_ddot / dt  # drive toward zero
            else:
                # Full 3rd-order chain: ψ̈ lags toward command
                psi_dot_dot  = psi_ddot
                psi_ddot_dot = (psi_ddot_cmd - psi_ddot) / p.tau_psi_ddot

            px_dot  = v * ca.cos(psi)
            py_dot  = v * ca.sin(psi)
            pz_dot  = vz

            # Ablation (i): no altitude coupling
            if abl.no_altitude_coupling:
                vz_dot = az
            else:
                vz_dot = az - p.k_psi_z * psi_dot**2

            az_dot  = p.wn_z**2 * (p.k_z * (u_flap - u_level_p) - vz) \
                      - 2 * p.zeta_z * p.wn_z * az

            xk_next = xk + dt * ca.vertcat(
                px_dot,        # d(px)/dt
                py_dot,        # d(py)/dt
                pz_dot,        # d(pz)/dt
                psi_dot,       # d(ψ)/dt  = ψ̇  (state[7])
                v_dot,         # d(v)/dt
                vz_dot,        # d(vz)/dt
                az_dot,        # d(az)/dt
                psi_dot_dot,   # d(ψ̇)/dt — ablation (ii) aware
                psi_ddot_dot,  # d(ψ̈)/dt — ablation (ii) aware
            )

            g.append(X[:, k + 1] - xk_next)
            g.append(Theta[k + 1] - (Theta[k] + dt * vthetak))

        opt_vars = ca.vertcat(
            ca.reshape(X, -1, 1), ca.reshape(U, -1, 1),
            Theta, Vtheta)

        lbx, ubx = [], []
        for _ in range(N + 1):
            # 9-state: [px,py,pz,ψ,v,vz,az,ψ̇,ψ̈]
            lbx.extend([-60, -60, -0.5, -np.inf, 0.05, -5.0, -20.0, -10.0, -30.0])
            ubx.extend([ 60,  60,  40,   np.inf, p.v_max * 1.1, 5.0, 20.0, 10.0, 30.0])
        for _ in range(N):
            lbx.extend([p.u_flap_min, p.u_rud_min])
            ubx.extend([p.u_flap_max, p.u_rud_max])
        for _ in range(N + 1):
            lbx.append(-0.5)
            ubx.append(traj.total_arc * 1.1)
        for _ in range(N):
            lbx.append(0.0)
            ubx.append(5.0)

        n_eq = nx + 1 + N * (nx + 1)

        nlp = {'x': opt_vars, 'f': obj, 'g': ca.vertcat(*g), 'p': Param}

        if self.solver_name == 'knitro':
            alg_name = self.KNITRO_ALG_NAMES.get(self.knitro_alg,
                                                  f'Unknown({self.knitro_alg})')
            opts = {
                'knitro': {
                    'outlev': 0,
                    'maxit': 200,
                    'feastol': 1e-4,
                    'opttol': 1e-4,
                    'bar_initpt': 3,
                    'nlp_algorithm': self.knitro_alg,
                    'linsolver': 0,
                },
                'print_time': 0,
                'warn_initial_bounds': False,
            }
            self.solver = ca.nlpsol('mpcc', 'knitro', nlp, opts)
        else:
            opts = {
                'ipopt.print_level': 0,
                'ipopt.max_iter': 200,
                'ipopt.tol': 1e-4,
                'ipopt.warm_start_init_point': 'yes',
                'print_time': 0,
            }
            self.solver = ca.nlpsol('mpcc', 'ipopt', nlp, opts)

        self.lbx = np.array(lbx)
        self.ubx = np.array(ubx)
        self.lbg = np.zeros(n_eq)
        self.ubg = np.zeros(n_eq)
        self.dims = {
            'n_X': nx * (N + 1), 'n_U': nu * N,
            'n_Theta': N + 1, 'n_Vtheta': N,
        }

    def solve(self, x0, theta0, u_prev, warm_start=None):
        x0_int = x0.copy()
        p = self.params
        param = np.concatenate([x0_int, [theta0], u_prev[:2], [p.u_level]])

        if not np.all(np.isfinite(param)):
            raise ValueError(f"Non-finite param: {param}")

        if warm_start is None:
            N, nx, nu = self.N, self.nx, self.nu
            gamma_now = self.traj.climb_angle_np(theta0)
            X_init = np.tile(x0_int.reshape(-1, 1), (1, N + 1))
            u_flap_target = p.u_level + gamma_now / p.gamma_slope
            u_init = np.array([u_flap_target, u_prev[1]])
            U_init = np.tile(u_init.reshape(-1, 1), (1, N))
            Theta_init = theta0 + np.arange(N + 1) * self.dt * 2.0
            Vtheta_init = 2.0 * np.ones(N)
            warm_start = np.concatenate([
                X_init.flatten(order='F'), U_init.flatten(order='F'),
                Theta_init, Vtheta_init])

        expected_len = (self.nx * (self.N + 1)
                        + self.nu * self.N
                        + (self.N + 1) + self.N)
        if len(warm_start) != expected_len:
            raise ValueError(
                f"warm_start length {len(warm_start)} "
                f"!= expected {expected_len} — resetting")

        if not np.all(np.isfinite(warm_start)):
            warm_start = None
            return self.solve(x0, theta0, u_prev, None)

        param = np.asarray(param, dtype=np.float64)
        warm_start = np.asarray(warm_start, dtype=np.float64)

        t_start = time.perf_counter()
        sol = self.solver(x0=warm_start, lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg, p=param)
        t_solve = time.perf_counter() - t_start

        sol_x = sol['x'].full().flatten()
        dd = self.dims
        X_sol = sol_x[:dd['n_X']].reshape(self.nx, self.N + 1, order='F')
        U_sol = sol_x[dd['n_X']:dd['n_X'] + dd['n_U']].reshape(
            self.nu, self.N, order='F')
        Theta_sol = sol_x[dd['n_X'] + dd['n_U']:
                          dd['n_X'] + dd['n_U'] + dd['n_Theta']]
        Vtheta_sol = sol_x[dd['n_X'] + dd['n_U'] + dd['n_Theta']:]

        return {
            'u': U_sol[:, 0],
            'v_theta': Vtheta_sol[0],
            'X': X_sol, 'U': U_sol, 'Theta': Theta_sol,
            'warm': sol_x,
            'cost': float(sol['f']),
            'solve_time': t_solve,
        }


# ═══════════════════════════════════════════════════════════════════════
#  KALMAN VELOCITY ESTIMATOR (for speed)
# ═══════════════════════════════════════════════════════════════════════

class KalmanVelocityEstimator:
    """
    Constant-velocity KF: [x, y, z, vx, vy, vz]
    Used for: position, velocity, speed.
    NOT used for yaw — quaternion is used instead.
    """

    def __init__(self, sigma_accel=1.0, sigma_pos=0.05):
        self.sigma_a = sigma_accel
        self.sigma_p = sigma_pos
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1.0
        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)
        self.R = np.eye(3) * sigma_pos**2
        self._initialized = False
        self._last_time = None

    def initialize(self, position, t):
        self.x[:3] = position
        self.x[3:] = 0.0
        self.P = np.diag([
            self.sigma_p**2, self.sigma_p**2, self.sigma_p**2,
            1.0, 1.0, 1.0])
        self._last_time = t
        self._initialized = True

    @property
    def initialized(self):
        return self._initialized

    def predict_update(self, position, t):
        if not self._initialized:
            self.initialize(position, t)
            return
        dt = t - self._last_time
        if dt <= 0:
            return
        if dt > 0.5:
            self.initialize(position, t)
            return
        self._last_time = t

        F = np.eye(6)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        sa2 = self.sigma_a**2
        q11 = (dt**3 / 3.0) * sa2
        q12 = (dt**2 / 2.0) * sa2
        q22 = dt * sa2
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = q11; Q[i, i+3] = q12
            Q[i+3, i] = q12; Q[i+3, i+3] = q22

        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q
        y = np.array(position) - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(6) - K @ self.H) @ P_pred

    @property
    def pos(self):
        return self.x[:3].copy()

    @property
    def vel(self):
        return self.x[3:].copy()

    @property
    def speed(self):
        return np.linalg.norm(self.x[3:])


# ═══════════════════════════════════════════════════════════════════════
#  YAW ESTIMATOR — velocity-direction based, with delay compensation
# ═══════════════════════════════════════════════════════════════════════

class YawRateEstimator:
    """
    Velocity-direction yaw estimator with delay compensation.

    Yaw is derived from atan2(vy, vx) instead of the mocap quaternion.
    A 2-state KF on [ψ, ψ̇] smooths the velocity-derived yaw measurement.

    At low speeds (below `min_speed`), the quaternion yaw from mocap is
    used as the measurement source instead, since atan2 is unreliable
    near zero velocity.  If no quaternion is supplied, the filter coasts
    on its prediction.

    Delay compensation projects the filtered [ψ, ψ̇] forward by
    `delay_comp_s` seconds:
        ψ_comp  = ψ + ψ̇ · delay_comp_s
        ψ̇_comp = ψ̇   (constant-rate assumption)

    Parameters
    ----------
    sigma_yaw_accel : float
        Process noise on yaw acceleration (rad/s²).
    sigma_yaw_meas : float
        Measurement noise for velocity-derived yaw (rad).
    sigma_yaw_meas_quat : float
        Measurement noise for quaternion yaw fallback (rad).
    min_speed : float
        Below this speed (m/s) the filter switches from velocity yaw
        to quaternion yaw.  Default 0.3 m/s.
    delay_comp_s : float
        Delay compensation lookahead (seconds).  Default 0.2 s.
    """

    def __init__(self, sigma_yaw_accel=5.0, sigma_yaw_meas=0.001,
                 sigma_yaw_meas_quat=0.01, min_speed=0.3,
                 delay_comp_s=0.2):
        self.sigma_a = sigma_yaw_accel
        self.sigma_m = sigma_yaw_meas
        self.sigma_m_quat = sigma_yaw_meas_quat
        self.min_speed = min_speed
        self.delay_comp_s = delay_comp_s

        self.x = np.zeros(2)          # [ψ, ψ̇]
        self.P = np.diag([0.1, 1.0])
        self.H = np.array([[1.0, 0.0]])
        self.R_base = sigma_yaw_meas**2
        self._initialized = False
        self._last_time = None
        self._using_quat = False

    def initialize(self, yaw, t):
        self.x[0] = yaw
        self.x[1] = 0.0
        self.P = np.diag([self.sigma_m**2, 1.0])
        self._last_time = t
        self._initialized = True

    @property
    def initialized(self):
        return self._initialized

    def update(self, vx, vy, t, quat_yaw=None):
        speed = np.sqrt(vx**2 + vy**2)
        yaw_vel = np.arctan2(vy, vx)

        if not self._initialized:
            init_yaw = quat_yaw if quat_yaw is not None else yaw_vel
            self.initialize(init_yaw, t)
            return
        dt = t - self._last_time
        if dt <= 0:
            return
        if dt > 0.5:
            init_yaw = quat_yaw if quat_yaw is not None else yaw_vel
            self.initialize(init_yaw, t)
            return
        self._last_time = t

        F = np.array([[1.0, dt], [0.0, 1.0]])
        sa2 = self.sigma_a**2
        Q = np.array([
            [dt**3/3 * sa2, dt**2/2 * sa2],
            [dt**2/2 * sa2, dt * sa2],
        ])

        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        if speed < self.min_speed and quat_yaw is not None:
            yaw_meas = quat_yaw
            R_eff = self.sigma_m_quat**2
            self._using_quat = True
        elif speed > self.min_speed:
            yaw_meas = yaw_vel
            R_eff = self.R_base
            self._using_quat = False
        else:
            yaw_meas = yaw_vel
            ratio = max(speed, 1e-6) / self.min_speed
            R_eff = self.R_base / (ratio**2 + 1e-6)
            R_eff = min(R_eff, self.R_base * 1000.0)
            self._using_quat = False

        R = np.array([[R_eff]])
        y_innov = np.arctan2(np.sin(yaw_meas - x_pred[0]),
                             np.cos(yaw_meas - x_pred[0]))
        S = self.H @ P_pred @ self.H.T + R
        K = P_pred @ self.H.T / S[0, 0]
        self.x = x_pred + K.flatten() * y_innov
        self.P = (np.eye(2) - K @ self.H) @ P_pred

    @property
    def yaw(self):
        if self._using_quat:
            return self.x[0]
        return self.x[0] + self.x[1] * self.delay_comp_s

    @property
    def yaw_raw(self):
        return self.x[0]

    @property
    def yaw_rate(self):
        return self.x[1]


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

def quat_to_yaw(qx, qy, qz, qw):
    """Quaternion → yaw (ZYX Euler convention)."""
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy**2 + qz**2))


def compute_v_eq(p, u_flap):
    """Equilibrium speed for given flap."""
    a = p.k_thrust * u_flap / p.v_max + p.k_drag
    b = p.k_thrust * u_flap
    return max(0.5, b / a) if a > 0 else 2.0


def simulate_step(x, u, dt, p, disturbance_fn=None, ablation=None):
    """RK4 integration of the 9-state model with optional disturbance.

    State: [px, py, pz, ψ, v, vz, az, ψ̇, ψ̈]
    Yaw dynamics (3rd order):
        d(ψ)/dt  = ψ̇
        d(ψ̇)/dt  = ψ̈
        d(ψ̈)/dt  = (ψ̈_cmd − ψ̈) / τ_psi_ddot
        where ψ̈_cmd = k_eff · u_rud · v / τ_yaw

    Ablation flags modify dynamics when active (see AblationFlags).
    """
    abl = ablation or AblationFlags()
    v_bar = compute_v_eq(p, p.u_level)  # fixed nominal speed for abl (iii)

    def f(x, u):
        px, py, pz, psi, v, vz, az, psi_dot, psi_ddot = x
        u_flap = np.clip(u[0], p.u_flap_min, p.u_flap_max)
        u_rud  = np.clip(u[1], p.u_rud_min,  p.u_rud_max)

        # C-inf smooth clamp (matches the OCP model above), eps = p.thrust_eps
        # Original (hard): thrust_factor = max(0, 1 - v / p.v_max)
        _tf = 1 - v / p.v_max
        thrust_factor = 0.5 * (_tf + np.sqrt(_tf * _tf + p.thrust_eps**2))
        v_dot = p.k_thrust * u_flap * thrust_factor - p.k_drag * v

        vz_target = p.k_z * (u_flap - p.u_level)
        az_dot = p.wn_z**2 * (vz_target - vz) - 2 * p.zeta_z * p.wn_z * az

        # Ablation (iii): fixed airspeed in yaw command
        v_yaw = v_bar if abl.fixed_yaw_airspeed else v

        k_eff        = p.k_yaw / (1 + p.k_sat * psi_dot**2)
        psi_ddot_cmd = k_eff * (u_rud + p.rud_trim) * v_yaw

        # Ablation (ii): 2nd-order yaw (bypass lag state)
        if abl.no_yaw_accel_state:
            psi_dot_dot  = psi_ddot_cmd
            psi_ddot_dot = -psi_ddot / max(dt, 1e-4)
        else:
            psi_dot_dot  = psi_ddot
            psi_ddot_dot = (psi_ddot_cmd - psi_ddot) / p.tau_psi_ddot

        # Ablation (i): no altitude coupling
        if abl.no_altitude_coupling:
            vz_dot = az
        else:
            vz_dot = az - p.k_psi_z * psi_dot**2

        return np.array([
            v * np.cos(psi),           # d(px)/dt
            v * np.sin(psi),           # d(py)/dt
            vz,                        # d(pz)/dt
            psi_dot,                   # d(ψ)/dt
            v_dot,                     # d(v)/dt
            vz_dot,                    # d(vz)/dt
            az_dot,                    # d(az)/dt
            psi_dot_dot,               # d(ψ̇)/dt — ablation (ii) aware
            psi_ddot_dot,              # d(ψ̈)/dt — ablation (ii) aware
        ])

    # Ablation (viii): Euler integration instead of RK4
    if abl.euler_integration:
        x_new = x + dt * f(x, u)
    else:
        k1 = f(x, u)
        k2 = f(x + 0.5*dt*k1, u)
        k3 = f(x + 0.5*dt*k2, u)
        k4 = f(x + dt*k3, u)
        x_new = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    if disturbance_fn is not None:
        d = disturbance_fn(0.0, x)
        x_new[0] += d[0] * dt
        x_new[1] += d[1] * dt
        x_new[2] += d[2] * dt
        x_new[4] += d[3] * dt
        x_new[7] += d[4] * dt

    return x_new


# ── Disturbance models ──

class DisturbanceNone:
    name = 'none'
    label = 'No disturbance'
    def __call__(self, t, x):
        return np.zeros(5)


class DisturbanceWind:
    name = 'wind'
    label = 'Crosswind (0.3 m/s)'

    def __init__(self, wind_speed=0.3, wind_heading=np.pi/2):
        self.wx = wind_speed * np.cos(wind_heading)
        self.wy = wind_speed * np.sin(wind_heading)

    def __call__(self, t, x):
        psi = x[3]
        v_wind = self.wx * np.cos(psi) + self.wy * np.sin(psi)
        v_cross = -self.wx * np.sin(psi) + self.wy * np.cos(psi)
        return np.array([
            self.wx, self.wy, 0.0,
            -v_wind * 0.3, v_cross * 0.1,
        ])


class DisturbanceGust:
    name = 'gust'
    label = 'Gusts (0.5 m/s, 1Hz)'

    def __init__(self, intensity=0.5, freq=1.0, seed=42):
        self.intensity = intensity
        self.freq = freq
        self.rng = np.random.RandomState(seed)
        self.gusts = []
        t = 0.5
        while t < 30.0:
            t += self.rng.exponential(1.0 / freq)
            dur = 0.1 + self.rng.uniform(0, 0.3)
            angle = self.rng.uniform(0, 2*np.pi)
            vz = self.rng.uniform(-0.3, 0.3)
            self.gusts.append((t, intensity * np.cos(angle),
                               intensity * np.sin(angle), vz * intensity, dur))

    def __call__(self, t, x):
        dx, dy, dz = 0.0, 0.0, 0.0
        for gt, gx, gy, gz, dur in self.gusts:
            if gt <= t < gt + dur:
                phase = (t - gt) / dur * np.pi
                env = np.sin(phase) ** 2
                dx += gx * env
                dy += gy * env
                dz += gz * env
        psi = x[3]
        v_wind = dx * np.cos(psi) + dy * np.sin(psi)
        v_cross = -dx * np.sin(psi) + dy * np.cos(psi)
        return np.array([dx, dy, dz, -v_wind * 0.3, v_cross * 0.1])


class DisturbanceTurbulence:
    name = 'turbulence'
    label = 'Turbulence (σ=0.2 m/s)'

    def __init__(self, sigma=0.2, tau=0.3, seed=42):
        self.sigma = sigma
        self.tau = tau
        self.rng = np.random.RandomState(seed)
        n = 3000
        dt = 0.01
        self.turb = np.zeros((n, 3))
        state = np.zeros(3)
        for i in range(1, n):
            noise = self.rng.randn(3) * sigma * np.sqrt(2 * dt / tau)
            state = state * (1 - dt / tau) + noise
            self.turb[i] = state
        self.dt = dt

    def __call__(self, t, x):
        idx = int(t / self.dt)
        idx = min(idx, len(self.turb) - 1)
        tx, ty, tz = self.turb[idx]
        psi = x[3]
        v_wind = tx * np.cos(psi) + ty * np.sin(psi)
        v_cross = -tx * np.sin(psi) + ty * np.cos(psi)
        return np.array([tx, ty, tz, -v_wind * 0.3, v_cross * 0.1])


DISTURBANCE_MODELS = {
    'none': DisturbanceNone,
    'wind': DisturbanceWind,
    'gust': DisturbanceGust,
    'turbulence': DisturbanceTurbulence,
}


# ═══════════════════════════════════════════════════════════════════════
#  SOLVER FACTORY
# ═══════════════════════════════════════════════════════════════════════

VALID_SOLVERS = ('ipopt', 'knitro')


def build_mpc(solver_choice, xfly, traj, horizon, dt, logger=None,
              knitro_alg=1, ablation=None):
    def log(msg):
        if logger:
            logger(msg)
        else:
            print(msg)

    solver_choice = solver_choice.lower()
    if solver_choice not in VALID_SOLVERS:
        raise ValueError(
            f"Unknown solver '{solver_choice}'. "
            f"Choose from: {VALID_SOLVERS}")

    if solver_choice == 'knitro':
        if not KNITRO_AVAILABLE:
            raise RuntimeError(
                "Knitro requested but not available. "
                "Ensure Knitro is installed and licensed.")
        alg_name = MPCController.KNITRO_ALG_NAMES.get(knitro_alg,
                                                       f'Unknown({knitro_alg})')
        log(f"Building MPC with Knitro (alg={knitro_alg}: {alg_name})...")
        mpc = MPCController(
            xfly, traj, horizon=horizon, dt=dt,
            solver_name='knitro', knitro_alg=knitro_alg,
            ablation=ablation)
        log(f"MPC ready (Knitro/{alg_name}).")
        return mpc, f'knitro-{alg_name}'

    else:  # ipopt
        log("Building MPC with IPOPT...")
        mpc = MPCController(
            xfly, traj, horizon=horizon, dt=dt, solver_name='ipopt',
            ablation=ablation)
        log("MPC ready (IPOPT).")
        return mpc, 'ipopt'


# ═══════════════════════════════════════════════════════════════════════
#  ROS 2 NODE
# ═══════════════════════════════════════════════════════════════════════

if ROS_AVAILABLE:

    class MPCCXFlyNode(Node):
        def __init__(self, ablation=None):
            super().__init__("mpcc_xfly")
            self._ablation = ablation or AblationFlags()

            self.declare_parameter("mode", "real")
            self.declare_parameter("solver", "knitro")
            self.declare_parameter("optitrack_topic",
                "/optitrack_multiplexer_node/rigid_body/XFly2")
            # One timestamped message carries both channels, so the
            # bridge can never pair a fresh flapping command with a
            # stale rudder one. Same interface xfly_teleop uses.
            self.declare_parameter("cmd_topic",
                "/xfly_bridge/cmd")
            self.declare_parameter("enable_topic",
                "/xfly_bridge/enable")
            self.declare_parameter("control_rate", 100.0)
            self.declare_parameter("kf_sigma_accel", 2.0)
            self.declare_parameter("kf_sigma_pos", 0.05)
            self.declare_parameter("yr_sigma_accel", 0.1)
            self.declare_parameter("yr_sigma_meas", 0.001)
            self.declare_parameter("yr_min_speed", 0.3)
            self.declare_parameter("tracking_timeout", 0.5)
            self.declare_parameter("mpc_horizon", 15)
            self.declare_parameter("mpc_dt", 0.1)
            self.declare_parameter("auto_arm", True)
            self.declare_parameter("startup_hold_time", 0.5)
            self.declare_parameter("trajectory", "straight")
            self.declare_parameter("sysid", "")
            self.declare_parameter("adapt_flap", False)
            self._mode = self.get_parameter("mode").value
            solver_choice = self.get_parameter("solver").value
            optitrack_topic = self.get_parameter("optitrack_topic").value
            cmd_topic = self.get_parameter("cmd_topic").value
            enable_topic = self.get_parameter("enable_topic").value
            self._ctrl_rate = self.get_parameter("control_rate").value
            kf_sa = self.get_parameter("kf_sigma_accel").value
            kf_sp = self.get_parameter("kf_sigma_pos").value
            yr_sa = self.get_parameter("yr_sigma_accel").value
            yr_sm = self.get_parameter("yr_sigma_meas").value
            yr_min_spd = self.get_parameter("yr_min_speed").value
            self._tracking_timeout = self.get_parameter(
                "tracking_timeout").value
            mpc_N = self.get_parameter("mpc_horizon").value
            mpc_dt = self.get_parameter("mpc_dt").value
            auto_arm = self.get_parameter("auto_arm").value
            self._startup_hold = self.get_parameter(
                "startup_hold_time").value
            traj_type = self.get_parameter("trajectory").value
            sysid_test = self.get_parameter("sysid").value.strip()
            self._adapt_flap = self.get_parameter("adapt_flap").value

            self._xfly = XFlyParams()
            self._sysid = sysid_test
            self._sysid_delta_flap = 0.0
            self._sysid_delta_rud = 0.0
            self._command_scale = 1.0

            if traj_type in ('helix', 'helix_reverse'):
                direction = -1 if traj_type == 'helix_reverse' else 1
                y_sign = direction
                self._traj = CompositeTrajectory(
                    start=(-3.7, y_sign * 1.5, 0.11),
                    join=(0.0, y_sign * 1.5, 0.3),
                    center=(0.0, 0.0),
                    z_top=0.7,
                    radius=1.5,
                    n_climb_rev=1.0,
                    n_level_rev=3.0,
                    direction=direction)
            else:
                self._traj = StraightLineTrajectory(
                    start=(-3.6, 2.7, 0.1),
                    mid=(0.0, 0.0, 0.5),
                    end=(3.6, -2.7, 0.9))

            if traj_type == 'external':
                if not EXTERNAL_TRAJ_AVAILABLE:
                    self.get_logger().error(
                        "external_trajectory.py not found!")
                    raise RuntimeError("ExternalTrajectory not available")
                csv_path = self.declare_parameter(
                    'trajectory_csv', 'track_1.csv'
                ).get_parameter_value().string_value
                n_loops = self.declare_parameter(
                    'n_loops', 3
                ).get_parameter_value().integer_value
                self._traj = ExternalTrajectory(
                    csv_path,
                    start_pos=(3, 2, 0.35),
                    n_loops=n_loops,
                )

            self._kf = KalmanVelocityEstimator(
                sigma_accel=kf_sa, sigma_pos=kf_sp)
            self._yr = YawRateEstimator(
                sigma_yaw_accel=yr_sa, sigma_yaw_meas=yr_sm,
                min_speed=yr_min_spd,
                delay_comp_s=self._xfly.yaw_delay_comp)

            if sysid_test:
                traj_type = 'helix'
                self._traj = CompositeTrajectory(
                    start=(-3.7, 1.5, 0.11),
                    join=(0.0, 1.5, 0.3),
                    center=(0.0, 0.0),
                    z_top=0.7,
                    radius=1.5,
                    n_climb_rev=1.0,
                    n_level_rev=2.0,
                    direction=1)
                self.get_logger().info(
                    f"╔══════════════════════════════════════╗")
                self.get_logger().info(
                    f"║  SYSID MODE: {sysid_test:24s}║")
                self.get_logger().info(
                    f"║  MPC + perturbation overlay          ║")
                self.get_logger().info(
                    f"║  Trajectory: helix (forced)          ║")
                self.get_logger().info(
                    f"╚══════════════════════════════════════╝")
                self._sysid_schedule = self._build_sysid_schedule(
                    sysid_test)

            self.get_logger().info(
                f"Building MPC (N={mpc_N}, dt={mpc_dt}s, "
                f"solver={solver_choice}, traj={traj_type}) — "
                f"9-state, τ_ψ̈={self._xfly.tau_psi_ddot*1000:.0f}ms, "
                f"delay_comp={self._xfly.yaw_delay_comp*1000:.0f}ms ...")
            try:
                self._mpc, self._solver_backend = build_mpc(
                    solver_choice, self._xfly, self._traj,
                    mpc_N, mpc_dt,
                    logger=lambda msg: self.get_logger().info(msg),
                    ablation=self._ablation)
            except RuntimeError as e:
                self.get_logger().warn(
                    f"{solver_choice} failed: {e} — "
                    f"falling back to IPOPT")
                self._mpc, self._solver_backend = build_mpc(
                    'ipopt', self._xfly, self._traj,
                    mpc_N, mpc_dt,
                    logger=lambda msg: self.get_logger().info(msg),
                    ablation=self._ablation)
            if sysid_test:
                self._solver_backend = f'sysid_{sysid_test}'

            if self._adapt_flap:
                p = self._xfly
                self.get_logger().info(
                    f"Adaptive flap scaling ENABLED: "
                    f"gain={p.command_scale_gain}, "
                    f"range=[{p.command_scale_min}, {p.command_scale_max}]")

            self._lock = threading.Lock()
            self._theta = 0.0
            self._u_prev = np.array([self._xfly.u_level, 0.0])
            self._warm = None
            self._armed = False
            self._last_tracking_time = 0.0
            self._last_vz = 0.0
            self._last_vz_t = 0.0
            self._az_est = 0.0
            # ψ̈ estimator state (finite diff of ψ̇, low-pass filtered)
            self._psi_ddot_est = 0.0
            self._last_psi_dot = 0.0
            self._last_psi_dot_t = 0.0
            self._quat_yaw = 0.0
            self._quat_yaw_valid = False
            self._trajectory_complete = False
            self._solve_count = 0
            self._tracking_stable_since = None
            self._control_active = False
            self._opti_count = 0
            self._vel_yaw_raw = 0.0

            # Async MPC
            self._mpc_solving = False
            self._mpc_result = None
            self._mpc_result_meta = None
            self._last_u_flap = self._xfly.u_level
            self._last_u_rud = 0.0

            # Flight log
            self._flight_log = []
            self._state_log = []
            self._mpc_dt = mpc_dt
            self._mpc_N = mpc_N

            self._pub_cmd = self.create_publisher(
                Vector3Stamped, cmd_topic, 10)
            self._pub_enable = self.create_publisher(
                Bool, enable_topic, 10)
            self._pub_state = self.create_publisher(
                Odometry, "~/state_estimate", 10)
            self._pub_theta = self.create_publisher(
                Float32, "~/path_progress", 10)
            self._pub_path = self.create_publisher(
                Path, "~/predicted_path", 10)

            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(
                RigidBodyStamped, optitrack_topic,
                self._cb_optitrack, qos)
            self.create_subscription(
                Bool, "~/arm", self._cb_arm, 10)
            self.create_subscription(
                UInt8, "/xfly_bridge/battery_level",
                self._cb_battery, 10)

            self._ctrl_dt = 1.0 / self._ctrl_rate
            self._ctrl_timer = self.create_timer(
                self._ctrl_dt, self._control_loop)

            if auto_arm:
                self._armed = True
                self.get_logger().warn(
                    "Auto-armed! Drone WILL fly on tracking.")

            self.get_logger().info(
                f"MPCC node started — {self._solver_backend.upper()}, "
                f"9-state (ψ̈ as state, τ_ψ̈={self._xfly.tau_psi_ddot*1000:.0f}ms), "
                f"delay_comp={self._xfly.yaw_delay_comp*1000:.0f}ms, "
                f"k_yaw={self._xfly.k_yaw:.3f}, "
                f"ablation={self._ablation.tag()}, "
                f"listening on: {optitrack_topic}")

        def _build_sysid_schedule(self, test_name):
            """Build perturbation schedule for system identification."""
            if test_name == 'flap_id':
                return [
                    (0.0,   0.0,    0.0),
                    (4.0,  +0.08,   0.0),
                    (4.7,  -0.08,   0.0),
                    (5.4,   0.0,    0.0),
                    (6.5,  +0.15,   0.0),
                    (7.2,  -0.15,   0.0),
                    (7.9,   0.0,    0.0),
                    (9.0,  +0.08,   0.0),
                    (9.7,  -0.08,   0.0),
                    (10.4,  0.0,    0.0),
                    (11.5, +0.20,   0.0),
                    (12.2, -0.20,   0.0),
                    (12.9,  0.0,    0.0),
                ]
            elif test_name == 'rudder_id':
                return [
                    (0.0,   0.0,    0.0),
                    (4.0,   0.0,   +0.3),
                    (4.7,   0.0,   -0.3),
                    (5.4,   0.0,    0.0),
                    (6.5,   0.0,   +0.6),
                    (7.2,   0.0,   -0.6),
                    (7.9,   0.0,    0.0),
                    (9.0,   0.0,   +0.3),
                    (9.7,   0.0,   -0.3),
                    (10.4,  0.0,    0.0),
                    (11.5,  0.0,   +0.8),
                    (12.2,  0.0,   -0.8),
                    (12.9,  0.0,    0.0),
                ]
            elif test_name == 'full_id':
                return [
                    (0.0,   0.0,    0.0),
                    (4.0,  +0.08,   0.0),
                    (4.7,  -0.08,   0.0),
                    (5.4,   0.0,    0.0),
                    (6.0,  +0.15,   0.0),
                    (6.7,  -0.15,   0.0),
                    (7.4,   0.0,    0.0),
                    (8.0,  +0.20,   0.0),
                    (8.7,  -0.20,   0.0),
                    (9.4,   0.0,    0.0),
                    (10.0,  0.0,   +0.3),
                    (10.7,  0.0,   -0.3),
                    (11.4,  0.0,    0.0),
                    (12.0,  0.0,   +0.6),
                    (12.7,  0.0,   -0.6),
                    (13.4,  0.0,    0.0),
                    (14.0,  0.0,   +0.8),
                    (14.7,  0.0,   -0.8),
                    (15.4,  0.0,    0.0),
                ]
            else:
                self.get_logger().error(
                    f"Unknown sysid test: '{test_name}'. "
                    f"Available: flap_id, rudder_id, full_id")
                return [(0.0, 0.0, 0.0), (15.0, 0.0, 0.0)]

        def _cb_optitrack(self, msg: 'RigidBodyStamped'):
            rb = msg.rigid_body
            t_sec = msg.stamp.sec + msg.stamp.nanosec * 1e-9
            t_wall = time.time()

            if not rb.tracking_valid:
                if not hasattr(self, '_invalid_streak'):
                    self._invalid_streak = 0
                self._invalid_streak += 1
                if self._invalid_streak >= 10:
                    with self._lock:
                        self._tracking_stable_since = None
                if not hasattr(self, '_invalid_count'):
                    self._invalid_count = 0
                self._invalid_count += 1
                if self._invalid_count % 240 == 1:
                    self.get_logger().warn(
                        f"OptiTrack tracking_valid=False "
                        f"({self._invalid_count} msgs)",
                        throttle_duration_sec=2.0)
                return

            if hasattr(self, '_invalid_streak'):
                self._invalid_streak = 0

            pos = np.array([
                rb.pose.position.x,
                rb.pose.position.y,
                rb.pose.position.z])

            q = rb.pose.orientation
            yaw = quat_to_yaw(q.q_x, q.q_y, q.q_z, q.q_w)

            with self._lock:
                first_track = not self._kf.initialized
                self._raw_pos_z = pos[2]
                self._kf.predict_update(pos, t_sec)
                self._last_tracking_time = t_wall
                if self._tracking_stable_since is None:
                    self._tracking_stable_since = t_wall

                self._yr.update(self._kf.vel[0], self._kf.vel[1],
                                t_sec, quat_yaw=yaw)

                vx, vy = self._kf.vel[0], self._kf.vel[1]
                self._vel_yaw_raw = np.arctan2(vy, vx)

                self._quat_yaw = yaw
                self._quat_yaw_valid = True

                # 240 Hz state log
                if self._control_active:
                    spd = self._kf.speed
                    vx_l, vy_l, vz_l = self._kf.vel
                    self._state_log.append({
                        't': t_sec,
                        'pos_x': self._kf.pos[0],
                        'pos_y': self._kf.pos[1],
                        'pos_z': self._kf.pos[2],
                        'vel_x': vx_l, 'vel_y': vy_l, 'vel_z': vz_l,
                        'speed': spd,
                        'yaw': self._yr.yaw if self._yr.initialized else yaw,
                        'yaw_raw_filtered': (self._yr.yaw_raw
                                             if self._yr.initialized else yaw),
                        'yaw_rate': (self._yr.yaw_rate
                                     if self._yr.initialized else 0.0),
                        'yaw_quat_raw': yaw,
                        'yaw_vel_raw': self._vel_yaw_raw,
                        'flap_cmd': self._last_u_flap,
                        'rud_cmd': self._last_u_rud,
                    })

            if first_track:
                self.get_logger().info(
                    f"First tracking: pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}), "
                    f"ψ_quat={np.degrees(yaw):.1f}°")
            self._opti_count += 1

        def _cb_arm(self, msg: Bool):
            with self._lock:
                self._armed = msg.data
            self.get_logger().info(
                f"Flight {'ARMED' if msg.data else 'DISARMED'}")
            if not msg.data:
                self._control_active = False
                self._trajectory_complete = False
                self._mpc_result = None
                self._mpc_result_meta = None
                self._last_u_flap = self._xfly.u_level
                self._last_u_rud = 0.0
                with self._lock:
                    self._theta = 0.0
                    self._warm = None
                    self._solve_count = 0
                    self._psi_ddot_est = 0.0
            enable_msg = Bool()
            enable_msg.data = msg.data
            self._pub_enable.publish(enable_msg)

        def _cb_battery(self, msg: 'UInt8'):
            """Update u_level from battery percentage using linear fit."""
            batt = float(msg.data)  # 0–100 %
            # Linear fit: flap = -0.00549 * batt + 1.021
            u_level_new = -0.00549197 * batt + 1.02054319
            u_level_new = max(self._xfly.u_flap_min,
                              min(self._xfly.u_flap_max, u_level_new))
            old = self._xfly.u_level
            self._xfly.u_level = u_level_new
            if abs(old - u_level_new) > 0.005:
                self.get_logger().info(
                    f"Battery {int(batt)}% → u_level "
                    f"{old:.3f} → {u_level_new:.3f}")

        def _control_loop(self):
            """Non-blocking control loop — runs at 100 Hz."""
            with self._lock:
                armed = self._armed
                kf_init = self._kf.initialized
                last_track = self._last_tracking_time

            now = time.time()

            if kf_init and (now - last_track) > self._tracking_timeout:
                if armed:
                    self.get_logger().warn(
                        "Tracking lost! Zero flapping.",
                        throttle_duration_sec=2.0)
                    self._publish_commands(0.0, 0.0)
                    self._last_u_flap = 0.0
                    self._last_u_rud = 0.0
                    if self._control_active:
                        self._control_active = False
                        self.get_logger().warn("Control deactivated.")
                return

            if not kf_init or not armed:
                if not hasattr(self, '_wait_count'):
                    self._wait_count = 0
                self._wait_count += 1
                if self._wait_count % int(self._ctrl_rate * 2) == 1:
                    self.get_logger().info(
                        f"Waiting: kf_init={kf_init}, armed={armed}, "
                        f"quat_valid={self._quat_yaw_valid}, "
                        f"opti_msgs={self._opti_count}",
                        throttle_duration_sec=2.0)
                return

            # Startup hold
            if not self._control_active:
                with self._lock:
                    stable_since = self._tracking_stable_since
                if stable_since is None:
                    return
                elapsed = now - stable_since
                if elapsed < self._startup_hold:
                    self.get_logger().info(
                        f"Startup hold: {self._startup_hold-elapsed:.1f}s",
                        throttle_duration_sec=0.5)
                    return
                with self._lock:
                    pos = self._kf.pos
                theta_init, _ = self._traj.project_to_path(
                    pos, 0.0,
                    search_radius=self._traj.total_arc,
                    n_samples=200)
                with self._lock:
                    self._theta = theta_init
                self._control_active = True
                self.get_logger().info(
                    f"Control active! θ₀={theta_init:.2f}, "
                    f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}), "
                    f"ψ_quat={np.degrees(self._quat_yaw):.1f}°")

            if self._trajectory_complete:
                self._publish_commands(0.0, 0.0)
                return

            # ── Check if background solve completed ──
            if self._mpc_result is not None:
                sol = self._mpc_result
                meta = self._mpc_result_meta
                self._mpc_result = None
                self._mpc_result_meta = None

                u_cmd = sol['u']
                v_theta = sol['v_theta']

                dt_actual = now - meta['t_solve_start']
                theta_new = min(
                    meta['theta'] + dt_actual * v_theta,
                    self._traj.total_arc)
                if theta_new >= self._traj.total_arc - 0.1:
                    self._trajectory_complete = True
                    self.get_logger().info("Trajectory complete!")

                new_warm = (sol['warm']
                            if sol['solve_time'] < 0.5 else None)

                with self._lock:
                    self._theta = theta_new
                    self._u_prev = u_cmd.copy()
                    self._warm = new_warm

                self._solve_count += 1

                u_flap = float(np.clip(
                    u_cmd[0], self._xfly.u_flap_min,
                    self._xfly.u_flap_max))
                u_rud = float(np.clip(
                    u_cmd[1], self._xfly.u_rud_min,
                    self._xfly.u_rud_max))

                # Adaptive flap scaling
                if self._adapt_flap:
                    p = self._xfly
                    z_ref = self._traj.position_np(theta_new)[2]
                    z_err = z_ref - meta['pos'][2]
                    self._command_scale += (self._ctrl_dt
                                            * p.command_scale_gain
                                            * z_err)
                    self._command_scale = np.clip(
                        self._command_scale,
                        p.command_scale_min, p.command_scale_max)
                    u_flap = float(np.clip(
                        self._command_scale * u_flap,
                        p.u_flap_min, p.u_flap_max))

                # Suppress rudder at low speed
                speed = meta['speed']
                if speed < 1.0:
                    u_rud = u_rud * max(0.0, (speed - 0.5) / 0.5)

                self._last_u_flap = u_flap
                self._last_u_rud = u_rud

                # Diagnostics
                theta_msg = Float32()
                theta_msg.data = float(theta_new)
                self._pub_theta.publish(theta_msg)
                self._publish_predicted_path(sol['X'])
                self._publish_state_estimate(
                    meta['pos'], meta['vel'],
                    meta['yaw_est'])

                # Flight log
                ref_pos = self._traj.position_np(theta_new)
                ref_heading = self._traj.heading(theta_new)
                ref_gamma = self._traj.climb_angle_np(theta_new)
                self._flight_log.append({
                    't': time.time(),
                    # ── Estimated state ──
                    'pos_x': meta['pos'][0],
                    'pos_y': meta['pos'][1],
                    'pos_z': meta['pos'][2],
                    'raw_z': meta['raw_z'],
                    'vel_x': meta['vel'][0],
                    'vel_y': meta['vel'][1],
                    'vel_z': meta['vel'][2],
                    'az_est': meta['az_est'],
                    'speed': speed,
                    'yaw': meta['yaw_est'],
                    'yaw_rate': meta['yaw_rate_est'],
                    'psi_ddot': meta['psi_ddot_est'],   # ψ̈ state estimate
                    # ── Raw measurements ──
                    'yaw_quat_raw': meta['quat_yaw'],
                    'yaw_vel_raw': meta['vel_yaw_raw'],
                    # ── Commands ──
                    'flap_cmd': u_flap,
                    'rud_cmd': u_rud,
                    'flap_raw': float(u_cmd[0]),
                    'rud_raw': float(u_cmd[1]),
                    'delta_flap': self._sysid_delta_flap,
                    'delta_rud': self._sysid_delta_rud,
                    'command_scale': self._command_scale,
                    # ── Path ──
                    'theta': theta_new,
                    'v_theta': v_theta,
                    'ref_x': ref_pos[0],
                    'ref_y': ref_pos[1],
                    'ref_z': ref_pos[2],
                    'ref_heading': ref_heading,
                    'ref_gamma': ref_gamma,
                    # ── Solver ──
                    'solve_time': sol['solve_time'],
                    # ── Predictions (arrays) ──
                    'pred_x':        sol['X'][0, :].copy(),
                    'pred_y':        sol['X'][1, :].copy(),
                    'pred_z':        sol['X'][2, :].copy(),
                    'pred_yaw':      sol['X'][3, :].copy(),
                    'pred_v':        sol['X'][4, :].copy(),
                    'pred_vz':       sol['X'][5, :].copy(),
                    'pred_az':       sol['X'][6, :].copy(),
                    'pred_yaw_rate': sol['X'][7, :].copy(),
                    'pred_psi_ddot': sol['X'][8, :].copy(),  # ψ̈ predictions
                    'pred_u_flap':   sol['U'][0, :].copy(),
                    'pred_u_rud':    sol['U'][1, :].copy(),
                    # ── backward compat ──
                    'pos': meta['pos'].copy(),
                })

                if self._solve_count % 30 == 0:
                    seg = "CLIMB" if theta_new < self._traj.s1 else "LEVEL"
                    self.get_logger().info(
                        f"[{seg}] θ={theta_new:.1f}/"
                        f"{self._traj.total_arc:.0f} | "
                        f"pos=({meta['pos'][0]:.2f},"
                        f"{meta['pos'][1]:.2f},"
                        f"{meta['pos'][2]:.2f}) | "
                        f"v={speed:.2f} "
                        f"ψ={np.degrees(meta['yaw_est']):+.0f}° "
                        f"ψ̇={np.degrees(meta['yaw_rate_est']):+.0f}°/s "
                        f"ψ̈={np.degrees(meta['psi_ddot_est']):+.0f}°/s² | "
                        f"f={u_flap:.2f} r={u_rud:+.2f}"
                        f"{f' s={self._command_scale:.3f}' if self._adapt_flap else ''}"
                        f" | {sol['solve_time']*1000:.0f}ms")

            # ── Always publish commands ──
            pub_flap = self._last_u_flap
            pub_rud = self._last_u_rud
            self._sysid_delta_flap = 0.0
            self._sysid_delta_rud = 0.0
            if self._sysid:
                if not hasattr(self, '_sysid_t0'):
                    self._sysid_t0 = now
                t_elapsed = now - self._sysid_t0
                schedule = self._sysid_schedule
                d_flap, d_rud = schedule[-1][1], schedule[-1][2]
                for i, (t_start, df, dr) in enumerate(schedule):
                    if i + 1 < len(schedule):
                        if t_elapsed < schedule[i + 1][0]:
                            d_flap, d_rud = df, dr
                            break
                    else:
                        d_flap, d_rud = df, dr
                self._sysid_delta_flap = d_flap
                self._sysid_delta_rud = d_rud
                pub_flap = float(np.clip(
                    pub_flap + d_flap,
                    self._xfly.u_flap_min, self._xfly.u_flap_max))
                pub_rud = float(np.clip(
                    pub_rud + d_rud,
                    self._xfly.u_rud_min, self._xfly.u_rud_max))
            self._publish_commands(pub_flap, pub_rud)

            # ── Kick off background solve ──
            if not self._mpc_solving:
                with self._lock:
                    pos = self._kf.pos
                    vel = self._kf.vel
                    speed = self._kf.speed
                    if self._yr.initialized:
                        yaw_est = self._yr.yaw
                        yaw_rate_est = self._yr.yaw_rate
                    else:
                        yaw_est = self._quat_yaw
                        yaw_rate_est = 0.0
                    theta = self._theta
                    u_prev = self._u_prev.copy()
                    warm = self._warm
                    quat_yaw_snap = self._quat_yaw
                    vel_yaw_raw_snap = self._vel_yaw_raw
                    raw_z = getattr(self, '_raw_pos_z', pos[2])

                speed = max(0.05, min(speed, self._xfly.v_max * 1.1))

                # Estimate vertical acceleration (az) from vz derivative
                t_now = time.perf_counter()
                vz_now = vel[2]
                dt_vz = t_now - self._last_vz_t
                if dt_vz > 0.002 and dt_vz < 0.1 and self._last_vz_t > 0:
                    az_raw = (vz_now - self._last_vz) / dt_vz
                    alpha_az = 0.3
                    self._az_est = alpha_az * az_raw + (1 - alpha_az) * self._az_est
                self._last_vz = vz_now
                self._last_vz_t = t_now

                # Estimate ψ̈ from finite diff of ψ̇, low-pass filtered
                psi_dot_now = yaw_rate_est
                dt_pd = t_now - self._last_psi_dot_t
                if dt_pd > 0.002 and dt_pd < 0.1 and self._last_psi_dot_t > 0:
                    psi_ddot_raw = (psi_dot_now - self._last_psi_dot) / dt_pd
                    alpha_pdd = 0.1
                    self._psi_ddot_est = (alpha_pdd * psi_ddot_raw
                                          + (1 - alpha_pdd) * self._psi_ddot_est)
                self._last_psi_dot = psi_dot_now
                self._last_psi_dot_t = t_now

                # 9-state initial condition: append ψ̈ estimate
                x0 = np.array([
                    pos[0], pos[1], pos[2],
                    yaw_est, speed, vel[2], self._az_est,
                    yaw_rate_est,
                    self._psi_ddot_est,   # ψ̈ (9th state)
                ])

                # Re-project theta
                theta_proj, dist = self._traj.project_to_path(
                    pos, theta, search_radius=3.0)
                if dist > 0.5 or self._solve_count < 3:
                    theta = max(theta, theta_proj)

                if self._solve_count == 0:
                    self.get_logger().info(
                        f"FIRST SOLVE:\n"
                        f"  x0={x0}\n"
                        f"  theta={theta:.3f}\n"
                        f"  u_prev={u_prev}\n"
                        f"  warm={warm is not None}")

                meta = {
                    'pos': pos, 'vel': vel,
                    'speed': speed,
                    'az_est': self._az_est,
                    'psi_ddot_est': self._psi_ddot_est,
                    'yaw_est': yaw_est,
                    'yaw_rate_est': yaw_rate_est,
                    'quat_yaw': quat_yaw_snap,
                    'vel_yaw_raw': vel_yaw_raw_snap,
                    'raw_z': raw_z,
                    'theta': theta,
                    't_solve_start': time.time(),
                }

                self._mpc_solving = True
                thread = threading.Thread(
                    target=self._solve_background,
                    args=(x0, theta, u_prev, warm, meta),
                    daemon=True)
                thread.start()

        def _solve_background(self, x0, theta, u_prev, warm, meta):
            """Run MPC solve in background thread."""
            try:
                sol = self._mpc.solve(x0, theta, u_prev, warm)
                self._mpc_result_meta = meta
                self._mpc_result = sol
            except Exception as e:
                self.get_logger().error(
                    f"MPC FAIL: {e}\n"
                    f"  x0={x0}\n"
                    f"  theta={theta:.3f}",
                    throttle_duration_sec=1.0)
                with self._lock:
                    self._warm = None
            finally:
                self._mpc_solving = False

        def _publish_commands(self, flap, rudder):
            # x = flapping [0, 1], y = rudder [-1, 1]; the bridge
            # clamps both. z is unused.
            msg = Vector3Stamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.vector.x = float(flap)
            msg.vector.y = float(rudder)
            msg.vector.z = 0.0
            self._pub_cmd.publish(msg)

        def _publish_predicted_path(self, X_sol):
            path_msg = Path()
            stamp = self.get_clock().now().to_msg()
            path_msg.header.stamp = stamp
            path_msg.header.frame_id = "world"
            for k in range(X_sol.shape[1]):
                pose = PoseStamped()
                pose.header.stamp = stamp
                pose.header.frame_id = "world"
                pose.pose.position.x = float(X_sol[0, k])
                pose.pose.position.y = float(X_sol[1, k])
                pose.pose.position.z = float(X_sol[2, k])
                yaw = float(X_sol[3, k])
                pose.pose.orientation.z = float(np.sin(yaw / 2))
                pose.pose.orientation.w = float(np.cos(yaw / 2))
                path_msg.poses.append(pose)
            self._pub_path.publish(path_msg)

        def _publish_state_estimate(self, pos, vel, yaw):
            msg = Odometry()
            stamp = self.get_clock().now().to_msg()
            msg.header.stamp = stamp
            msg.header.frame_id = "world"
            msg.child_frame_id = "xfly"
            msg.pose.pose.position.x = float(pos[0])
            msg.pose.pose.position.y = float(pos[1])
            msg.pose.pose.position.z = float(pos[2])
            cy, sy = np.cos(yaw/2), np.sin(yaw/2)
            msg.pose.pose.orientation.z = float(sy)
            msg.pose.pose.orientation.w = float(cy)
            msg.twist.twist.linear.x = float(vel[0])
            msg.twist.twist.linear.y = float(vel[1])
            msg.twist.twist.linear.z = float(vel[2])
            self._pub_state.publish(msg)

        def plot_flight_log(self):
            """Generate post-flight diagnostic figures on ctrl+c.

            Figure 1 (3×2) — Yaw, ψ̈, Controls & Path:
              (0,0) Filtered yaw + predictions + rudder
              (0,1) Yaw sources: filtered vs raw velocity vs raw quaternion
              (1,0) Yaw rate (ψ̇) + MPC predictions
              (1,1) Yaw acceleration (ψ̈) estimated + MPC predictions + rudder
              (2,0) Controls & Speed
              (2,1) XY top-down path colored by speed with yaw arrows

            Figure 2 (2×2) — Altitude, Vertical Dynamics & Performance:
              (0,0) Altitude: ref + actual + predicted + flap overlay
              (0,1) Vertical velocity (vz) + predicted vz
              (1,0) Contour + altitude errors
              (1,1) Solve time + histogram inset
            """
            log = self._flight_log
            if len(log) < 5:
                self.get_logger().warn(
                    f"Only {len(log)} log entries — skipping plot.")
                return

            import matplotlib
            if not os.environ.get('DISPLAY'):
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            from datetime import datetime

            # ── Extract data ──
            t0 = log[0]['t']
            t = np.array([e['t'] - t0 for e in log])

            def wrap_180(deg):
                return (np.asarray(deg) + 180.0) % 360.0 - 180.0

            yaw         = wrap_180(np.degrees(np.array([e['yaw'] for e in log])))
            yaw_vel_raw = wrap_180(np.degrees(np.array([e['yaw_vel_raw'] for e in log])))
            yaw_quat_raw= wrap_180(np.degrees(np.array([e['yaw_quat_raw'] for e in log])))
            yr          = np.degrees(np.array([e['yaw_rate'] for e in log]))
            psi_ddot_arr= np.degrees(np.array([e.get('psi_ddot', 0.0) for e in log]))
            rud         = np.array([e['rud_cmd'] for e in log])
            flap        = np.array([e['flap_cmd'] for e in log])
            spd_arr     = np.array([e['speed'] for e in log])
            pos_arr     = np.array([e['pos'] for e in log])
            raw_z_arr   = np.array([e.get('raw_z', e['pos'][2]) for e in log])
            theta_arr   = np.array([e['theta'] for e in log])
            vz_arr      = np.array([e['vel_z'] for e in log])
            az_arr      = np.array([e.get('az_est', 0.0) for e in log])
            st          = np.array([e['solve_time'] for e in log]) * 1000
            dt          = self._mpc_dt
            N           = self._mpc_N
            traj        = self._traj

            traj_heading = np.degrees(traj.heading(0))
            interval = max(1, int(0.1 / (t[1] - t[0]))) if len(t) > 1 else 1
            cmap = plt.cm.coolwarm
            pred_indices = list(range(0, len(log), interval))

            s_ref = np.linspace(0, traj.total_arc, 200)
            ref_pts = np.array([traj.position_np(s) for s in s_ref])
            ref_z = np.array([traj.position_np(th)[2] for th in theta_arr])

            lat_errors, alt_errors = [], []
            for i in range(len(log)):
                ref  = traj.position_np(theta_arr[i])
                tang = traj.tangent_np(theta_arr[i])
                err_3d = pos_arr[i] - ref
                e_lag  = np.dot(err_3d, tang)
                e_cont = np.sqrt(max(0, np.dot(err_3d, err_3d) - e_lag**2))
                lat_errors.append(e_cont)
                alt_errors.append(pos_arr[i, 2] - ref[2])
            lat_errors = np.array(lat_errors)
            alt_errors = np.array(alt_errors)

            # Loop-only contour stats (exclude approach phase)
            s_approach = getattr(traj, 's_approach', getattr(traj, 's1', 0.0))
            loop_mask = theta_arr >= s_approach
            lat_errors_loop = lat_errors[loop_mask] if np.any(loop_mask) else lat_errors

            log_dir = os.path.expanduser('~/mpcc_flight_logs')
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # ══════════════════════════════════════════════════
            #  FIGURE 1: Yaw, ψ̈, Controls & Path  (3×2)
            # ══════════════════════════════════════════════════
            fig1 = plt.figure(figsize=(22, 21))
            gs1 = GridSpec(3, 2, figure=fig1, hspace=0.30, wspace=0.25)
            fig1.suptitle(
                f'Post-Flight: Yaw, ψ̈, Controls & Path — '
                f'{self._solver_backend.upper()} | '
                f'τ_ψ̈={self._xfly.tau_psi_ddot*1000:.0f}ms | '
                f'{len(log)} solves over {t[-1]:.1f}s',
                fontsize=14, fontweight='bold', y=0.995)

            # ── (0,0) Filtered yaw + predictions + rudder ──
            ax = fig1.add_subplot(gs1[0, 0])
            ax.plot(t, yaw, 'k-', lw=2, label='Filtered ψ (MPC input)', zorder=5)
            ax.axhline(traj_heading, color='g', ls='--', lw=1,
                       label=f'Path heading ({traj_heading:.0f}°)')
            for idx in pred_indices:
                e = log[idx]
                t_pred = t[idx] + np.arange(N + 1) * dt
                yaw_pred = wrap_180(np.degrees(e['pred_yaw']))
                frac = idx / max(len(log) - 1, 1)
                ax.plot(t_pred, yaw_pred, '-', color=cmap(frac), lw=0.8, alpha=0.6)
            ax.plot([], [], '-', color=cmap(0.5), lw=0.8, alpha=0.6,
                    label='MPC predictions')
            rud_scaled = rud * 180.0
            ax.fill_between(t, 0, rud_scaled, alpha=0.12, color='red',
                            label='Rudder (scaled ±180°)')
            ax_rud = ax.twinx()
            for idx in pred_indices:
                e = log[idx]
                pred_ur = e.get('pred_u_rud', None)
                if pred_ur is not None:
                    t_pred_u = t[idx] + np.arange(len(pred_ur)) * dt
                    frac = idx / max(len(log) - 1, 1)
                    ax_rud.plot(t_pred_u, pred_ur, '-', color=cmap(frac),
                                lw=0.8, alpha=0.5)
            ax_rud.plot([], [], '-', color=cmap(0.5), lw=0.8, alpha=0.5,
                        label='MPC predicted rudder')
            ax_rud.set_ylabel('Rudder cmd', color='red')
            ax_rud.tick_params(axis='y', labelcolor='red')
            ax_rud.set_ylim(-1.2, 1.2)
            ax_rud.legend(loc='lower right', fontsize=7)
            ax.set_ylabel('Yaw (°)')
            ax.set_xlabel('Time (s)')
            ax.set_title('Filtered Yaw + MPC Predictions')
            ax.set_ylim(-180, 180)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

            # ── (0,1) Yaw sources comparison ──
            ax = fig1.add_subplot(gs1[0, 1])
            ax.plot(t, yaw_vel_raw, '-', color='dodgerblue', lw=0.8,
                    alpha=0.6, label='Raw velocity ψ (atan2(vy,vx))')
            ax.plot(t, yaw_quat_raw, '-', color='darkorange', lw=0.8,
                    alpha=0.6, label='Raw quaternion ψ (mocap)')
            ax.plot(t, yaw, 'k-', lw=2.0, label='Filtered ψ (KF output)', zorder=5)
            ax.axhline(traj_heading, color='g', ls='--', lw=1,
                       label=f'Path heading ({traj_heading:.0f}°)')
            ax.set_ylabel('Yaw (°)')
            ax.set_xlabel('Time (s)')
            ax.set_title('Yaw Sources: Filtered vs Raw Velocity vs Raw Quaternion')
            ax.set_ylim(-180, 180)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

            # ── (1,0) Yaw rate ψ̇ + predictions ──
            ax = fig1.add_subplot(gs1[1, 0])
            ax.plot(t, yr, 'k-', lw=2, label='Estimated ψ̇', zorder=5)
            for idx in pred_indices:
                e = log[idx]
                t_pred = t[idx] + np.arange(N + 1) * dt
                yr_pred = np.degrees(e['pred_yaw_rate'])
                frac = idx / max(len(log) - 1, 1)
                ax.plot(t_pred, yr_pred, '-', color=cmap(frac), lw=0.8, alpha=0.6)
            ax.plot([], [], '-', color=cmap(0.5), lw=0.8, alpha=0.6,
                    label='MPC predicted ψ̇')
            ax.axhline(0, color='gray', alpha=0.3)
            ax.set_ylabel('Yaw rate ψ̇ (°/s)')
            ax.set_xlabel('Time (s)')
            ax.set_title('Yaw Rate ψ̇ + MPC Predictions')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

            # ── (1,1) Yaw acceleration ψ̈ — the new state ──
            ax = fig1.add_subplot(gs1[1, 1])
            ax.plot(t, psi_ddot_arr, color='darkorchid', lw=2,
                    label='Estimated ψ̈ (finite diff, filtered)', zorder=5)
            for idx in pred_indices:
                e = log[idx]
                pdd_pred = e.get('pred_psi_ddot', None)
                if pdd_pred is not None:
                    t_pred = t[idx] + np.arange(N + 1) * dt
                    pdd_pred_deg = np.degrees(pdd_pred)
                    frac = idx / max(len(log) - 1, 1)
                    ax.plot(t_pred, pdd_pred_deg, '-', color=cmap(frac),
                            lw=0.8, alpha=0.6)
            ax.plot([], [], '-', color=cmap(0.5), lw=0.8, alpha=0.6,
                    label='MPC predicted ψ̈')
            ax.axhline(0, color='gray', alpha=0.3)
            # Overlay rudder on twin axis for correlation
            ax_r = ax.twinx()
            ax_r.plot(t, rud, '-', color='red', lw=1.0, alpha=0.5,
                      label='Rudder cmd')
            ax_r.set_ylabel('Rudder', color='red')
            ax_r.tick_params(axis='y', labelcolor='red')
            ax_r.set_ylim(-1.5, 1.5)
            ax_r.legend(loc='lower right', fontsize=7)
            ax.set_ylabel('Yaw accel ψ̈ (°/s²)', color='darkorchid')
            ax.tick_params(axis='y', labelcolor='darkorchid')
            ax.set_xlabel('Time (s)')
            ax.set_title(
                f'Yaw Acceleration ψ̈ (new state, τ={self._xfly.tau_psi_ddot*1000:.0f}ms lag)')
            lines1, labels1 = ax.get_legend_handles_labels()
            ax.legend(lines1, labels1, loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

            # ── (2,0) Controls + speed ──
            ax = fig1.add_subplot(gs1[2, 0])
            ax.plot(t, flap, 'b-', lw=1.5, label='Flap')
            ax.axhline(self._xfly.u_level, color='b', ls=':', lw=1,
                       alpha=0.4, label=f'u_level={self._xfly.u_level:.2f}')
            ax.plot(t, rud, 'r-', lw=1.5, label='Rudder')
            ax.axhline(0, color='gray', alpha=0.2)
            ax2 = ax.twinx()
            ax2.plot(t, spd_arr, '-', color='green', lw=2, alpha=0.8,
                     label='Speed', zorder=5)
            for idx in pred_indices:
                e = log[idx]
                t_pred = t[idx] + np.arange(N + 1) * dt
                v_pred = e.get('pred_v', np.full(N+1, e['speed']))
                frac = idx / max(len(log) - 1, 1)
                ax2.plot(t_pred, v_pred, '-', color=cmap(frac), lw=0.8, alpha=0.5)
            ax2.plot([], [], '-', color=cmap(0.5), lw=0.8, alpha=0.5,
                     label='MPC predicted speed')
            ax2.set_ylabel('Speed (m/s)', color='green')
            ax.fill_between(t, self._xfly.u_level, flap,
                            alpha=0.08, color='blue',
                            label='Flap (deviation from level)')
            ax.set_ylabel('Command')
            ax.set_xlabel('Time (s)')
            ax.set_title('Controls & Speed')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

            # ── (2,1) XY top-down view with yaw arrows ──
            ax = fig1.add_subplot(gs1[2, 1])
            ax.plot(ref_pts[:, 0], ref_pts[:, 1], 'g--', lw=2,
                    label='Reference path')
            # Color actual path by speed
            for i in range(len(pos_arr) - 1):
                frac = np.clip(spd_arr[i] / self._xfly.v_max, 0, 1)
                ax.plot(pos_arr[i:i+2, 0], pos_arr[i:i+2, 1], '-',
                        color=plt.cm.plasma(frac), lw=2, alpha=0.8)
            ax.plot(pos_arr[0, 0], pos_arr[0, 1], 'go', ms=10, label='Start')
            ax.plot(pos_arr[-1, 0], pos_arr[-1, 1], 'rs', ms=10, label='End')
            for idx in pred_indices:
                e = log[idx]
                x_pred = e.get('pred_x', None)
                y_pred = e.get('pred_y', None)
                if x_pred is not None and y_pred is not None:
                    frac = idx / max(len(log) - 1, 1)
                    ax.plot(x_pred, y_pred, '-', color=cmap(frac),
                            lw=0.8, alpha=0.5)
            arrow_int = max(1, int(0.5 / (t[1] - t[0]))) if len(t) > 1 else 1
            yaw_rad = np.array([e['yaw'] for e in log])
            for idx in range(0, len(log), arrow_int):
                dx = 0.15 * np.cos(yaw_rad[idx])
                dy = 0.15 * np.sin(yaw_rad[idx])
                ax.annotate('', xy=(pos_arr[idx, 0]+dx, pos_arr[idx, 1]+dy),
                            xytext=(pos_arr[idx, 0], pos_arr[idx, 1]),
                            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
            sm = plt.cm.ScalarMappable(
                cmap='plasma', norm=plt.Normalize(0, self._xfly.v_max))
            sm.set_array([])
            cbar = fig1.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Speed (m/s)')
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
            ax.set_title('Top View (XY) — colored by speed')
            ax.plot([], [], '-', color='steelblue', lw=0.8, alpha=0.5,
                    label='MPC predictions')
            ax.legend(loc='best', fontsize=8)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            fig1.tight_layout()
            path1 = os.path.join(log_dir, f'flight_{timestamp}_yaw_path.png')
            fig1.savefig(path1, dpi=150, bbox_inches='tight')
            self.get_logger().info(f"Saved {path1}")

            # ══════════════════════════════════════════════════
            #  FIGURE 2: Altitude, Vertical Dynamics & Performance (2×2)
            # ══════════════════════════════════════════════════
            fig2 = plt.figure(figsize=(22, 14))
            gs2 = GridSpec(2, 2, figure=fig2, hspace=0.30, wspace=0.25)
            fig2.suptitle(
                f'Post-Flight: Altitude, Vertical Dynamics & Performance — '
                f'{self._solver_backend.upper()} | '
                f'loop contour err: {np.mean(lat_errors_loop):.3f}m (mean) / '
                f'{np.max(lat_errors_loop):.3f}m (max)',
                fontsize=14, fontweight='bold', y=0.995)

            # ── (0,0) Altitude ──
            ax = fig2.add_subplot(gs2[0, 0])
            ax.plot(t, ref_z, 'g--', lw=2, label='Reference Z')
            ax.plot(t, raw_z_arr, '-', color='darkorange', lw=0.8,
                    alpha=0.5, label='Raw Z (mocap)')
            ax.plot(t, pos_arr[:, 2], 'k-', lw=2, label='Actual Z', zorder=5)
            for idx in pred_indices:
                e = log[idx]
                t_pred = t[idx] + np.arange(N + 1) * dt
                z_pred = e.get('pred_z', np.full(N+1, e['pos'][2]))
                frac = idx / max(len(log) - 1, 1)
                ax.plot(t_pred, z_pred, '-', color=cmap(frac), lw=0.8, alpha=0.6)
            ax.plot([], [], '-', color=cmap(0.5), lw=0.8, alpha=0.6,
                    label='MPC predicted Z')
            ax.set_ylabel('Z (m)')
            ax.set_xlabel('Time (s)')
            ax2 = ax.twinx()
            ax2.plot(t, flap, '-', color='blue', lw=0.8, alpha=0.4)
            ax2.fill_between(t, self._xfly.u_level, flap,
                             alpha=0.08, color='blue',
                             label='Flap (deviation from level)')
            ax2.set_ylabel('Flap cmd', color='blue', alpha=0.6)
            ax2.tick_params(axis='y', colors='blue', labelsize=8)
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
            ax.set_title('Altitude: Reference + Actual + Predicted')
            ax.grid(True, alpha=0.3)

            # ── (0,1) Vertical velocity (vz) + predicted vz + az ──
            ax = fig2.add_subplot(gs2[0, 1])
            ax.plot(t, vz_arr, 'k-', lw=2, label='vz (estimated)', zorder=5)
            for idx in pred_indices:
                e = log[idx]
                t_pred = t[idx] + np.arange(N + 1) * dt
                vz_pred = e.get('pred_vz', np.zeros(N+1))
                frac = idx / max(len(log) - 1, 1)
                ax.plot(t_pred, vz_pred, '-', color=cmap(frac), lw=0.8, alpha=0.6)
            ax.plot([], [], '-', color=cmap(0.5), lw=0.8, alpha=0.6,
                    label='MPC predicted vz')
            ax.axhline(0, color='gray', alpha=0.3)
            ax.set_ylabel('vz (m/s)')
            ax.set_xlabel('Time (s)')
            ax2 = ax.twinx()
            ax2.plot(t, az_arr, '-', color='purple', lw=0.8, alpha=0.5,
                     label='az (estimated)')
            for idx in pred_indices:
                e = log[idx]
                t_pred = t[idx] + np.arange(N + 1) * dt
                az_pred = e.get('pred_az', np.zeros(N+1))
                frac = idx / max(len(log) - 1, 1)
                ax2.plot(t_pred, az_pred, '-', color='orchid', lw=0.6, alpha=0.4)
            ax2.plot([], [], '-', color='orchid', lw=0.6, alpha=0.4,
                     label='MPC predicted az')
            ax2.set_ylabel('az (m/s²)', color='purple', alpha=0.6)
            ax2.tick_params(axis='y', colors='purple', labelsize=8)
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
            ax.set_title('Vertical Dynamics: vz + az (Actual + Predicted)')
            ax.grid(True, alpha=0.3)

            # ── (1,0) Contour + altitude errors ──
            ax = fig2.add_subplot(gs2[1, 0])
            ax.plot(t, lat_errors, 'r-', lw=1.5, label='Contour error')
            ax.plot(t, np.abs(alt_errors), 'b-', lw=1, alpha=0.7,
                    label='|Altitude error|')
            ax.axhline(np.mean(lat_errors_loop), color='r', ls='--', lw=1, alpha=0.5,
                       label=f'Mean loop contour: {np.mean(lat_errors_loop):.3f}m')
            ax.set_ylabel('Error (m)')
            ax.set_xlabel('Time (s)')
            ax.set_title('Tracking Errors')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

            # ── (1,1) Solve time + histogram inset ──
            ax = fig2.add_subplot(gs2[1, 1])
            ax.plot(t, st, 'b-', lw=0.8, alpha=0.7, label='Solve time')
            ax.axhline(np.mean(st), color='red', ls='--', lw=1.5,
                       label=f'Mean: {np.mean(st):.1f} ms')
            ax.axhline(np.percentile(st, 95), color='orange', ls=':',
                       lw=1.5, label=f'P95: {np.percentile(st, 95):.1f} ms')
            ctrl_dt_ms = self._ctrl_dt * 1000
            ax.axhline(ctrl_dt_ms, color='red', ls='-', lw=2, alpha=0.3,
                       label=f'Control dt: {ctrl_dt_ms:.0f} ms')
            ax.set_ylabel('Solve time (ms)')
            ax.set_xlabel('Time (s)')
            ax.set_title('MPC Solve Time')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax_ins = ax.inset_axes([0.62, 0.55, 0.35, 0.40])
            ax_ins.hist(st, bins=25, color='steelblue',
                        edgecolor='white', alpha=0.8)
            ax_ins.axvline(np.mean(st), color='red', ls='--', lw=1)
            ax_ins.set_xlabel('ms', fontsize=7)
            ax_ins.set_ylabel('Count', fontsize=7)
            ax_ins.tick_params(labelsize=6)

            fig2.tight_layout()
            path2 = os.path.join(log_dir, f'flight_{timestamp}_alt_vz.png')
            fig2.savefig(path2, dpi=150, bbox_inches='tight')
            self.get_logger().info(f"Saved {path2}")

            # ══════════════════════════════════════════════════
            #  FIGURE 3: 3D Trajectory (Ref, Actual, MPC Predictions)
            # ══════════════════════════════════════════════════
            # Guarded: a 3D-plotting failure must NOT abort the routine
            # before the CSV/state/params exports below run.
            path3 = "(3d plot skipped)"
            try:
                # Importing Axes3D registers the '3d' projection. This can
                # raise ImportError if a stale system mpl_toolkits shadows the
                # one shipped with the installed matplotlib — guarded here so
                # it only skips the 3D plot, never the CSV exports below.
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                fig3 = plt.figure(figsize=(18, 14))
                ax3d = fig3.add_subplot(111, projection='3d')
                fig3.suptitle(
                    f'3D Trajectory — '
                    f'{self._solver_backend.upper()} | '
                    f'loop contour: {np.mean(lat_errors_loop):.3f}m mean | '
                    f'{len(log)} solves over {t[-1]:.1f}s',
                    fontsize=14, fontweight='bold')

                # Reference path (dense)
                ax3d.plot(ref_pts[:, 0], ref_pts[:, 1], ref_pts[:, 2],
                          'g--', lw=2, alpha=0.6, label='Reference path')

                # Actual path colored by speed
                for i in range(len(pos_arr) - 1):
                    frac = np.clip(spd_arr[i] / self._xfly.v_max, 0, 1)
                    ax3d.plot(pos_arr[i:i+2, 0], pos_arr[i:i+2, 1],
                              pos_arr[i:i+2, 2],
                              '-', color=plt.cm.plasma(frac), lw=2.5, alpha=0.9)

                # Start / End markers
                ax3d.scatter(*pos_arr[0, :3], c='lime', s=120, marker='o',
                             edgecolors='k', zorder=10, label='Start')
                ax3d.scatter(*pos_arr[-1, :3], c='red', s=120, marker='s',
                             edgecolors='k', zorder=10, label='End')

                # MPC predicted horizons at regular intervals
                pred_3d_interval = max(1, int(2.0 / (t[1] - t[0]))) if len(t) > 1 else 1
                pred_3d_indices = list(range(0, len(log), pred_3d_interval))
                for idx in pred_3d_indices:
                    e = log[idx]
                    x_p = e.get('pred_x', None)
                    y_p = e.get('pred_y', None)
                    z_p = e.get('pred_z', None)
                    if x_p is not None and y_p is not None and z_p is not None:
                        frac = idx / max(len(log) - 1, 1)
                        ax3d.plot(x_p, y_p, z_p, '-', color=cmap(frac),
                                  lw=1.0, alpha=0.4)
                        # Small dot at prediction start
                        ax3d.scatter(x_p[0], y_p[0], z_p[0],
                                     c=[cmap(frac)], s=8, alpha=0.5)
                ax3d.plot([], [], [], '-', color=cmap(0.5), lw=1.0, alpha=0.5,
                          label=f'MPC horizon (every {pred_3d_interval * (t[1]-t[0]) if len(t)>1 else 0:.1f}s)')

                # Gate markers (if external trajectory with gates)
                if hasattr(traj, '_all_s') and hasattr(traj, 's_approach'):
                    # Mark loop start (join point)
                    join = traj.position_np(traj.s_approach)
                    ax3d.scatter(*join, c='cyan', s=100, marker='D',
                                 edgecolors='k', zorder=10, label='Loop join')

                # Ground plane (faint)
                xlim = ax3d.get_xlim(); ylim = ax3d.get_ylim()
                xx, yy = np.meshgrid(
                    np.linspace(xlim[0], xlim[1], 3),
                    np.linspace(ylim[0], ylim[1], 3))
                ax3d.plot_surface(xx, yy, np.zeros_like(xx),
                                  alpha=0.05, color='gray')

                # Formatting
                ax3d.set_xlabel('X (m)', fontsize=11)
                ax3d.set_ylabel('Y (m)', fontsize=11)
                ax3d.set_zlabel('Z (m)', fontsize=11)
                ax3d.legend(loc='upper left', fontsize=9)

                # Color bar for speed
                sm3 = plt.cm.ScalarMappable(
                    cmap='plasma', norm=plt.Normalize(0, self._xfly.v_max))
                sm3.set_array([])
                cbar3 = fig3.colorbar(sm3, ax=ax3d, shrink=0.5, pad=0.08)
                cbar3.set_label('Speed (m/s)')

                # Nice default view angle
                ax3d.view_init(elev=25, azim=-60)

                fig3.tight_layout()
                path3 = os.path.join(log_dir, f'flight_{timestamp}_3d.png')
                fig3.savefig(path3, dpi=150, bbox_inches='tight')
                self.get_logger().info(f"Saved {path3}")
            except Exception as ex:
                path3 = "(3d plot failed)"
                self.get_logger().error(f"3D plot failed (continuing): {ex}")

            # ══════════════════════════════════════════════════
            #  CSV EXPORT
            # ══════════════════════════════════════════════════
            csv_path = os.path.join(log_dir, f'flight_{timestamp}_log.csv')
            try:
                import csv
                scalar_cols = [
                    't', 'pos_x', 'pos_y', 'pos_z', 'raw_z',
                    'vel_x', 'vel_y', 'vel_z', 'az_est', 'speed',
                    'yaw', 'yaw_rate', 'psi_ddot',
                    'yaw_quat_raw', 'yaw_vel_raw',
                    'flap_cmd', 'rud_cmd', 'flap_raw', 'rud_raw',
                    'delta_flap', 'delta_rud', 'command_scale',
                    'theta', 'v_theta',
                    'ref_x', 'ref_y', 'ref_z',
                    'ref_heading', 'ref_gamma',
                    'solve_time',
                ]
                pred_state_keys = [
                    'pred_x', 'pred_y', 'pred_z',
                    'pred_yaw', 'pred_v', 'pred_vz', 'pred_az',
                    'pred_yaw_rate', 'pred_psi_ddot',
                ]
                pred_cmd_keys = ['pred_u_flap', 'pred_u_rud']
                sample_pred = log[0].get('pred_x', None)
                N_pred = len(sample_pred) if sample_pred is not None else 0
                N_cmd = max(N_pred - 1, 0)

                header = list(scalar_cols)
                for pk in pred_state_keys:
                    for j in range(N_pred):
                        header.append(f'{pk}_{j}')
                for pk in pred_cmd_keys:
                    for j in range(N_cmd):
                        header.append(f'{pk}_{j}')

                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    for e in log:
                        row = []
                        for col in scalar_cols:
                            row.append(e.get(col, ''))
                        for pk in pred_state_keys:
                            arr = e.get(pk, None)
                            if arr is not None:
                                for j in range(N_pred):
                                    row.append(arr[j] if j < len(arr) else '')
                            else:
                                row.extend([''] * N_pred)
                        for pk in pred_cmd_keys:
                            arr = e.get(pk, None)
                            if arr is not None:
                                for j in range(N_cmd):
                                    row.append(arr[j] if j < len(arr) else '')
                            else:
                                row.extend([''] * N_cmd)
                        writer.writerow(row)
                self.get_logger().info(f"Saved {csv_path}")
            except Exception as ex:
                self.get_logger().error(f"CSV export failed: {ex}")

            # ── 240 Hz STATE CSV ──
            state_csv_path = os.path.join(log_dir,
                                          f'flight_{timestamp}_state.csv')
            state_log = self._state_log
            if state_log:
                try:
                    import csv as csv_mod
                    state_cols = [
                        't', 'pos_x', 'pos_y', 'pos_z',
                        'vel_x', 'vel_y', 'vel_z', 'speed',
                        'yaw', 'yaw_rate',
                        'yaw_quat_raw', 'yaw_vel_raw',
                        'flap_cmd', 'rud_cmd',
                    ]
                    with open(state_csv_path, 'w', newline='') as f:
                        writer = csv_mod.writer(f)
                        writer.writerow(state_cols)
                        for e in state_log:
                            writer.writerow([e.get(c, '') for c in state_cols])
                    self.get_logger().info(
                        f"Saved {state_csv_path} "
                        f"({len(state_log)} rows @ ~240Hz)")
                except Exception as ex:
                    self.get_logger().error(f"State CSV export failed: {ex}")
            else:
                state_csv_path = "(no state log)"

            # ── Save parameters ──
            params_path = os.path.join(log_dir,
                                       f'flight_{timestamp}_params.txt')
            try:
                p = self._xfly
                with open(params_path, 'w') as f:
                    f.write("# XFlyParams used for this flight\n")
                    f.write(f"# Timestamp: {timestamp}\n")
                    f.write(f"# Solver: {self._solver_backend}\n")
                    f.write(f"# MPC N={self._mpc_N}, dt={self._mpc_dt}\n")
                    f.write(f"# Trajectory: {type(traj).__name__}\n")
                    f.write(f"# Total arc: {traj.total_arc:.2f}m\n")
                    f.write(f"# Solves: {len(log)}, "
                            f"duration: {t[-1]:.1f}s\n")
                    f.write(f"# State log: {len(state_log)} frames "
                            f"(~{len(state_log)/max(t[-1],0.01):.0f} Hz)\n")
                    f.write(f"# Control rate: {self._ctrl_rate:.0f} Hz\n\n")
                    f.write("[XFlyParams]\n")
                    for field_name in sorted(vars(p)):
                        val = getattr(p, field_name)
                        f.write(f"{field_name} = {val}\n")
                    f.write(f"\n[MPC_Weights]\n")
                    for wk, wv in sorted(self._mpc.weights.items()):
                        f.write(f"{wk} = {wv}\n")
                self.get_logger().info(f"Saved {params_path}")
            except Exception as ex:
                self.get_logger().error(f"Params export failed: {ex}")

            # ── Print stats ──
            self.get_logger().info(
                f"Loop contour error: mean={np.mean(lat_errors_loop):.3f}m, "
                f"max={np.max(lat_errors_loop):.3f}m")
            self.get_logger().info(
                f"Solve time: mean={np.mean(st):.1f}ms, "
                f"max={np.max(st):.1f}ms, "
                f"P95={np.percentile(st, 95):.1f}ms")
            self.get_logger().info(
                f"Flight log: {len(log)} solves over {t[-1]:.1f}s, "
                f"{len(state_log)} state frames")
            print(f"\n{'='*60}")
            print(f"  PLOTS & LOGS SAVED:")
            print(f"    {path1}")
            print(f"    {path2}")
            print(f"    {path3}")
            print(f"    {csv_path}")
            print(f"    {state_csv_path}")
            print(f"    {params_path}")
            print(f"{'='*60}\n")

            try:
                plt.show()
            except Exception:
                pass
            plt.close('all')


# ═══════════════════════════════════════════════════════════════════════
#  SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def run_simulation(duration=20.0, solver_choice='ipopt', knitro_alg=1,
                   disturbance='none', trajectory='straight',
                   trajectory_csv='', n_loops=3, ablation=None):
    import matplotlib.pyplot as plt

    ablation = ablation or AblationFlags()
    dist_cls = DISTURBANCE_MODELS.get(disturbance, DisturbanceNone)
    dist = dist_cls()

    abl_labels = ablation.active_labels()
    abl_tag = ablation.tag()

    print("\n" + "=" * 60)
    print(f"  Solver: {solver_choice.upper()}")
    print(f"  Trajectory: {trajectory}")
    print(f"  Disturbance: {dist.label}")
    if abl_labels:
        print(f"  Ablation [{abl_tag}]:")
        for lbl in abl_labels:
            print(f"    • {lbl}")
    else:
        print(f"  Ablation: BASELINE (all components active)")
    print("=" * 60)

    xfly = XFlyParams()

    if trajectory in ('helix', 'helix_reverse'):
        direction = -1 if trajectory == 'helix_reverse' else 1
        y_sign = direction
        traj = CompositeTrajectory(
            start=(-3.7, y_sign * 1.5, 0.11),
            join=(0.0, y_sign * 1.5, 0.3),
            center=(0.0, 0.0),
            z_top=0.7,
            radius=1.5,
            n_climb_rev=1.0,
            n_level_rev=3.0,
            direction=direction)
        if duration <= 20.0:
            duration = 80.0
            print(f"  (auto duration → {duration}s for helix)")
    else:
        traj = StraightLineTrajectory(
            start=(-3.6, 2.7, 0.1),
            mid=(0.0, 0.0, 0.5),
            end=(3.6, -2.7, 0.9))

    if trajectory == 'external':
        if not EXTERNAL_TRAJ_AVAILABLE:
            print("ERROR: external_trajectory.py not found in path")
            sys.exit(1)
        csv_path = trajectory_csv if trajectory_csv else 'track_1.csv'
        traj = ExternalTrajectory(
            csv_path,
            start_pos=(3, 2, 0.35),
            n_loops=n_loops,
        )
        if duration <= 20.0:
            duration = 150.0
            print(f"  (auto duration → {duration}s for external)")

    gamma0 = traj.gamma1
    u_flap0 = xfly.u_level + gamma0 / xfly.gamma_slope
    v0 = compute_v_eq(xfly, u_flap0)
    psi0 = traj.heading(0)
    print(f"Climb: γ={np.degrees(gamma0):.1f}°, "
          f"ψ={np.degrees(psi0):.1f}°, u_flap={u_flap0:.3f}, v={v0:.2f}")
    print(f"Yaw model: k_yaw={xfly.k_yaw:.3f}, τ_ψ̈={xfly.tau_psi_ddot:.3f}s")

    mpc_dt, mpc_N = 0.1, 20
    mpc, backend = build_mpc(solver_choice, xfly, traj,
                             horizon=mpc_N, dt=mpc_dt,
                             knitro_alg=knitro_alg,
                             ablation=ablation)
    warm = None
    backend_label = f"{backend} [{abl_tag}]" if abl_labels else backend
    print(f"Backend: {backend_label}\n" + "-" * 60)

    # 9-state: [px, py, pz, ψ, v, vz, az, ψ̇, ψ̈]
    x0 = np.array([traj.P0[0], traj.P0[1], traj.P0[2],
                   psi0, v0, 0.0, 0.0, 0.0, 0.0])
    theta = 0.0
    u = np.array([u_flap0, 0.0])

    t_hist, x_hist, u_hist, theta_hist = [0.0], [x0.copy()], [u.copy()], [0.0]
    cont_err, alt_err, solve_times = [], [], []

    dt_sim, dt_ctrl = 0.01, 0.05
    n_steps = int(duration / dt_sim)
    ctrl_steps = int(dt_ctrl / dt_sim)

    for i in range(1, n_steps + 1):
        t = i * dt_sim
        if i % ctrl_steps == 0:
            t_start = time.perf_counter()

            sol = mpc.solve(x0, theta, u, warm)
            warm = sol['warm'] if sol['solve_time'] < 1.0 else None
            u = sol['u']
            theta += dt_ctrl * sol['v_theta']
            theta = min(theta, traj.total_arc)

            t_solve = time.perf_counter() - t_start
            solve_times.append(t_solve)

            ref  = traj.position_np(theta)
            tang = traj.tangent_np(theta)
            err  = x0[:3] - ref
            e_lag = np.dot(err, tang)
            cont_err.append(np.sqrt(max(0, np.dot(err, err) - e_lag**2)))
            alt_err.append(x0[2] - ref[2])

            if hasattr(traj, 's2'):
                seg = ("APPR" if theta < traj.s1 else
                       "HELIX" if theta < traj.s2 else "CIRC")
            else:
                seg = "CLIMB" if theta < traj.s1 else "LEVEL"
            if i % (ctrl_steps * 20) == 0:
                print(
                    f"t={t:5.1f}s [{seg:5s}] θ={theta:5.1f}/{traj.total_arc:.0f} "
                    f"pos=({x0[0]:+5.2f},{x0[1]:+5.2f},{x0[2]:+5.2f}) "
                    f"v={x0[4]:.2f} vz={x0[5]:+.2f} az={x0[6]:+.2f} "
                    f"ψ̇={np.degrees(x0[7]):+.1f}°/s "
                    f"ψ̈={np.degrees(x0[8]):+.1f}°/s² "
                    f"f={u[0]:.2f} r={u[1]:+.3f} "
                    f"{t_solve*1000:.0f}ms")

        x0 = simulate_step(x0, u, dt_sim, xfly,
                           disturbance_fn=lambda _unused, _x: dist(t, _x),
                           ablation=ablation)
        t_hist.append(t); x_hist.append(x0.copy())
        u_hist.append(u.copy()); theta_hist.append(theta)

        if theta >= traj.total_arc - 0.1:
            print(f"\n✓ Done at t={t:.1f}s")
            break

    t_arr    = np.array(t_hist)
    states   = np.array(x_hist)
    controls = np.array(u_hist)
    thetas   = np.array(theta_hist)
    cont_err = np.array(cont_err)
    alt_err  = np.array(alt_err)

    print(f"\nSUMMARY ({backend.upper()}, {dist.label}, {abl_tag})")
    print(f"  Distance: {thetas[-1]:.1f}/{traj.total_arc:.1f}m")
    print(f"  Contour err: mean={np.mean(cont_err):.4f} "
          f"max={np.max(cont_err):.4f}")
    print(f"  Alt err: mean={np.mean(np.abs(alt_err)):.4f}")
    print(f"  Speed: mean={np.mean(states[:,4]):.2f}")
    print(f"  Solver: {np.mean(solve_times)*1000:.1f}ms")
    print(f"  Rudder: [{controls[:,1].min():.3f}, {controls[:,1].max():.3f}]")
    print(f"  ψ̇  range: [{np.degrees(states[:,7].min()):.1f}, "
          f"{np.degrees(states[:,7].max()):.1f}]°/s")
    print(f"  ψ̈  range: [{np.degrees(states[:,8].min()):.1f}, "
          f"{np.degrees(states[:,8].max()):.1f}]°/s²")

    # ── Simulation plots (3×3) ──
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(
        f'MPCC 9-state ({backend.upper()}) — {dist.label} '
        f'[τ_ψ̈={xfly.tau_psi_ddot*1000:.0f}ms] [{abl_tag}]',
        fontsize=14, fontweight='bold')

    s_ref = np.linspace(0, traj.total_arc, 200)
    ref_pts = traj.positions_array(s_ref)

    # (0,0) XY trajectory
    ax = axes[0, 0]
    ax.plot(ref_pts[:, 0], ref_pts[:, 1], 'b--', lw=2, alpha=0.5)
    ax.plot(states[:, 0], states[:, 1], 'r-', lw=1.5)
    ax.plot(states[0, 0], states[0, 1], 'go', ms=10)
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title('XY'); ax.axis('equal'); ax.grid(True, alpha=0.3)

    # (0,1) Altitude
    ax = axes[0, 1]
    refs_t = np.array([traj.position_np(th) for th in thetas])
    ax.plot(t_arr, refs_t[:, 2], 'b--', label='Ref Z')
    ax.plot(t_arr, states[:, 2], 'r-', label='Actual Z')
    ax.set_xlabel('Time'); ax.set_ylabel('Z')
    ax.set_title('Altitude'); ax.legend(); ax.grid(True, alpha=0.3)

    # (0,2) Errors
    ax = axes[0, 2]
    t_err = np.linspace(0, t_arr[-1], len(cont_err))
    ax.plot(t_err, cont_err, 'r-', label='Contour')
    ax.plot(t_err, np.abs(alt_err), 'g-', label='|Alt|')
    ax.set_xlabel('Time'); ax.set_ylabel('Error (m)')
    ax.set_title('Errors'); ax.legend(); ax.grid(True, alpha=0.3)

    # (1,0) Flap
    ax = axes[1, 0]
    ax.plot(t_arr, controls[:, 0], 'b-', lw=1.5)
    ax.axhline(xfly.u_level, color='orange', ls='--', alpha=0.5)
    ax.set_xlabel('Time'); ax.set_ylabel('Flap')
    ax.set_title('Flapping'); ax.grid(True, alpha=0.3)

    # (1,1) Rudder
    ax = axes[1, 1]
    ax.plot(t_arr, controls[:, 1], 'r-', lw=1.5)
    ax.set_xlabel('Time'); ax.set_ylabel('Rudder')
    ax.set_title('Rudder'); ax.grid(True, alpha=0.3)

    # (1,2) Yaw & Yaw Rate
    ax = axes[1, 2]
    ax.plot(t_arr, np.degrees(states[:, 7]), 'm-', lw=1.5, label='ψ̇')
    ax2 = ax.twinx()
    yaw_deg_sim = (np.degrees(states[:, 3]) + 180.0) % 360.0 - 180.0
    ax2.plot(t_arr, yaw_deg_sim, 'b-', lw=1, alpha=0.5, label='ψ')
    ax.set_xlabel('Time'); ax.set_ylabel('ψ̇ (°/s)', color='m')
    ax2.set_ylabel('ψ (°)', color='b')
    ax.set_title('Yaw & Yaw Rate')
    ax.legend(loc='upper left'); ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # (2,0) Yaw acceleration ψ̈ — the new state
    ax = axes[2, 0]
    ax.plot(t_arr, np.degrees(states[:, 8]), color='darkorchid',
            lw=1.5, label='ψ̈ (state)')
    ax.axhline(0, color='gray', alpha=0.3)
    # Overlay rudder on twin axis
    ax2 = ax.twinx()
    ax2.plot(t_arr, controls[:, 1], 'r-', lw=0.8, alpha=0.5, label='Rudder')
    ax2.set_ylabel('Rudder', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('ψ̈ (°/s²)', color='darkorchid')
    ax.tick_params(axis='y', labelcolor='darkorchid')
    ax.set_title(f'Yaw Acceleration ψ̈ (τ={xfly.tau_psi_ddot*1000:.0f}ms lag)')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2,1) Vertical velocity (vz)
    ax = axes[2, 1]
    ax.plot(t_arr, states[:, 5], 'b-', lw=1.5, label='vz')
    ax.axhline(0, color='gray', alpha=0.3)
    ax.set_xlabel('Time'); ax.set_ylabel('vz (m/s)')
    ax.set_title('Vertical Velocity'); ax.legend(); ax.grid(True, alpha=0.3)

    # (2,2) Speed
    ax = axes[2, 2]
    ax.plot(t_arr, states[:, 4], 'g-', lw=1.5, label='Speed')
    ax.set_xlabel('Time'); ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed'); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.expanduser(f'~/mpcc_sim_{abl_tag}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved {out_path}")
    plt.show()

    # ── Solve time distribution ──
    st = np.array(solve_times) * 1000
    ctrl_idx = np.arange(len(st)) * dt_ctrl

    fig2, (ax_hist, ax_time) = plt.subplots(1, 2, figsize=(16, 5))
    fig2.suptitle(f'MPC Solve Time Distribution ({backend.upper()})',
                  fontsize=13, fontweight='bold')

    ax_hist.hist(st, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax_hist.axvline(np.mean(st), color='red', ls='--', lw=1.5,
                    label=f'Mean: {np.mean(st):.2f} ms')
    ax_hist.axvline(np.median(st), color='orange', ls='--', lw=1.5,
                    label=f'Median: {np.median(st):.2f} ms')
    ax_hist.axvline(np.percentile(st, 95), color='darkred', ls=':', lw=1.5,
                    label=f'P95: {np.percentile(st, 95):.2f} ms')
    ax_hist.axvline(np.max(st), color='black', ls=':', lw=1,
                    label=f'Max: {np.max(st):.2f} ms')
    ax_hist.axvline(dt_ctrl * 1000, color='red', ls='-', lw=2, alpha=0.3,
                    label=f'Control dt: {dt_ctrl*1000:.0f} ms')
    ax_hist.set_xlabel('Solve time (ms)')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Distribution')
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)

    ax_time.plot(ctrl_idx, st, 'b-', lw=0.8, alpha=0.7)
    ax_time.axhline(np.mean(st), color='red', ls='--', lw=1,
                    label=f'Mean: {np.mean(st):.2f} ms')
    ax_time.axhline(dt_ctrl * 1000, color='red', ls='-', lw=2,
                    alpha=0.3, label=f'Control dt: {dt_ctrl*1000:.0f} ms')
    ax_time.set_xlabel('Time (s)')
    ax_time.set_ylabel('Solve time (ms)')
    ax_time.set_title('Over time')
    ax_time.legend(fontsize=9)
    ax_time.grid(True, alpha=0.3)

    fig2.tight_layout()
    out_path2 = os.path.expanduser(f'~/mpcc_solve_time_sim_{abl_tag}.png')
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight')

    print(f"\nSolve time stats ({backend.upper()}):")
    print(f"  Mean:   {np.mean(st):.2f} ms")
    print(f"  Median: {np.median(st):.2f} ms")
    print(f"  Std:    {np.std(st):.2f} ms")
    print(f"  Min:    {np.min(st):.2f} ms")
    print(f"  Max:    {np.max(st):.2f} ms")
    print(f"  P95:    {np.percentile(st, 95):.2f} ms")
    print(f"  P99:    {np.percentile(st, 99):.2f} ms")
    print(f"  N:      {len(st)} solves")
    print(f"Saved {out_path2}")
    plt.show()

    return {
        'backend': backend,
        'ablation_tag': abl_tag,
        'solve_times_ms': st,
        'cont_err': cont_err,
        'alt_err': alt_err,
        'distance': thetas[-1],
        'mean_speed': np.mean(states[:, 4]),
        'positions': states[:, :3],
        'total_arc': traj.total_arc,
    }


def main():
    parser = argparse.ArgumentParser(
        description="MPCC Controller for XFly")
    parser.add_argument("--sim", action="store_true",
                        help="Run in simulation mode")
    parser.add_argument("--real", action="store_true",
                        help="Run with ROS 2 (real or bag)")
    parser.add_argument("--solver", type=str, default="ipopt",
                        choices=["ipopt", "knitro"],
                        help="NLP solver backend (default: ipopt)")
    parser.add_argument("--knitro-alg", type=str, default="2",
                        help="Knitro algorithm: 0=Auto, 1=Interior/Direct, "
                             "2=Interior/CG, 3=Active Set, 4=SQP, 5=Multi")
    parser.add_argument("--disturbance", type=str, default="none",
                        choices=["none", "wind", "gust", "turbulence"],
                        help="Disturbance type for simulation")
    parser.add_argument("--trajectory", type=str, default="straight",
                        choices=["straight", "helix", "helix_reverse", "external"],
                        help="Trajectory type")
    parser.add_argument("--trajectory-csv", type=str, default="",
                        help="Path to external trajectory CSV "
                             "(used with --trajectory=external)")
    parser.add_argument("--n-loops", type=int, default=6,
                        help="Number of racing loops (external trajectory)")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--sysid", type=str, default="",
                        choices=["", "flap_id", "rudder_id", "full_id"],
                        help="System identification test")
    parser.add_argument("--adapt-flap", action="store_true",
                        help="Enable adaptive flap scaling")

    # ── Ablation study flags ──
    abl_group = parser.add_argument_group(
        'Ablation study',
        'Each flag removes exactly one design choice for ablation analysis.')
    abl_group.add_argument(
        "--abl-no-alt-coupling", action="store_true",
        help="(i) Remove turn-induced altitude coupling k_psi_z * psi_dot^2")
    abl_group.add_argument(
        "--abl-2nd-order-yaw", action="store_true",
        help="(ii) Collapse 3rd-order yaw to 2nd order (rudder → psi_dot)")
    abl_group.add_argument(
        "--abl-fixed-yaw-airspeed", action="store_true",
        help="(iii) Replace speed-dependent yaw cmd with fixed v_bar")
    abl_group.add_argument(
        "--abl-no-flap-ff", action="store_true",
        help="(iv) Remove flapping feedforward term from cost")
    abl_group.add_argument(
        "--abl-no-heading", action="store_true",
        help="(v) Remove heading alignment penalty from cost")
    abl_group.add_argument(
        "--abl-no-lag", action="store_true",
        help="(vi) Remove along-track lag penalty from cost")
    abl_group.add_argument(
        "--abl-merged-contour", action="store_true",
        help="(vii) Merge XY/Z contouring into single 3D weight")
    abl_group.add_argument(
        "--abl-euler-sim", action="store_true",
        help="(viii) Use forward-Euler instead of RK4 in simulation plant")
    abl_group.add_argument(
        "--abl-all", action="store_true",
        help="Run ALL single-ablation experiments sequentially and "
             "print a comparison table")

    args, ros_args = parser.parse_known_args()

    # ── Build ablation flags ──
    ablation = AblationFlags(
        no_altitude_coupling = args.abl_no_alt_coupling,
        no_yaw_accel_state   = args.abl_2nd_order_yaw,
        fixed_yaw_airspeed   = args.abl_fixed_yaw_airspeed,
        no_flap_feedforward  = args.abl_no_flap_ff,
        no_heading_penalty   = True, # args.abl_no_heading,
        no_lag_penalty       = args.abl_no_lag,
        no_contour_z_split   = args.abl_merged_contour,
        euler_integration    = False, #args.abl_euler_sim,
    )

    if not args.sim and not args.real:
        if ROS_AVAILABLE:
            args.real = True
        else:
            args.sim = True

    print(f"Solver availability: "
          f"IPOPT=✓  "
          f"Knitro={'✓' if KNITRO_AVAILABLE else '✗'}")

    if args.sim:
        knitro_alg = int(args.knitro_alg) if args.knitro_alg.isdigit() else 1

        if args.abl_all:
            # ── Run full ablation study: baseline + each single ablation ──
            import matplotlib
            matplotlib.use('Agg')   # non-interactive for batch
            import matplotlib.pyplot as plt

            configs = [
                ("baseline",           AblationFlags()),
                ("(i) no alt coupling", AblationFlags(no_altitude_coupling=True)),
                ("(ii) 2nd-order yaw", AblationFlags(no_yaw_accel_state=True)),
                ("(iii) fixed v_yaw",  AblationFlags(fixed_yaw_airspeed=True)),
                ("(iv) no flap FF",    AblationFlags(no_flap_feedforward=True)),
                ("(v) no heading",     AblationFlags(no_heading_penalty=True)),
                ("(vi) no lag",        AblationFlags(no_lag_penalty=True)),
                ("(vii) merged 3D",    AblationFlags(no_contour_z_split=True)),
                ("(viii) Euler sim",   AblationFlags(euler_integration=True)),
            ]

            results = []
            for label, abl_cfg in configs:
                print(f"\n{'#'*60}")
                print(f"  ABLATION: {label}")
                print(f"{'#'*60}")
                try:
                    res = run_simulation(
                        duration=args.duration,
                        solver_choice=args.solver,
                        knitro_alg=knitro_alg,
                        disturbance=args.disturbance,
                        trajectory=args.trajectory,
                        trajectory_csv=args.trajectory_csv,
                        n_loops=args.n_loops,
                        ablation=abl_cfg)
                    results.append((label, res))
                except Exception as e:
                    print(f"  *** FAILED: {e}")
                    results.append((label, None))
                plt.close('all')

            # ── Print comparison table ──
            print("\n" + "=" * 90)
            print(f"  ABLATION STUDY RESULTS — {args.trajectory} trajectory, "
                  f"{args.disturbance} disturbance")
            print("=" * 90)
            header = (f"{'Config':<24s} {'Cont(mean)':>10s} {'Cont(max)':>10s} "
                      f"{'Alt(mean)':>10s} {'Speed':>7s} {'Dist':>7s} "
                      f"{'Solve(ms)':>10s}")
            print(header)
            print("-" * 90)
            for label, res in results:
                if res is None:
                    print(f"{label:<24s}  {'FAILED':>10s}")
                    continue
                ce = res['cont_err']
                ae = res['alt_err']
                print(f"{label:<24s} "
                      f"{np.mean(ce):10.4f} {np.max(ce):10.4f} "
                      f"{np.mean(np.abs(ae)):10.4f} "
                      f"{res['mean_speed']:7.2f} "
                      f"{res['distance']:7.1f} "
                      f"{np.mean(res['solve_times_ms']):10.2f}")
            print("=" * 90)
            return

        # ── Single run (possibly with individual ablation flags) ──
        run_simulation(duration=args.duration,
                       solver_choice=args.solver,
                       knitro_alg=knitro_alg,
                       disturbance=args.disturbance,
                       trajectory=args.trajectory,
                       trajectory_csv=args.trajectory_csv,
                       n_loops=args.n_loops,
                       ablation=ablation)
        return

    if not ROS_AVAILABLE:
        print("ERROR: ROS 2 not found")
        sys.exit(1)

    if args.solver:
        ros_args += ['--ros-args', '-p', f'solver:={args.solver}']
    if args.trajectory:
        ros_args += ['--ros-args', '-p', f'trajectory:={args.trajectory}']
    if args.sysid:
        ros_args += ['--ros-args', '-p', f'sysid:={args.sysid}']
    if args.adapt_flap:
        ros_args += ['--ros-args', '-p', 'adapt_flap:=true']
    if args.trajectory_csv:
        ros_args += ['--ros-args', '-p', f'trajectory_csv:={args.trajectory_csv}']
    ros_args += ['--ros-args', '-p', f'n_loops:={args.n_loops}']

    rclpy.init(args=ros_args)
    node = MPCCXFlyNode(ablation=ablation)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Spin error: {e}")
    finally:
        try:
            node.get_logger().info(
                f"Shutting down — {len(node._flight_log)} log entries")
            node.plot_flight_log()
        except Exception as e:
            import traceback
            print(f"Plot failed: {e}")
            traceback.print_exc()

        try:
            if node._armed:
                node._publish_commands(0.0, 0.0)
        except Exception:
            pass

        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
