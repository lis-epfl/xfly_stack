"""
ExternalTrajectory v7 — Unified B-spline, zero branching
=========================================================
CasADi path: ONE B-spline per axis over [0, total_arc].
  - Hermite-smoothed approach + unrolled loop sampled into a
    single LUT → 3 interpolant calls, no if_else, no fmod.
  - Tangent from jacobian of the unified interpolant.
NumPy path: Hermite (approach) + periodic B-spline (loop).
"""

import numpy as np
import casadi as ca
import csv
import os


class ExternalTrajectory:
    """
    MPCC-compatible trajectory from optimizer CSV.

    CasADi interface uses a single unified B-spline per axis over
    the full arc-length domain [0, total_arc]. No branching, no
    modular arithmetic — minimal expression graph for the solver.

    NumPy interface uses the exact Hermite curve (approach) and
    periodic B-spline (loop) for high-fidelity evaluation.
    """

    def __init__(self,
                 csv_path: str,
                 start_pos=(3, 2.5, 0.35),
                 n_loops=3,
                 n_approach_pts=6,
                 ds_resample=1.0):
        """
        Parameters
        ----------
        csv_path : str
            Trajectory CSV (seg, t, x, y, z, ...).
        start_pos : tuple
            Ground start position (aligned with loop entry tangent).
        n_loops : int
            Racing loop repetitions.
        n_approach_pts : int
            LUT points for approach (few needed — it's straight).
        ds_resample : float
            Arc-length spacing for loop resampling (meters).
            1.0m works well with cubic B-spline smoothing.
        """
        self.start_pos = np.array(start_pos)
        self.n_loops = n_loops

        # ── Load CSV ──
        raw_pts = []
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                raw_pts.append([float(row['x']), float(row['y']),
                                float(row['z'])])
        raw_pts = np.array(raw_pts)

        # ── Find segment boundaries (gate positions) ──
        seg_starts = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = int(row['seg'])
                if s not in seg_starts:
                    seg_starts[s] = len(seg_starts)
        n_segs = len(seg_starts)
        pts_per_seg = len(raw_pts) // n_segs

        gate_positions = []
        for si in range(n_segs):
            gate_positions.append(raw_pts[si * pts_per_seg].copy())

        # ── Rotate loop so it starts from the gate closest to start_pos ──
        max_z_diff = 0.5
        dists = [np.linalg.norm(self.start_pos - gp) for gp in gate_positions]
        z_diffs = [abs(self.start_pos[2] - gp[2]) for gp in gate_positions]
        eligible = [i for i in range(len(gate_positions))
                    if z_diffs[i] <= max_z_diff]
        if not eligible:
            print(f"  WARNING: no gate within {max_z_diff}m height diff "
                  f"of start_pos (z={self.start_pos[2]:.2f}). "
                  f"Falling back to closest gate.")
            closest_gate = int(np.argmin(dists))
        else:
            closest_gate = int(min(eligible, key=lambda i: dists[i]))

        if closest_gate != 0:
            rotate_by = closest_gate * pts_per_seg
            raw_pts = np.vstack([raw_pts[rotate_by:], raw_pts[:rotate_by]])
            print(f"  Rotated loop: start gate G{closest_gate+1} "
                  f"(dist={dists[closest_gate]:.2f}m from start_pos)")
        else:
            print(f"  Loop already starts from closest gate G1 "
                  f"(dist={dists[0]:.2f}m)")

        # ── Arc-length for one closed loop ──
        loop_pts = np.vstack([raw_pts, raw_pts[0:1]])
        loop_arc = np.zeros(len(loop_pts))
        for i in range(1, len(loop_pts)):
            loop_arc[i] = loop_arc[i-1] + np.linalg.norm(
                loop_pts[i] - loop_pts[i-1])
        self.loop_length = loop_arc[-1]

        # ── Resample loop uniformly ──
        n_loop_pts = max(int(self.loop_length / ds_resample), 30)
        loop_s_uniform = np.linspace(0, self.loop_length, n_loop_pts,
                                      endpoint=False)
        loop_resampled = np.zeros((n_loop_pts, 3))
        for i, s in enumerate(loop_s_uniform):
            idx = np.searchsorted(loop_arc, s, side='right') - 1
            idx = max(0, min(idx, len(loop_pts) - 2))
            ds_seg = loop_arc[idx+1] - loop_arc[idx]
            frac = (s - loop_arc[idx]) / max(ds_seg, 1e-12)
            loop_resampled[i] = (loop_pts[idx] +
                                  frac * (loop_pts[idx+1] - loop_pts[idx]))

        self._loop_resampled = loop_resampled
        self._loop_s_uniform = loop_s_uniform

        # ── Approach geometry ──
        self.P1 = loop_resampled[0].copy()
        self.P0 = np.array(start_pos, dtype=float)

        d_app = self.P1 - self.P0
        self.L_approach = np.linalg.norm(d_app)
        self.dir_approach = d_app / max(self.L_approach, 1e-10)
        self.gamma1 = np.arctan2(
            d_app[2], np.sqrt(d_app[0]**2 + d_app[1]**2))
        self.gamma_approach = self.gamma1
        self.s_approach = self.L_approach
        self.s1 = self.s_approach
        self.total_arc = self.L_approach + n_loops * self.loop_length

        # ── Periodic loop B-spline (for NumPy interface) ──
        n_ghost = 3
        ghost_s_lo = loop_s_uniform[-n_ghost:] - self.loop_length
        ghost_s_hi = loop_s_uniform[:n_ghost] + self.loop_length
        ghost_pts_lo = loop_resampled[-n_ghost:]
        ghost_pts_hi = loop_resampled[:n_ghost]

        periodic_s = np.concatenate([ghost_s_lo, loop_s_uniform, ghost_s_hi])
        periodic_pts = np.vstack([ghost_pts_lo, loop_resampled, ghost_pts_hi])

        self._ca_loop_px = ca.interpolant('lpx', 'bspline',
            [periodic_s], periodic_pts[:, 0])
        self._ca_loop_py = ca.interpolant('lpy', 'bspline',
            [periodic_s], periodic_pts[:, 1])
        self._ca_loop_pz = ca.interpolant('lpz', 'bspline',
            [periodic_s], periodic_pts[:, 2])

        # Loop tangent (for NumPy interface)
        s_sym = ca.MX.sym('s_loop')
        lpx = self._ca_loop_px(s_sym)
        lpy = self._ca_loop_py(s_sym)
        lpz = self._ca_loop_pz(s_sym)
        dlpx = ca.jacobian(lpx, s_sym)
        dlpy = ca.jacobian(lpy, s_sym)
        dlpz = ca.jacobian(lpz, s_sym)
        tn = ca.sqrt(dlpx**2 + dlpy**2 + dlpz**2 + 1e-12)
        self._ca_loop_tangent = ca.Function('loop_tang', [s_sym],
            [dlpx/tn, dlpy/tn, dlpz/tn],
            ['s'], ['tx', 'ty', 'tz'])

        # ── Hermite approach: compute tangent-matched curve ──
        loop_tan_0 = self._ca_loop_tangent(0.0)
        self._loop_entry_tangent = np.array([
            float(loop_tan_0[0]), float(loop_tan_0[1]),
            float(loop_tan_0[2])])

        L = self.L_approach
        self._hermite_M0 = L * self.dir_approach
        self._hermite_M1 = L * self._loop_entry_tangent

        print(f"  Approach tangent at start: "
              f"{tuple(np.round(self.dir_approach, 4))}")
        print(f"  Loop tangent at entry:     "
              f"{tuple(np.round(self._loop_entry_tangent, 4))}")

        # ══════════════════════════════════════════════════════
        #  UNIFIED B-SPLINE for CasADi (approach + unrolled loop)
        #  → 3 interpolant calls, zero branching, zero fmod
        # ══════════════════════════════════════════════════════

        # Sample the Hermite approach densely (especially near the
        # junction where curvature is highest)
        n_app_samples = max(int(self.L_approach / (ds_resample * 0.25)), 20)
        uni_s = []
        uni_pts = []
        for i in range(n_app_samples):
            sv = i * self.L_approach / n_app_samples
            tv = sv / max(self.L_approach, 1e-10)
            tv2, tv3 = tv*tv, tv*tv*tv
            h00 = 2*tv3 - 3*tv2 + 1
            h10 = tv3 - 2*tv2 + tv
            h01 = -2*tv3 + 3*tv2
            h11 = tv3 - tv2
            pt = (h00*self.P0 + h10*self._hermite_M0
                  + h01*self.P1 + h11*self._hermite_M1)
            uni_s.append(sv)
            uni_pts.append(pt)

        # Sample the loop from the periodic B-spline (inherits C2
        # smoothness across loop boundaries automatically)
        s_offset = self.L_approach
        for loop_i in range(n_loops):
            for i in range(n_loop_pts):
                s_local = loop_s_uniform[i]
                uni_s.append(s_offset + s_local)
                uni_pts.append(np.array([
                    float(self._ca_loop_px(s_local)),
                    float(self._ca_loop_py(s_local)),
                    float(self._ca_loop_pz(s_local)),
                ]))
            s_offset += self.loop_length

        # Final point (close the last loop)
        uni_s.append(s_offset)
        uni_pts.append(np.array([
            float(self._ca_loop_px(0.0)),
            float(self._ca_loop_py(0.0)),
            float(self._ca_loop_pz(0.0)),
        ]))

        uni_s = np.array(uni_s)
        uni_pts = np.array(uni_pts)

        # Build 3 unified CasADi B-spline interpolants
        self._ca_uni_px = ca.interpolant('upx', 'bspline',
            [uni_s], uni_pts[:, 0])
        self._ca_uni_py = ca.interpolant('upy', 'bspline',
            [uni_s], uni_pts[:, 1])
        self._ca_uni_pz = ca.interpolant('upz', 'bspline',
            [uni_s], uni_pts[:, 2])

        # Unified tangent from jacobian
        su = ca.MX.sym('s_uni')
        upx = self._ca_uni_px(su)
        upy = self._ca_uni_py(su)
        upz = self._ca_uni_pz(su)
        dupx = ca.jacobian(upx, su)
        dupy = ca.jacobian(upy, su)
        dupz = ca.jacobian(upz, su)
        tnu = ca.sqrt(dupx**2 + dupy**2 + dupz**2 + 1e-12)
        self._ca_uni_tangent = ca.Function('uni_tang', [su],
            [dupx/tnu, dupy/tnu, dupz/tnu],
            ['s'], ['tx', 'ty', 'tz'])

        # ── LUT for legacy NumPy interface (plotting, etc.) ──
        self._all_s = uni_s
        self._all_pts = uni_pts

        n_total = len(self._all_pts)
        self._all_tan = np.zeros((n_total, 3))
        for i in range(n_total - 1):
            d = self._all_pts[i+1] - self._all_pts[i]
            nrm = np.linalg.norm(d)
            self._all_tan[i] = d / max(nrm, 1e-10)
        self._all_tan[-1] = self._all_tan[-2]

        n_uni = len(uni_s)
        print(f"ExternalTrajectory: loaded {len(raw_pts)} pts "
              f"from {os.path.basename(csv_path)}")
        print(f"  Approach: {self.L_approach:.2f}m, "
              f"\u03b3={np.degrees(self.gamma1):.1f}\u00b0")
        print(f"  Loop: {self.loop_length:.2f}m \u00d7 {n_loops} = "
              f"{self.loop_length * n_loops:.2f}m")
        print(f"  Total arc: {self.total_arc:.2f}m")
        print(f"  Unified LUT: {n_uni} pts "
              f"(~{ds_resample:.1f}m spacing)")
        print(f"  Start: {tuple(np.round(self.P0, 3))}")
        print(f"  Join:  {tuple(np.round(self.P1, 3))}")

    # ═══════════ CasADi symbolic interface ═══════════
    # Single interpolant call per axis — no if_else, no fmod.

    def position(self, s):
        sc = ca.fmin(ca.fmax(s, 0), self.total_arc - 1e-6)
        return (self._ca_uni_px(sc),
                self._ca_uni_py(sc),
                self._ca_uni_pz(sc))

    def tangent(self, s):
        sc = ca.fmin(ca.fmax(s, 0), self.total_arc - 1e-6)
        out = self._ca_uni_tangent(sc)
        return (out[0], out[1], out[2])

    def climb_angle_ca(self, s):
        _, _, tz = self.tangent(s)
        return ca.asin(ca.fmin(ca.fmax(tz, -0.99), 0.99))

    # ═══════════ NumPy interface ═══════════
    # Uses exact Hermite (approach) + periodic B-spline (loop).

    def _hermite_pos_np(self, t):
        """NumPy cubic Hermite position, t in [0,1]."""
        t2 = t * t
        t3 = t2 * t
        h00 = 2*t3 - 3*t2 + 1
        h10 = t3 - 2*t2 + t
        h01 = -2*t3 + 3*t2
        h11 = t3 - t2
        return (h00*self.P0 + h10*self._hermite_M0
                + h01*self.P1 + h11*self._hermite_M1)

    def _hermite_deriv_np(self, t):
        """NumPy cubic Hermite dp/dt, t in [0,1]."""
        t2 = t * t
        dh00 = 6*t2 - 6*t
        dh10 = 3*t2 - 4*t + 1
        dh01 = -6*t2 + 6*t
        dh11 = 3*t2 - 2*t
        return (dh00*self.P0 + dh10*self._hermite_M0
                + dh01*self.P1 + dh11*self._hermite_M1)

    def position_np(self, s):
        """Evaluate position using Hermite (approach) or periodic
        B-spline (loop)."""
        s = float(np.clip(s, 0, self.total_arc - 1e-6))
        if s < self.s_approach:
            t = s / self.L_approach
            return self._hermite_pos_np(t)
        s_loop = (s - self.s_approach) % self.loop_length
        return np.array([
            float(self._ca_loop_px(s_loop)),
            float(self._ca_loop_py(s_loop)),
            float(self._ca_loop_pz(s_loop)),
        ])

    def tangent_np(self, s):
        """Evaluate tangent using Hermite derivative (approach) or
        periodic B-spline derivative (loop)."""
        s = float(np.clip(s, 0, self.total_arc - 1e-6))
        if s < self.s_approach:
            t = s / self.L_approach
            d = self._hermite_deriv_np(t)
            nrm = np.linalg.norm(d)
            return d / max(nrm, 1e-10)
        s_loop = (s - self.s_approach) % self.loop_length
        out = self._ca_loop_tangent(s_loop)
        return np.array([float(out[0]), float(out[1]), float(out[2])])

    def heading(self, s):
        t = self.tangent_np(s)
        return np.arctan2(t[1], t[0])

    def climb_angle_np(self, s):
        t = self.tangent_np(s)
        return np.arctan2(t[2], np.sqrt(t[0]**2 + t[1]**2))

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


def load_external_trajectory(csv_path, start_pos=(3, 2.5, 0.35),
                              n_loops=3, **kwargs):
    return ExternalTrajectory(csv_path, start_pos=start_pos,
                               n_loops=n_loops, **kwargs)
