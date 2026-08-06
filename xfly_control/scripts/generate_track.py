#!/usr/bin/env python3
"""
generate_track.py — End-to-end racing trajectory generator
===========================================================
Merge of `optimize_track.py` (optimizer core) and
`generate_trajectory.py` (post-processing), for the XFly MPCC
controller.

Given gate positions/yaws plus drone constraints, produces a
trajectory CSV ready for `external_trajectory.py`.

Pipeline:
  1. Define gates (position + yaw, optional final height)
  2. Optimize quintic Bezier control points — minimize the
     curvature integral subject to turn radius, climb angle,
     bounding box and gate clearance. C1 continuity is structural
     (see TrajectoryOptimizer); C2 is penalized.
  3. Sample the Bezier segments
  4. Reorder so the loop starts from the lowest gate
     (short ground approach)
  5. Rescale Z onto the desired final gate heights
  6. Recompute curvature/climb from the *final* geometry and
     re-validate
  7. Export CSV + gates JSON (+ optional plot)

Usage:
  python generate_track.py --track oval
  python generate_track.py --gates my_gates.json --auto-yaw
  python generate_track.py --track wave --output wave.csv --plot

Then feed the CSV to the controller:
  python mpcc_node.py --sim --trajectory external \
      --trajectory-csv trajectory.csv --n-loops 3

Exit status is 0 when the exported trajectory meets every
constraint and 1 otherwise, so it can be used in a build step.
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize


# ═══════════════════════════════════════════════════════════════
#  QUINTIC BEZIER MATH
# ═══════════════════════════════════════════════════════════════

def qb(t, P):
    """Evaluate quintic Bezier. P:(6,3), t:(N,) -> (N,3)"""
    t = np.asarray(t, dtype=float).reshape(-1, 1); s = 1 - t
    return np.column_stack([s**5, 5*s**4*t, 10*s**3*t**2,
                            10*s**2*t**3, 5*s*t**4, t**5]) @ P


def qb1(t, P):
    """First derivative C'(t)."""
    t = np.asarray(t, dtype=float).reshape(-1, 1); s = 1 - t
    dP = 5 * np.diff(P, axis=0)
    return np.column_stack([s**4, 4*s**3*t, 6*s**2*t**2,
                            4*s*t**3, t**4]) @ dP


def qb2(t, P):
    """Second derivative C''(t)."""
    t = np.asarray(t, dtype=float).reshape(-1, 1); s = 1 - t
    ddP = 4 * np.diff(5 * np.diff(P, axis=0), axis=0)
    return np.column_stack([s**3, 3*s**2*t, 3*s*t**2, t**3]) @ ddP


def curvature(t, P):
    """κ(t) = |C'×C''| / |C'|³"""
    d1 = qb1(t, P); d2 = qb2(t, P)
    return (np.linalg.norm(np.cross(d1, d2), axis=1) /
            np.maximum(np.linalg.norm(d1, axis=1), 1e-10)**3)


def climb_rate(t, P):
    """Climb rate |dz/ds| = sin(climb angle)."""
    d1 = qb1(t, P)
    return np.abs(d1[:, 2]) / np.maximum(np.linalg.norm(d1, axis=1), 1e-10)


def arc_length(P, n=200):
    pts = qb(np.linspace(0, 1, n), P)
    return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))


def climb_deg(cr):
    """Climb angle in degrees from climb rate (|dz/ds| = sin γ)."""
    return np.degrees(np.arcsin(np.clip(cr, -1.0, 1.0)))


def _curv_vec(P, t):
    """Curvature vector dT/ds at parameter t, in 1/m.

    Unlike C'', this is independent of how the segment is
    parameterized, so comparing it across a junction measures
    geometric curvature continuity rather than a coincidence of
    control-point spacing.
    """
    d1 = qb1(np.array([t]), P)[0]
    d2 = qb2(np.array([t]), P)[0]
    sp = max(np.linalg.norm(d1), 1e-9)
    T = d1 / sp
    return (d2 - np.dot(d2, T) * T) / sp**2


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class DroneConfig:
    """Physical constraints for the flapping-wing drone."""
    min_turn_radius: float = 1.8        # meters
    max_climb_deg: float = 20.0         # degrees
    max_speed: float = 4.0              # m/s (lap-time estimate only)
    gate_width: float = 0.80            # meters
    gate_height: float = 0.60           # meters
    drone_margin: float = 0.10          # meters

    @property
    def max_climb_rate(self):
        """sin(γ_max) — compared against |dz/ds|."""
        return np.sin(np.radians(self.max_climb_deg))


@dataclass
class ArenaConfig:
    """Arena geometry."""
    flight_box: Tuple = (7.5, 7.5, 4.0)    # trajectory must stay inside
    center: Tuple = (0.0, 0.0, 1.5)        # box center
    min_traj_z: float = 0.15               # ground clearance


class Gate:
    """A gate defined by position and yaw (rotation about z).

    `z` is the height used during optimization; `z_final` (optional)
    is the height the gate should end up at after Z rescaling. Set
    `z_final` to None to keep `z`.
    """

    def __init__(self, x, y, z, yaw_deg, z_final=None):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.yaw_deg = float(yaw_deg)
        self.z_final = None if z_final is None else float(z_final)

    @property
    def yaw(self):
        return np.radians(self.yaw_deg)

    @property
    def position(self):
        return np.array([self.x, self.y, self.z])

    @property
    def normal(self):
        return np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])

    @property
    def z_out(self):
        """Final height (z_final if set, else z)."""
        return self.z if self.z_final is None else self.z_final

    def __repr__(self):
        zf = "" if self.z_final is None else f" → z={self.z_final:.3f}"
        return (f"Gate({self.x:+.3f}, {self.y:+.3f}, z={self.z:.3f}, "
                f"yaw={self.yaw_deg:+.1f}°{zf})")


# ═══════════════════════════════════════════════════════════════
#  OPTIMIZER
# ═══════════════════════════════════════════════════════════════

class TrajectoryOptimizer:
    """Quintic Bezier trajectory optimizer with DE + L-BFGS-B.

    One quintic Bezier per gate-to-gate segment, with C1 continuity
    enforced *structurally* rather than by penalty.

    Every gate i carries a tangent vector

        T_i = d_i · [cos γ_i·cos ψ_i, cos γ_i·sin ψ_i, sin γ_i]

    where ψ_i is the gate yaw (fixed), d_i > 0 is a tangent length
    and γ_i is a climb angle. Segment i is then built as

        P0 = g_i        P1 = g_i + T_i
        P2, P3 = free   P4 = g_{i+1} − T_{i+1}    P5 = g_{i+1}

    so C'(1) of segment i−1 and C'(0) of segment i are both exactly
    5·T_i — identical in direction *and* magnitude. Kinks and cusps
    at gates are therefore unrepresentable, and the trajectory
    always crosses a gate along its yaw. Bounding γ_i by the climb
    limit also enforces that limit exactly at the gates.

    Variables: 2 per gate (d_i, γ_i) + 6 per segment (P2, P3) = 8n.
    """

    def __init__(self, gates, drone=None, arena=None,
                 min_turn_radius=None, max_climb_deg=None,
                 gate_width=None, gate_height=None, drone_margin=None,
                 flight_box=None, flight_center=None, min_traj_z=None,
                 max_tangent=None, n_eval=100,
                 w_c2=150.0, w_z_band=100.0, w_z_mono=400.0):
        self.drone = drone if drone is not None else DroneConfig()
        self.arena = arena if arena is not None else ArenaConfig()

        # Penalty weights for curvature continuity and height shaping.
        # All three saturate well below these values — raising them
        # by 10x changes nothing — so they are plateaus, not knobs to
        # tune per track.
        self.w_c2 = float(w_c2)
        self.w_z_band = float(w_z_band)
        self.w_z_mono = float(w_z_mono)

        # Explicit kwargs override the config objects
        if min_turn_radius is not None:
            self.drone.min_turn_radius = min_turn_radius
        if max_climb_deg is not None:
            self.drone.max_climb_deg = max_climb_deg
        if gate_width is not None:
            self.drone.gate_width = gate_width
        if gate_height is not None:
            self.drone.gate_height = gate_height
        if drone_margin is not None:
            self.drone.drone_margin = drone_margin
        if flight_box is not None:
            self.arena.flight_box = tuple(flight_box)
        if flight_center is not None:
            self.arena.center = tuple(flight_center)
        if min_traj_z is not None:
            self.arena.min_traj_z = min_traj_z

        self.gates = list(gates)
        self.n = len(self.gates)
        if self.n < 2:
            raise ValueError("need at least 2 gates")

        self.min_turn_radius = self.drone.min_turn_radius
        self.max_climb_deg = self.drone.max_climb_deg
        self.max_climb_rate = self.drone.max_climb_rate
        self.mk = 1.0 / self.min_turn_radius

        fb = np.array(self.arena.flight_box, dtype=float)
        ctr = np.array(self.arena.center, dtype=float)
        self.flo = ctr - fb / 2
        self.fhi = ctr + fb / 2
        self.min_traj_z = self.arena.min_traj_z
        self.flo[2] = self.min_traj_z

        self.gate_half_w = self.drone.gate_width / 2
        self.gate_half_h = self.drone.gate_height / 2
        self.drone_margin = self.drone.drone_margin
        self.n_eval = n_eval

        # Precomputed evaluation points
        self.te = np.linspace(0, 1, n_eval)
        self.t_near = np.linspace(0.0, 0.12, 8)
        self.t_far = np.linspace(0.88, 1.0, 8)

        # Precomputed gate geometry
        self._gpos = np.array([g.position for g in self.gates])
        self._yaw = np.array([g.yaw for g in self.gates])
        self._cyaw = np.cos(self._yaw)
        self._syaw = np.sin(self._yaw)

        # Chord length to each gate's neighbours, used to scale the
        # tangent-length bounds.
        self._chord = np.array([
            np.linalg.norm(self._gpos[(i + 1) % self.n] - self._gpos[i])
            for i in range(self.n)])
        self.gam_max = np.radians(self.max_climb_deg)
        self.d_min = 0.15
        # Tangent-length cap. Scaling it off the adjoining chords alone
        # silently forbids a leg that loops the long way round to reach
        # a nearby gate — a short chord would starve the tangent of the
        # reach that loop needs. The arena floor below keeps that
        # topology available; the objective still prefers short
        # tangents wherever a direct link is the smoother option.
        arena_reach = 0.45 * min(self.arena.flight_box[0],
                                 self.arena.flight_box[1])
        forced = max_tangent is not None
        if forced:
            arena_reach = float(max_tangent)
        # The floor matters because tangent reach decides topology: a
        # short tangent can only link consecutive gates directly, so
        # scaling the cap off the adjoining chords alone makes a leg
        # that loops the long way round unrepresentable. Applying it
        # everywhere is safe now that C2 is measured geometrically —
        # under the old raw-C'' term it perturbed well-behaved tracks.
        self.d_max = np.zeros(self.n)
        for i in range(self.n):
            c = min(self._chord[i], self._chord[(i - 1) % self.n])
            d = max(0.6 * c, arena_reach)
            self.d_max[i] = max(d, self.d_min + 0.05)

    # ── packing helpers ──

    def _tangents(self, d, gam):
        """Per-gate tangent vectors T_i (shared by both sides)."""
        cg = np.cos(gam)
        return np.column_stack([d * cg * self._cyaw,
                                d * cg * self._syaw,
                                d * np.sin(gam)])

    def _unpack(self, vec):
        n = self.n
        return vec[:n], vec[n:2*n], vec[2*n:].reshape(n, 2, 3)

    def _pack(self, d, gam, inner):
        return np.concatenate([d, gam, np.asarray(inner).ravel()])

    def _build_segments(self, vec):
        n = self.n
        d, gam, inner = self._unpack(vec)
        T = self._tangents(d, gam)
        segs = []
        for i in range(n):
            j = (i + 1) % n
            g0 = self._gpos[i]
            g1 = self._gpos[j]
            segs.append(np.vstack([g0, g0 + T[i],
                                   inner[i, 0], inner[i, 1],
                                   g1 - T[j], g1]))
        return segs

    def _bounds(self):
        b = [(self.d_min, float(self.d_max[i])) for i in range(self.n)]
        b += [(-self.gam_max, self.gam_max)] * self.n
        for _ in range(self.n * 2):
            b += [(self.flo[0], self.fhi[0]),
                  (self.flo[1], self.fhi[1]),
                  (self.min_traj_z, self.fhi[2])]
        return b

    # ── cost ──

    def cost_function(self, vec):
        """Minimize ∫κ²ds + constraint penalties.

        C1 continuity and gate-normal alignment are structural (see
        the class docstring), so they carry no penalty term here.
        """
        n = self.n
        segs = self._build_segments(vec)
        te = self.te
        cost = 0.0
        pen = 0.0

        # Per-segment costs
        for si, P in enumerate(segs):
            k = curvature(te, P)
            pts = qb(te, P)
            ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            km = 0.5 * (k[:-1] + k[1:])

            # Curvature integral (primary objective)
            cost += np.sum(km**2 * ds)

            # Turning radius (5% safety margin)
            pen += 2000 * np.sum(np.maximum(k - self.mk * 0.95, 0)**2)

            # Climb rate (5% safety margin)
            cr = climb_rate(te, P)
            pen += 1000 * np.sum(
                np.maximum(cr - self.max_climb_rate * 0.95, 0)**2)

            # Bounding box
            for d in range(3):
                pen += 400 * (
                    np.sum(np.maximum(self.flo[d] - pts[:, d], 0)**2) +
                    np.sum(np.maximum(pts[:, d] - self.fhi[d], 0)**2))

            # ── Height-profile shaping ──
            # Nothing above constrains the *shape* of z: over a long
            # leg a half-metre hump is a very gentle bend, so the
            # curvature integral barely notices it and |dz/ds| stays
            # inside the climb limit throughout. Left alone the
            # optimizer ripples the height profile for free. These two
            # terms make each leg climb or descend exactly once.
            zz = pts[:, 2]
            z0 = self.gates[si].z
            z1 = self.gates[(si + 1) % n].z
            zlo, zhi = min(z0, z1), max(z0, z1)

            # (a) no overshoot outside the leg's own height band
            pen += self.w_z_band * np.sum(
                np.maximum(zz - zhi, 0)**2 + np.maximum(zlo - zz, 0)**2)

            # (b) climb once: charge for vertical travel beyond the
            #     leg's net height change (smooth |·| for the solver)
            dz = np.diff(zz)
            tv = np.sum(np.sqrt(dz * dz + 1e-10))
            pen += self.w_z_mono * max(
                tv - abs(zz[-1] - zz[0]), 0.0)**2

        # C2 continuity at gates, measured on the curvature vector
        # dT/ds rather than raw C''. C'' is parameterization-dependent
        # and grows with the tangent length d, so penalizing it makes
        # long-tangent solutions look catastrophically bad even when
        # their geometric curvature is perfectly continuous — which
        # is what previously drove the optimizer away from legs that
        # loop the long way round. dT/ds is in 1/m and scale-free.
        for i in range(n):
            pen += self.w_c2 * np.sum(
                (_curv_vec(segs[(i - 1) % n], 1.0) -
                 _curv_vec(segs[i], 0.0))**2)

        # Gate clearance — the curve must fit through the opening
        cl = self.gate_half_w - self.drone_margin
        cz = self.gate_half_h - self.drone_margin
        for si in range(n):
            P = segs[si]
            gs = self.gates[si]
            ge = self.gates[(si + 1) % n]

            perp_s = np.array([-gs.normal[1], gs.normal[0], 0.0])
            d = qb(self.t_near, P) - gs.position
            pen += 500 * np.sum(np.maximum(np.abs(d @ perp_s) - cl, 0)**2)
            pen += 500 * np.sum(np.maximum(np.abs(d[:, 2]) - cz, 0)**2)

            perp_e = np.array([-ge.normal[1], ge.normal[0], 0.0])
            d = qb(self.t_far, P) - ge.position
            pen += 500 * np.sum(np.maximum(np.abs(d @ perp_e) - cl, 0)**2)
            pen += 500 * np.sum(np.maximum(np.abs(d[:, 2]) - cz, 0)**2)

        return cost + pen

    # ── initial guess ──

    def _init_guess(self):
        """Initial tangent lengths, gate climb angles and inner CPs."""
        n = self.n

        # Tangent length: a fraction of the shorter adjoining chord.
        d0 = np.array([
            np.clip(0.3 * min(self._chord[i], self._chord[(i - 1) % n]),
                    self.d_min, self.d_max[i])
            for i in range(n)])

        # Gate climb angle: average of the incoming and outgoing
        # gate-to-gate slopes, clipped to the climb limit.
        gam0 = np.zeros(n)
        for i in range(n):
            p, nx = (i - 1) % n, (i + 1) % n
            din = self._gpos[i] - self._gpos[p]
            dout = self._gpos[nx] - self._gpos[i]
            s_in = din[2] / max(np.linalg.norm(din), 1e-9)
            s_out = dout[2] / max(np.linalg.norm(dout), 1e-9)
            gam0[i] = np.clip(np.arcsin(np.clip(0.5 * (s_in + s_out),
                                                -1, 1)),
                              -self.gam_max, self.gam_max)

        # Inner control points: push out along the tangents, then
        # bias toward the other gate.
        T = self._tangents(d0, gam0)
        inner = np.zeros((n, 2, 3))
        for i in range(n):
            j = (i + 1) % n
            P0, P5 = self._gpos[i], self._gpos[j]
            P2 = P0 + 1.8 * T[i] + 0.3 * (P5 - P0)
            P3 = P5 - 1.8 * T[j] + 0.3 * (P0 - P5)
            for k, (pt, a) in enumerate(((P2, 0.4), (P3, 0.6))):
                pt[2] = np.clip(P0[2] * (1 - a) + P5[2] * a,
                                self.min_traj_z, self.fhi[2])
                inner[i, k] = np.clip(pt, self.flo, self.fhi)
        return self._pack(d0, gam0, inner)

    def feasible(self, segments, n_check=300):
        """True when the analytic curve meets radius + climb limits."""
        t = np.linspace(0, 1, n_check)
        for P in segments:
            if 1.0 / (np.max(curvature(t, P)) + 1e-10) < self.min_turn_radius:
                return False
            if climb_deg(np.max(climb_rate(t, P))) > self.max_climb_deg:
                return False
        return True

    def junction_report(self, segments):
        """Turn angle at each gate, in degrees.

        With the structural parameterization these are ~0 by
        construction; a non-zero value means something is wrong.
        """
        n = self.n
        out = []
        for i in range(n):
            ti = qb1(np.array([1.0]), segments[(i - 1) % n])[0]
            to = qb1(np.array([0.0]), segments[i])[0]
            ti /= max(np.linalg.norm(ti), 1e-12)
            to /= max(np.linalg.norm(to), 1e-12)
            out.append(float(np.degrees(
                np.arccos(np.clip(np.dot(ti, to), -1.0, 1.0)))))
        return out

    # ── driver ──

    def optimize(self, de_maxiter=60, de_popsize=15, lbfgs_maxiter=800,
                 seed=42, skip_de=False, auto_de=True, verbose=True):
        """Run the optimization, escalating to DE only if needed.

        Policy (`auto_de=True`, the default): run the cheap
        L-BFGS-B polish from the analytic initial guess first. If
        the result already satisfies the radius and climb limits,
        stop — that is the common case and takes a few seconds. If
        it does not, escalate to a full DE global search followed
        by a second polish, and keep whichever result is better
        (feasible first, then lower cost).

        `skip_de=True` never escalates; `auto_de=False` always runs
        DE.

        Parameters
        ----------
        de_maxiter, de_popsize : int
            Differential-evolution budget.
        lbfgs_maxiter : int
            L-BFGS-B iteration cap.
        seed : int
            DE random seed. DE is the only stochastic part, so a
            fixed seed makes the whole pipeline deterministic.

        Returns
        -------
        segments : list of (6,3) ndarray
        result : scipy OptimizeResult
        """
        x0 = self._init_guess()
        bounds = self._bounds()
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        init_cost = self.cost_function(x0)

        if verbose:
            print(f"Gates ({self.n}):")
            for i, g in enumerate(self.gates):
                print(f"  G{i+1}: {g}")
            print(f"\nConstraints: R ≥ {self.min_turn_radius}m, "
                  f"climb ≤ {self.max_climb_deg}°")
            print(f"Gate opening: {self.drone.gate_width*100:.0f}×"
                  f"{self.drone.gate_height*100:.0f}cm, "
                  f"drone margin {self.drone_margin*100:.0f}cm")
            print(f"Flight box: {tuple(np.round(self.flo, 2))} → "
                  f"{tuple(np.round(self.fhi, 2))}")
            print(f"Variables: {len(x0)} "
                  f"({self.n}×(d, γ) + {self.n} segs × 2 CPs × 3)")
            print("\nGate spacing (XY):")
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    d = np.linalg.norm(self.gates[i].position[:2] -
                                       self.gates[j].position[:2])
                    tag = "OK" if d >= 2 * self.min_turn_radius else "CLOSE"
                    print(f"  G{i+1}–G{j+1}: {d:.2f}m [{tag}]")
            print(f"\nInit cost: {init_cost:.0f}")

        def polish(start, label):
            if verbose:
                print(f"\n— {label}: L-BFGS-B ({lbfgs_maxiter} iter) —")
            r = minimize(self.cost_function, start,
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': lbfgs_maxiter,
                                  'ftol': 1e-10})
            ok = self.feasible(self._build_segments(r.x))
            if verbose:
                print(f"  cost: {r.fun:.4f}   constraints: "
                      f"{'satisfied ✓' if ok else 'violated ✗'}")
            return r, ok

        best, best_ok = polish(x0, "Local refinement")

        run_de = (not skip_de) and (not best_ok or not auto_de)
        if run_de:
            if verbose:
                why = ("constraints violated" if not best_ok
                       else "auto-DE disabled")
                print(f"\n— Global search: Differential Evolution "
                      f"({de_maxiter} gen, pop {de_popsize}) — "
                      f"[{why}]")
            de = differential_evolution(
                self.cost_function, bounds,
                maxiter=de_maxiter, seed=seed, tol=1e-6,
                popsize=de_popsize, mutation=(0.5, 1.5),
                recombination=0.8, disp=False, x0=x0)
            if verbose:
                print(f"  DE cost: {de.fun:.4f}")
            cand, cand_ok = polish(de.x, "Post-DE refinement")
            for r, ok in ((de, self.feasible(self._build_segments(de.x))),
                          (cand, cand_ok)):
                # Feasibility wins; cost breaks ties.
                if (ok, -r.fun) > (best_ok, -best.fun):
                    best, best_ok = r, ok
            if verbose:
                print(f"\n  Kept: cost {best.fun:.4f}, constraints "
                      f"{'satisfied ✓' if best_ok else 'violated ✗'}")
        elif verbose and not skip_de:
            print("  Constraints already satisfied → skipping DE")

        return self._build_segments(best.x), best

    # ── reporting on the analytic Bezier (pre Z-scaling) ──

    def analyze(self, segments, label="", verbose=True):
        """Analyze the analytic Bezier segments.

        Returns (total_length, all_ok).
        """
        t = np.linspace(0, 1, 300)
        n = self.n
        total = 0.0
        all_ok = True
        rows = []

        for i, P in enumerate(segments):
            k = curvature(t, P)
            cr = climb_rate(t, P)
            L = arc_length(P)
            total += L
            mr = 1.0 / (np.max(k) + 1e-10)
            pts = qb(t, P)
            ma = climb_deg(np.max(cr))
            r_ok = mr >= self.min_turn_radius
            c_ok = ma <= self.max_climb_deg
            all_ok = all_ok and r_ok and c_ok
            rows.append((i, L, mr, ma, pts[:, 2].min(), pts[:, 2].max(),
                         r_ok, c_ok))

        # Kink check — a per-segment sweep cannot see a discontinuity
        # at the joins, which is exactly where cusps would appear.
        turns = self.junction_report(segments)
        if max(turns) > 1.0:
            all_ok = False

        if verbose:
            print(f"\n{'='*62}")
            print(f"  {label or 'Trajectory'}"
                  f"   [R ≥ {self.min_turn_radius}m, "
                  f"climb ≤ {self.max_climb_deg}°]")
            print(f"{'='*62}")
            for (i, L, mr, ma, zlo, zhi, r_ok, c_ok) in rows:
                j = (i + 1) % n
                print(f"  S{i+1} (G{i+1}→G{j+1}): L={L:6.2f}m  "
                      f"R={mr:5.2f}m {'✓' if r_ok else '✗ FAIL'}  "
                      f"climb={ma:5.1f}° {'✓' if c_ok else '✗ FAIL'}  "
                      f"Z=[{zlo:.2f}, {zhi:.2f}]")
            print("\n  Gate junctions (turn angle, 0° = smooth):")
            print("    " + "  ".join(
                f"G{i+1}={a:.2f}°{'' if a <= 1.0 else ' KINK'}"
                for i, a in enumerate(turns)))
            print(f"\n  Loop: {total:.2f}m  |  "
                  f"lap ≈ {total/self.drone.max_speed:.1f}s @ "
                  f"{self.drone.max_speed}m/s  |  "
                  f"{'ALL OK ✓' if all_ok else 'VIOLATIONS ✗'}")

        if verbose:
            self._report_clearance(segments)

        return total, all_ok

    def _report_clearance(self, segments):
        n = self.n
        t_near = np.linspace(0.0, 0.12, 10)
        t_far = np.linspace(0.88, 1.0, 10)
        cl = self.gate_half_w - self.drone_margin
        cz = self.gate_half_h - self.drone_margin
        print(f"\n  Gate clearance (budget: lat < {cl*100:.0f}cm, "
              f"z < {cz*100:.0f}cm):")
        for si in range(n):
            P = segments[si]
            for label, gate, tt in (
                    (f"entry G{si+1}", self.gates[si], t_near),
                    (f"exit  G{(si+1)%n+1}", self.gates[(si+1) % n], t_far)):
                perp = np.array([-gate.normal[1], gate.normal[0], 0.0])
                d = qb(tt, P) - gate.position
                lat = np.max(np.abs(d @ perp))
                zd = np.max(np.abs(d[:, 2]))
                ok = "OK" if (lat < cl and zd < cz) else "CLIP"
                print(f"    S{si+1} {label}: lat={lat*100:5.1f}cm  "
                      f"z={zd*100:5.1f}cm  [{ok}]")

    def export_gates_json(self, path, gates=None):
        gates = self.gates if gates is None else gates
        gd = [{"gate": i + 1,
               "pos": [round(g.x, 3), round(g.y, 3), round(g.z_out, 3)],
               "yaw_deg": round(g.yaw_deg, 1),
               "z_optimized": round(g.z, 3)}
              for i, g in enumerate(gates)]
        with open(path, 'w') as f:
            json.dump(gd, f, indent=2)


# ═══════════════════════════════════════════════════════════════
#  POST-PROCESSING: SAMPLE, REORDER, SCALE Z, EXPORT
# ═══════════════════════════════════════════════════════════════

def sample_segments(segments, n_pts_per_seg=150):
    """Sample every Bezier segment into a list of row dicts."""
    t = np.linspace(0, 1, n_pts_per_seg)
    rows = []
    for si, P in enumerate(segments):
        pts = qb(t, P)
        k = curvature(t, P)
        cr = climb_rate(t, P)
        for j in range(len(t)):
            rows.append({
                'seg': si,
                't': float(t[j]),
                'x': float(pts[j, 0]),
                'y': float(pts[j, 1]),
                'z': float(pts[j, 2]),
                'curvature': float(k[j]),
                'climb_rate': float(cr[j]),
                'climb_angle_deg': float(climb_deg(cr[j])),
            })
    return rows


def reorder_from_gate(rows, gates, start_gate_idx):
    """Rotate the loop so it starts at `start_gate_idx`.

    Original order is G0→G1→G2→…; with start_gate_idx=1 the result
    is G1→G2→G0→…. Returns (rows, gates) both reordered.
    """
    n = len(gates)
    start_gate_idx %= n
    out = []
    for new_seg in range(n):
        old_seg = (start_gate_idx + new_seg) % n
        for r in rows:
            if r['seg'] == old_seg:
                rc = dict(r)
                rc['seg'] = new_seg
                out.append(rc)
    gates_out = [gates[(start_gate_idx + i) % n] for i in range(n)]
    return out, gates_out


def fit_z_map(gates):
    """Least-squares affine map z → a·z + b hitting the z_final's.

    A *single global* affine map is used rather than a per-segment
    remap. Rescaling one coordinate by an affine function is a
    linear transform of the curve, so it preserves C1/C2 continuity
    exactly — the trajectory stays as smooth as it was optimized.
    A per-segment remap does not: neighbouring segments get
    different scale factors, which puts a slope discontinuity in Z
    at every gate.

    Returns (a, b, max_residual). The residual is 0 when the
    (z, z_final) pairs are collinear, which covers the usual case
    of shifting and/or compressing the whole height band.
    """
    z = np.array([g.z for g in gates], dtype=float)
    zf = np.array([g.z_out for g in gates], dtype=float)

    if np.ptp(z) < 1e-9:                    # all gates at one height
        a, b = 1.0, float(np.mean(zf - z))
    else:
        a, b = np.polyfit(z, zf, 1)
    resid = float(np.max(np.abs(a * z + b - zf))) if len(z) else 0.0
    return float(a), float(b), resid


def scale_z(rows, a, b):
    """Apply the affine height map z → a·z + b to every sample."""
    for row in rows:
        row['z'] = a * row['z'] + b
    return rows


def recompute_derived(rows):
    """Recompute curvature / climb from the sampled 3D points.

    Necessary after Z rescaling: the analytic Bezier values no
    longer describe the exported geometry. Derivatives are taken
    with respect to arc length on the closed loop, so the result is
    parameterization-independent and wraps correctly.
    """
    pts = np.array([[r['x'], r['y'], r['z']] for r in rows], dtype=float)
    n = len(pts)

    # Collapse consecutive duplicates (gate points are shared by the
    # end of one segment and the start of the next).
    uniq = []
    src = np.zeros(n, dtype=int)
    for i in range(n):
        if uniq and np.linalg.norm(pts[i] - uniq[-1]) <= 1e-9:
            src[i] = len(uniq) - 1
        else:
            uniq.append(pts[i])
            src[i] = len(uniq) - 1
    u = np.array(uniq)
    # Close the loop: if the last unique point repeats the first,
    # fold it back onto index 0.
    if len(u) > 1 and np.linalg.norm(u[-1] - u[0]) <= 1e-9:
        src[src == len(u) - 1] = 0
        u = u[:-1]
    m = len(u)
    if m < 5:
        return rows

    # Tile 3× so the finite differences see a periodic
    # neighbourhood, then keep the middle copy.
    tiled = np.vstack([u, u, u])

    # Climb rate from a single arc-length gradient (second-order
    # accurate, and stable on the non-uniform spacing produced by
    # sampling each Bezier uniformly in t).
    seg = np.maximum(np.linalg.norm(np.diff(tiled, axis=0), axis=1), 1e-12)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    r1 = np.gradient(tiled, s, axis=0)
    nr1 = np.maximum(np.linalg.norm(r1, axis=1), 1e-12)
    cr = (np.abs(r1[:, 2]) / nr1)[m:2*m]

    # Curvature from the circumcircle through each consecutive
    # triple (Menger curvature). Nesting np.gradient twice would
    # spike wherever the sample spacing jumps — at every segment
    # junction — whereas the circumcircle is exact for any spacing.
    p0, p1, p2 = tiled[:-2], tiled[1:-1], tiled[2:]
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    area2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    kap_inner = 2.0 * area2 / np.maximum(a * b * c, 1e-12)
    kap = kap_inner[m-1:2*m-1]

    for i, r in enumerate(rows):
        j = src[i]
        r['curvature'] = float(kap[j])
        r['climb_rate'] = float(cr[j])
        r['climb_angle_deg'] = float(climb_deg(cr[j]))
    return rows


def analyze_rows(rows, gates, drone, label="", verbose=True):
    """Validate the *exported* sampled trajectory.

    Uses the recomputed per-point curvature/climb, so this reflects
    the geometry actually written to the CSV (post Z-scaling).
    Returns (total_length, all_ok).
    """
    n = len(gates)
    pts = np.array([[r['x'], r['y'], r['z']] for r in rows])
    segs = np.array([r['seg'] for r in rows])
    kap = np.array([r['curvature'] for r in rows])
    cr = np.array([r['climb_rate'] for r in rows])

    closed = np.vstack([pts, pts[:1]])
    total = float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))

    all_ok = True
    out = []
    for si in range(n):
        msk = segs == si
        p = pts[msk]
        L = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        mr = 1.0 / (np.max(kap[msk]) + 1e-10)
        ma = float(climb_deg(np.max(cr[msk])))
        r_ok = mr >= drone.min_turn_radius
        c_ok = ma <= drone.max_climb_deg
        all_ok = all_ok and r_ok and c_ok
        out.append((si, L, mr, ma, p[:, 2].min(), p[:, 2].max(),
                    r_ok, c_ok))

    # Kink check on the exported polyline: compare the chord
    # arriving at each gate with the one leaving it.
    def chord_angle(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.degrees(np.arccos(
            np.clip(np.dot(a / na, b / nb), -1.0, 1.0))))

    turns = []
    for si in range(n):
        idx = np.flatnonzero(segs == si)
        prev = np.flatnonzero(segs == (si - 1) % n)
        turns.append(chord_angle(pts[idx[0]] - pts[prev[-2]],
                                 pts[idx[1]] - pts[idx[0]]))

    # A smooth curve sampled at spacing ds already turns by about
    # κ·ds per step, so the bar has to scale with sample density —
    # a fixed threshold would cry kink on coarsely sampled tracks.
    interior = [chord_angle(pts[i] - pts[i-1], pts[i+1] - pts[i])
                for i in range(1, len(pts) - 1)
                if segs[i-1] == segs[i] == segs[i+1]]
    kink_tol = max(5.0, 4.0 * float(np.median(interior))) if interior \
        else 5.0
    if max(turns) > kink_tol:
        all_ok = False

    if verbose:
        print(f"\n{'='*62}")
        print(f"  {label or 'Exported trajectory'}"
              f"   [R ≥ {drone.min_turn_radius}m, "
              f"climb ≤ {drone.max_climb_deg}°]")
        print(f"{'='*62}")
        print("  Gates:")
        for i, g in enumerate(gates):
            print(f"    G{i+1}: ({g.x:+.3f}, {g.y:+.3f}, "
                  f"z={g.z_out:.3f}) yaw={g.yaw_deg:+.1f}°")
        print()
        for (i, L, mr, ma, zlo, zhi, r_ok, c_ok) in out:
            j = (i + 1) % n
            print(f"  S{i+1} (G{i+1}→G{j+1}): L={L:6.2f}m  "
                  f"R={mr:5.2f}m {'✓' if r_ok else '✗ FAIL'}  "
                  f"climb={ma:5.1f}° {'✓' if c_ok else '✗ FAIL'}  "
                  f"Z=[{zlo:.2f}, {zhi:.2f}]")
        print(f"\n  Gate junctions (chord turn, tol {kink_tol:.2f}°):")
        print("    " + "  ".join(
            f"G{i+1}={a:.2f}°{'' if a <= kink_tol else ' KINK'}"
            for i, a in enumerate(turns)))
        print(f"\n  Loop: {total:.2f}m  |  "
              f"lap ≈ {total/drone.max_speed:.1f}s @ "
              f"{drone.max_speed}m/s  |  "
              f"{'ALL OK ✓' if all_ok else 'VIOLATIONS ✗'}")

    return total, all_ok


def export_csv(rows, path):
    """Write the trajectory CSV consumed by external_trajectory.py."""
    fieldnames = ['seg', 't', 'x', 'y', 'z', 'curvature',
                  'climb_rate', 'climb_angle_deg']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def plot_trajectory(rows, gates, path, drone):
    """Save a 3-panel PNG (3D path, top view, curvature/climb)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib unavailable — skipping plot)")
        return False

    # Some installs (e.g. a system matplotlib shadowed by a pip one)
    # ship without the 3D toolkit; fall back to a side view.
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        has3d = True
    except Exception:
        has3d = False

    pts = np.array([[r['x'], r['y'], r['z']] for r in rows])
    kap = np.array([r['curvature'] for r in rows])
    ang = np.array([r['climb_angle_deg'] for r in rows])
    gp = np.array([[g.x, g.y, g.z_out] for g in gates])
    closed = np.vstack([pts, pts[:1]])
    s = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(pts, axis=0), axis=1))])

    fig = plt.figure(figsize=(15, 5))

    if has3d:
        ax = fig.add_subplot(131, projection='3d')
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], lw=1.6)
        ax.scatter(gp[:, 0], gp[:, 1], gp[:, 2], c='r', s=50,
                   depthshade=False)
        for i, g in enumerate(gp):
            ax.text(g[0], g[1], g[2], f" G{i+1}", fontsize=9)
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.set_title('3D path')
    else:
        ax = fig.add_subplot(131)
        ax.plot(s, pts[:, 2], lw=1.6)
        for i in range(len(gates)):
            j = np.flatnonzero(np.array([r['seg'] for r in rows]) == i)[0]
            ax.axvline(s[j], ls=':', c='r', lw=1)
            ax.text(s[j], gp[i, 2], f" G{i+1}", fontsize=9, color='r')
        ax.set_xlabel('arc length [m]'); ax.set_ylabel('z [m]')
        ax.grid(alpha=0.3)
        ax.set_title('height profile')

    ax = fig.add_subplot(132)
    ax.plot(closed[:, 0], closed[:, 1], lw=1.6)
    ax.scatter(gp[:, 0], gp[:, 1], c='r', s=50, zorder=5)
    for i, g in enumerate(gates):
        ax.arrow(g.x, g.y, 0.5*np.cos(g.yaw), 0.5*np.sin(g.yaw),
                 head_width=0.12, color='r', zorder=5)
        ax.text(g.x, g.y, f"  G{i+1}", fontsize=9)
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('top view')

    ax = fig.add_subplot(133)
    ax.plot(s, 1.0 / np.maximum(kap, 1e-6), label='turn radius [m]')
    ax.axhline(drone.min_turn_radius, ls='--', c='r', lw=1,
               label=f'R min = {drone.min_turn_radius}m')
    ax.set_ylim(0, max(6.0, drone.min_turn_radius * 3))
    ax.set_xlabel('arc length [m]'); ax.set_ylabel('radius [m]')
    ax2 = ax.twinx()
    ax2.plot(s, ang, c='g', alpha=0.6, label='climb [°]')
    ax2.axhline(drone.max_climb_deg, ls='--', c='g', lw=1)
    ax2.set_ylabel('climb angle [°]', color='g')
    ax.grid(alpha=0.3); ax.legend(loc='lower right', fontsize=8)
    ax.set_title('constraints along the loop')

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


# ═══════════════════════════════════════════════════════════════
#  PRESET TRACKS
# ═══════════════════════════════════════════════════════════════

def PRESETS():
    """Built-in gate layouts (fresh Gate objects on every call).

    Every preset ships with flow-consistent yaws and is verified to
    satisfy R ≥ 1.8m and climb ≤ 20° at the default settings.
    """
    return {
        # 3 gates, equilateral triangle, mild height variation.
        'triangle': [
            Gate(0.0, 3.0, 0.325, 180.0),
            Gate(-2.8, -1.8, 1.00, -60.1),
            Gate(2.8, -1.8, 0.60, 60.1),
        ],
        # 3 gates, one high gate in the middle — the drone swings
        # up and over like a pendulum.
        'pendulum': [
            Gate(3.0, 1.5, 0.325, -69.7),
            Gate(-0.5, -1.5, 1.25, 175.2),
            Gate(-3.0, 1.5, 0.325, 64.9),
        ],
        # 3 gates, asymmetric circuit with an elevated back corner.
        'racetrack': [
            Gate(0.0, 3.0, 0.325, 0.0),
            Gate(1.5, -1.5, 0.325, -114.2),
            Gate(-2.0, -3.0, 1.00, 137.4),
        ],
        # 4 gates, alternating heights. Good baseline.
        'oval': [
            Gate(0.0, 3.0, 0.325, 180.0),
            Gate(-3.0, 0.0, 0.90, -90.0),
            Gate(0.0, -3.0, 0.325, 0.0),
            Gate(3.0, 0.0, 0.90, 90.0),
        ],
        # 5 gates on a pentagon, alternating heights — the drone
        # rides a height wave around the loop.
        'wave': [
            Gate(0.0, 3.0, 0.35, 180.0),
            Gate(-2.9, 0.9, 1.10, -108.1),
            Gate(-1.8, -2.5, 0.35, -36.0),
            Gate(1.8, -2.5, 1.10, 36.0),
            Gate(2.9, 0.9, 0.35, 108.1),
        ],
        # 6 gates, the longest loop.
        'hex': [
            Gate(0.0, 3.2, 0.35, 180.0),
            Gate(-2.8, 1.6, 1.00, -120.1),
            Gate(-2.8, -1.6, 0.35, -59.9),
            Gate(0.0, -3.2, 1.00, 0.0),
            Gate(2.8, -1.6, 0.35, 59.9),
            Gate(2.8, 1.6, 1.00, 120.1),
        ],
        # Demonstrates z_final: the shape is optimized with a
        # generous 0.35–1.60m height spread, then compressed onto
        # 0.50–1.10m gates. Compressing Z lowers the climb angle,
        # so feasibility is preserved.
        'zscale': [
            Gate(3.0, 1.5, 0.35, -69.7, z_final=0.50),
            Gate(-0.5, -1.5, 1.60, 175.2, z_final=1.10),
            Gate(-3.0, 1.5, 0.35, 64.9, z_final=0.50),
        ],
    }


def auto_yaw(gates):
    """Overwrite each gate yaw with the local flow direction.

    Gate i is aimed along the bisector of the incoming and outgoing
    chords, which is the heading a smooth loop through the gates
    actually wants. Handy when you know where the gates go but not
    which way they should face.

    Caveat: if the two chords are exactly opposite — collinear
    gates, i.e. the loop doubles back — the bisector is undefined
    and the outgoing chord is used instead. Such layouts usually
    need a hand-picked yaw.
    """
    n = len(gates)
    P = np.array([g.position for g in gates])
    for i in range(n):
        a = P[i] - P[(i - 1) % n]
        b = P[(i + 1) % n] - P[i]
        a = a[:2] / max(np.linalg.norm(a[:2]), 1e-9)
        b = b[:2] / max(np.linalg.norm(b[:2]), 1e-9)
        d = a + b
        if np.linalg.norm(d) < 1e-3:      # degenerate reversal
            d = b
        gates[i].yaw_deg = float(np.degrees(np.arctan2(d[1], d[0])))
    return gates


def load_gates(path):
    """Load gates from JSON: [{"pos":[x,y,z], "yaw_deg":d, "z_final":z}]"""
    with open(path) as f:
        data = json.load(f)
    return [Gate(g['pos'][0], g['pos'][1], g['pos'][2],
                 g['yaw_deg'], g.get('z_final')) for g in data]


# ═══════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════

def generate(gates, drone=None, arena=None, n_pts_per_seg=150,
             de_maxiter=60, de_popsize=15, lbfgs_maxiter=800, seed=42,
             skip_de=False, auto_de=True, reorder=True, start_gate=None,
             apply_zscale=True, verbose=True):
    """Run the full pipeline; returns (rows, gates_ordered, opt, ok)."""
    drone = drone if drone is not None else DroneConfig()
    arena = arena if arena is not None else ArenaConfig()

    opt = TrajectoryOptimizer(gates, drone=drone, arena=arena)
    segments, _ = opt.optimize(
        de_maxiter=de_maxiter, de_popsize=de_popsize,
        lbfgs_maxiter=lbfgs_maxiter, seed=seed,
        skip_de=skip_de, auto_de=auto_de, verbose=verbose)

    _, ok_bezier = opt.analyze(segments, "Optimized (original Z)",
                               verbose=verbose)

    rows = sample_segments(segments, n_pts_per_seg=n_pts_per_seg)
    gates_ordered = list(gates)

    # Reorder so the loop begins at the lowest final gate
    if reorder:
        if start_gate is None:
            idx = int(np.argmin([g.z_out for g in gates_ordered]))
        else:
            idx = int(start_gate)
        if idx != 0:
            rows, gates_ordered = reorder_from_gate(rows, gates_ordered, idx)
            if verbose:
                print(f"\n  Reordered: loop starts at G{idx+1} "
                      f"(z={gates_ordered[0].z_out:.2f}m)")
        elif verbose:
            print(f"\n  Loop already starts at the lowest gate G1 "
                  f"(z={gates_ordered[0].z_out:.2f}m)")

    # Rescale Z to the requested final heights
    scaled = False
    if apply_zscale and any(g.z_final is not None for g in gates_ordered):
        a, b, resid = fit_z_map(gates_ordered)
        rows = scale_z(rows, a, b)
        scaled = True
        if verbose:
            print(f"  Z rescaled: z → {a:.4f}·z + {b:+.4f}"
                  f"   (max gate error {resid*100:.1f}cm)")
        if resid > 0.01 and verbose:
            print("    WARNING: the requested z_final heights are not "
                  "an affine function of the optimized heights, so "
                  "they cannot all be hit by a smoothness-preserving "
                  "map. Optimize directly at the final heights "
                  "instead (drop z_final).")
        zmin = min(r['z'] for r in rows)
        if zmin < arena.min_traj_z and verbose:
            print(f"    WARNING: rescaled trajectory dips to "
                  f"{zmin:.2f}m, below the {arena.min_traj_z:.2f}m "
                  f"ground clearance.")

    # The exported geometry differs from the analytic Bezier once Z
    # has been rescaled — recompute so the CSV columns match it.
    rows = recompute_derived(rows)

    label = ("Exported (Z rescaled)" if scaled
             else "Exported (resampled)")
    _, ok = analyze_rows(rows, gates_ordered, drone, label, verbose=verbose)

    if verbose and scaled and ok_bezier and not ok:
        print("\n  WARNING: Z rescaling introduced constraint "
              "violations. Lower the z_final spread, raise "
              "--min-radius headroom, or optimize directly at the "
              "final heights (drop z_final).")

    return rows, gates_ordered, opt, ok


def main():
    p = argparse.ArgumentParser(
        description="Generate an optimized racing trajectory for the "
                    "XFly MPCC controller.")
    src = p.add_argument_group("track definition")
    src.add_argument("--track", type=str, default="",
                     choices=sorted(PRESETS().keys()),
                     help="use a preset gate layout")
    src.add_argument("--gates", type=str, default="",
                     help="gate JSON file (overrides --track)")
    src.add_argument("--auto-yaw", action='store_true',
                     help="ignore the given yaws and aim each gate "
                          "along the local flow direction")

    out = p.add_argument_group("output")
    out.add_argument("--output", type=str, default="trajectory.csv",
                     help="output CSV path (default: trajectory.csv)")
    out.add_argument("--n-pts", type=int, default=150,
                     help="samples per segment (default: 150)")
    out.add_argument("--plot", type=str, nargs='?', const='auto',
                     default="", help="save a PNG preview "
                     "(path optional; defaults next to the CSV)")
    out.add_argument("--quiet", action='store_true',
                     help="suppress progress output")

    dr = p.add_argument_group("drone / arena constraints")
    dr.add_argument("--min-radius", type=float, default=1.8,
                    help="minimum turn radius in m (default: 1.8)")
    dr.add_argument("--max-climb", type=float, default=20.0,
                    help="maximum climb angle in deg (default: 20)")
    dr.add_argument("--max-speed", type=float, default=4.0,
                    help="speed used for the lap-time estimate "
                         "(default: 4.0)")
    dr.add_argument("--gate-width", type=float, default=0.80,
                    help="gate width in m (default: 0.80)")
    dr.add_argument("--gate-height", type=float, default=0.60,
                    help="gate height in m (default: 0.60)")
    dr.add_argument("--drone-margin", type=float, default=0.10,
                    help="drone half-span margin in m (default: 0.10)")
    dr.add_argument("--flight-box", type=float, nargs=3,
                    default=[7.5, 7.5, 4.0], metavar=('X', 'Y', 'Z'),
                    help="flight volume size (default: 7.5 7.5 4.0)")
    dr.add_argument("--flight-center", type=float, nargs=3,
                    default=[0.0, 0.0, 1.5], metavar=('X', 'Y', 'Z'),
                    help="flight volume center (default: 0 0 1.5)")
    dr.add_argument("--min-traj-z", type=float, default=0.15,
                    help="ground clearance in m (default: 0.15)")

    op = p.add_argument_group("optimizer")
    op.add_argument("--de-iter", type=int, default=60,
                    help="DE max generations (default: 60)")
    op.add_argument("--de-pop", type=int, default=15,
                    help="DE population size (default: 15)")
    op.add_argument("--lbfgs-iter", type=int, default=800,
                    help="L-BFGS-B max iterations (default: 800)")
    op.add_argument("--seed", type=int, default=42,
                    help="DE random seed (default: 42)")
    op.add_argument("--no-de", action='store_true',
                    help="skip DE, L-BFGS-B only")
    op.add_argument("--force-de", action='store_true',
                    help="always run DE (disable the auto-skip)")

    po = p.add_argument_group("post-processing")
    po.add_argument("--no-reorder", action='store_true',
                    help="keep the gate order as given (do not rotate "
                         "the loop to start at the lowest gate)")
    po.add_argument("--start-gate", type=int, default=None,
                    help="rotate the loop to start at this gate index "
                         "(0-based); overrides the lowest-gate rule")
    po.add_argument("--no-zscale", action='store_true',
                    help="ignore z_final and keep the optimized heights")

    args = p.parse_args()
    verbose = not args.quiet

    if args.gates:
        gates = load_gates(args.gates)
        name = os.path.splitext(os.path.basename(args.gates))[0]
    elif args.track:
        gates = PRESETS()[args.track]
        name = args.track
    else:
        gates = PRESETS()['pendulum']
        name = 'pendulum'

    if args.auto_yaw:
        auto_yaw(gates)

    drone = DroneConfig(min_turn_radius=args.min_radius,
                        max_climb_deg=args.max_climb,
                        max_speed=args.max_speed,
                        gate_width=args.gate_width,
                        gate_height=args.gate_height,
                        drone_margin=args.drone_margin)
    arena = ArenaConfig(flight_box=tuple(args.flight_box),
                        center=tuple(args.flight_center),
                        min_traj_z=args.min_traj_z)

    if verbose:
        print("╔════════════════════════════════════════════════╗")
        print("║  XFly racing trajectory generator              ║")
        print("╚════════════════════════════════════════════════╝")
        print(f"  Track: {name}\n")

    rows, gates_ordered, opt, ok = generate(
        gates, drone=drone, arena=arena,
        n_pts_per_seg=args.n_pts,
        de_maxiter=args.de_iter, de_popsize=args.de_pop,
        lbfgs_maxiter=args.lbfgs_iter, seed=args.seed,
        skip_de=args.no_de, auto_de=not args.force_de,
        reorder=not args.no_reorder, start_gate=args.start_gate,
        apply_zscale=not args.no_zscale, verbose=verbose)

    export_csv(rows, args.output)
    gates_path = os.path.splitext(args.output)[0] + '_gates.json'
    opt.export_gates_json(gates_path, gates_ordered)

    plot_path = ""
    if args.plot:
        plot_path = (os.path.splitext(args.output)[0] + '.png'
                     if args.plot == 'auto' else args.plot)
        if not plot_trajectory(rows, gates_ordered, plot_path, drone):
            plot_path = ""

    if verbose:
        print(f"\n  CSV:   {args.output}  ({len(rows)} points)")
        print(f"  Gates: {gates_path}")
        if plot_path:
            print(f"  Plot:  {plot_path}")
        print(f"\n  Done {'✓' if ok else '(with violations ✗)'}")
        print(f"\n  Use it with:\n"
              f"    python mpcc_node.py --sim --trajectory external \\\n"
              f"        --trajectory-csv {args.output} --n-loops 3")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
