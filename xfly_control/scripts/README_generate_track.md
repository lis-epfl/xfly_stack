# generate_track.py — Racing Trajectory Generator

Turns a set of gates into a smooth, constraint-satisfying racing
trajectory CSV for the XFly MPCC controller.

Supersedes the two earlier scripts it was merged from:
`optimize_track.py` (optimizer core) and `generate_trajectory.py`
(reordering + Z rescaling). See [What changed](#what-changed-vs-the-two-original-scripts)
for the behaviour differences — several are corrections, not just
additions.

## Relation to the publication

This generator is a refinement of the method described in Appendix B of
the accompanying publication. The formulation is the same in outline —
one quintic Bézier per gate-to-gate leg, minimizing the curvature
integral subject to turn-radius, climb-angle and flight-volume limits —
but differs in five respects:

1. **C¹ continuity and gate traversal are structural rather than
   penalized.** Appendix B leaves all four interior control points of
   every leg free (12*n* unknowns) and enforces both properties through
   penalty terms. Here each gate carries a single tangent vector shared
   by the legs on either side, so the trajectory is C¹ by construction
   and always crosses a gate along its normal. The decision vector
   reduces to 8*n*.
2. **C² continuity is measured on the curvature vector** d**T**/d*s*
   rather than on **C**″. The latter depends on the parameterization
   and grows with the tangent length, which penalizes long legs for
   their parameterization rather than their geometry.
3. **The climb limit is applied as** |d*z*/d*s*| ≤ sin γ_max. The
   quantity |ż|/‖**C**′‖ is the sine of the climb angle, so the
   comparison against tan γ_max in Appendix B admits 21.35° at a 20°
   setting.
4. **The height profile of each leg is constrained.** Nothing in the
   published cost function acts on the shape of *z* between gates, and
   the curvature integral is nearly indifferent to it over a long leg.
   Two terms confine *z* to the band spanned by the leg's gate heights
   and charge for vertical travel beyond the net height change, so each
   leg climbs or descends once.
5. **The exported trajectory is verified.** Curvature and climb are
   recomputed from the sampled geometry after resampling and any height
   rescaling, and the tool exits non-zero if a constraint is violated.

Trajectories produced by either version are consumed identically by
`external_trajectory.py`; `track_1.csv` and `track_2.csv` are the
originals used in the publication and are unchanged.

## Quick start

```bash
pip install numpy scipy            # matplotlib only for --plot

python3 generate_track.py --track oval                    # preset
python3 generate_track.py --gates my_gates.json --auto-yaw
python3 generate_track.py --track wave --output wave.csv --plot
```

Then feed the CSV to the controller:

```bash
python3 mpcc_node.py --sim --trajectory external \
    --trajectory-csv wave.csv --n-loops 3
```

Exit status is `0` when the exported trajectory meets every
constraint and `1` otherwise, so it can gate a build step.

## What it does

Each gate-to-gate leg is a quintic Bézier. The optimizer minimizes
the curvature integral ∫κ²ds subject to:

| Constraint | Default | How it is enforced |
|---|---|---|
| Turning radius ≥ | 1.8 m | penalty (2000×) |
| Climb angle ≤ | 20° | penalty (1000×) + hard bound at gates |
| Flight volume | 7.5 × 7.5 × 4 m | penalty (400×) |
| Gate clearance | 80 × 60 cm, 10 cm margin | penalty (500×) |
| C2 continuity | — | penalty (150×) on the **curvature vector** |
| Height stays in each leg's band | — | penalty (100×) |
| Each leg climbs/descends once | — | penalty (400×) |
| **C1 continuity** | — | **structural — see below** |
| **Gate heading** | gate yaw | **structural — see below** |

### Height-profile shaping

Nothing in the curvature objective constrains the *shape* of z. Over
a long leg a half-metre hump is a very gentle bend, so ∫κ²ds barely
registers it and `|dz/ds|` stays inside the climb limit throughout —
the optimizer will ripple the height profile for free. Two terms
prevent that:

- **No overshoot** — z may not leave the band spanned by the leg's
  two gate heights. Without it, a leg between gates at 0.33 m and
  1.25 m was peaking at 1.95 m.
- **Climb once** — any vertical travel beyond the leg's net height
  change is charged for. This kills ripples *inside* the band, which
  the first term does not touch.

Both saturate: raising either by 10× changes nothing. They are
plateaus, not knobs to tune per track. Override with `w_z_band` /
`w_z_mono` on `TrajectoryOptimizer` if you need a track that
deliberately dips or climbs twice between gates.

### C2 is measured on the curvature vector

The C2 term compares dT/ds across each junction, not raw `C''`.
`C''` is parameterization-dependent and grows with the tangent
length, so penalizing it directly makes long-tangent solutions look
catastrophically bad even when their geometric curvature is
perfectly continuous. Measured on a real track, that mis-scoring
was severe: the correct solution scored 1,316,910 against 3,435 for
a visibly worse one, and the optimizer confidently chose the worse
one. dT/ds is in 1/m and scale-free, so the comparison is fair.

### C1 continuity is structural, not penalized

This is the main change from the original scripts and the reason
they produced broken tracks.

Each gate *i* carries one tangent vector

```
T_i = d_i · [cos γ_i·cos ψ_i,  cos γ_i·sin ψ_i,  sin γ_i]
```

where ψ_i is the gate yaw (fixed), d_i > 0 is a tangent length and
γ_i is a climb angle. Segment *i* is built as

```
P0 = g_i          P1 = g_i + T_i
P2, P3 = free     P4 = g_{i+1} − T_{i+1}      P5 = g_{i+1}
```

so `C'(1)` of segment *i−1* and `C'(0)` of segment *i* are both
exactly `5·T_i` — identical in direction **and** magnitude. Kinks
and cusps at gates become unrepresentable rather than merely
expensive, and the drone always crosses a gate along its yaw.
Bounding γ_i by the climb limit also enforces that limit exactly at
the gates.

Optimization variables: 2 per gate (d, γ) + 6 per segment (P2, P3)
= **8n**, down from 12n.

> **Gate yaw is now a hard constraint.** Previously the yaw was a
> soft preference (weight 25) the optimizer could ignore. If your
> yaws do not describe a flow direction a smooth loop can actually
> follow, the run will now report infeasible instead of silently
> emitting a path with a cusp in it. Use `--auto-yaw` to derive
> yaws from the gate positions.

## Optimization strategy

1. **Local refinement** — L-BFGS-B from an analytic initial guess.
   A few seconds. This alone solves every preset.
2. **Escalation** — only if step 1 violates the radius or climb
   limit: a full Differential Evolution global search followed by a
   second polish. The better result wins (feasibility first, cost
   breaks ties).

`--no-de` never escalates; `--force-de` always runs DE. DE is the
only stochastic part, and it is seeded, so runs are reproducible —
the same command twice produces a byte-identical CSV.

## Verified results

Every preset below was run end to end at default settings. `R min`
and `climb max` are measured on the **exported CSV** (after
resampling and any Z rescaling), not on the analytic curve.

| Track | Gates | Loop | R min | Climb max | Max junction | Time |
|---|---|---|---|---|---|---|
| `triangle` | 3 | 20.65 m | 2.95 m | 9.2° | 0.90° | 20 s |
| `pendulum` | 3 | 18.07 m | 1.89 m | 19.0° | 1.27° | 21 s |
| `racetrack` | 3 | 18.68 m | 1.89 m | 14.0° | 1.33° | 21 s |
| `oval` | 4 | 19.16 m | 2.83 m | 10.6° | 0.95° | 28 s |
| `wave` | 5 | 19.59 m | 2.23 m | 18.4° | 0.78° | 33 s |
| `hex` | 6 | 20.74 m | 2.46 m | 16.5° | 0.86° | 35 s |
| `zscale` | 3 | 18.63 m | 1.83 m | 9.4° | 1.16° | 21 s |

Every preset also shows **zero height overshoot** — the trajectory's
z range equals its gates' z range exactly, to the centimetre.

All satisfy R ≥ 1.8 m and climb ≤ 20°. For reference, the
trajectories actually flown by this project (`trajectory_v10.csv`,
`trajectory_4gate.csv`) sit at R min 1.89–1.94 m and climb ≤ 19.3°,
so these are in the real operating envelope.

"Max junction" is the chord turn angle at a gate on the sampled
polyline. It is ~1° purely because of sampling density (a smooth
curve turns by κ·ds per step); on the analytic curve it is 0.000°
at every gate for every preset.

### Reproducing an existing track

Given only the gate positions and the heading through each — read
back out of a shipped CSV, no reference curve and no extra
waypoints — the optimizer recovers this much of the original:

| Source | Loop length | Deviation | R | Kink |
|---|---|---|---|---|
| `trajectory_v10.csv` | 29.85 → 29.68 m (**0.6% off**) | 22 cm mean, 62 cm max | 1.83 m ✓ | 0.000° |
| `trajectory_v8.csv` | 27.12 → 27.18 m (**0.2% off**) | 25 cm mean, 82 cm max | 1.70 m ✗ | 0.000° |
| `trajectory_clov_inverted.csv` | 32.74 → 35.60 m (8.7% off) | 26 cm mean, 66 cm max | 1.83 m ✓ | 0.000° |
| `trajectory_4gate.csv` | 50.10 → 48.17 m (3.9% off) | 44 cm mean, 153 cm max | 1.62 m ✗ | 0.000° |

The loop-around-and-climb-through topology is recovered, including
the v-series legs that leave a gate, circle, and re-enter a second
gate a metre higher. Two caveats: `v8` and `4gate` come back smooth
but below the 1.8 m turn radius — their shipped versions only met
that figure by concentrating all the hard turning into corners, and
removing the corners has to put that curvature somewhere. And the
deviation floor is ~20 cm, so this reproduces the *shape*, not the
exact line.

### Presets

```
triangle    3 gates, equilateral, mild height variation
pendulum    3 gates, one high gate in the middle
racetrack   3 gates, asymmetric with an elevated back corner
oval        4 gates, alternating heights — good baseline
wave        5 gates on a pentagon, alternating heights
hex         6 gates, longest loop
zscale      3 gates, demonstrates z_final rescaling
```

## Gate JSON format

```json
[
  {"pos": [0.0,  3.0, 0.35], "yaw_deg":  180.0},
  {"pos": [-2.9, 0.4, 1.05], "yaw_deg": -114.0},
  {"pos": [0.6, -3.0, 0.35], "yaw_deg":  -10.0},
  {"pos": [2.9,  0.4, 1.05], "yaw_deg":  100.0}
]
```

- `pos` — `[x, y, z]` in metres.
- `yaw_deg` — the direction the drone flies **through** the gate.
- `z_final` — optional; see [Z rescaling](#z-rescaling).

### Gate design rules

1. **Spacing** — if two gates are linked *directly*, keep them
   ≥ 2·R apart (3.6 m at the default radius): a semicircular link at
   R = 1.8 m has a chord of exactly 2R.

   This does **not** mean nearby gates are forbidden. If the yaws
   send the path the long way round — out, around the loop and back
   in — the chord bound never applies. `trajectory_v10.csv` has a
   gate pair 1.35 m apart in XY but 0.92 m apart in height, joined
   by a 12 m loop, and it regenerates fine at R 1.83 m. What matters
   is the route between the gates, not their separation.
2. **Yaw** — must point along the intended flow direction. This is
   now a hard constraint. `--auto-yaw` sets each gate to the
   bisector of its incoming and outgoing chords, which is what a
   smooth loop wants.
3. **Height** — between 0.15 m and 3.5 m. For 80 × 60 cm gates keep
   the centre ≥ 0.32 m above ground.

`--auto-yaw` cannot help when the gates are collinear and the loop
doubles back on itself: the bisector is undefined there and the
outgoing chord is used instead. Such layouts need hand-picked yaws
(or an extra gate to open the return path out).

## Z rescaling

Set `z_final` on a gate to optimize the shape at one set of heights
and export at another. Useful when optimizing directly at the final
heights is ill-conditioned — typically a tall track you want to
compress into the arena.

The whole loop is mapped by a **single global affine transform**
`z → a·z + b`, least-squares fitted to the `(z, z_final)` pairs.
Rescaling one coordinate affinely is a linear transform of the
curve, so C1/C2 continuity is preserved exactly. The reported
`max gate error` is 0 whenever the pairs are collinear, which
covers shifting and/or compressing a height band:

```
$ python3 generate_track.py --track zscale
  Z rescaled: z → 0.4800·z + +0.3320   (max gate error 0.0cm)
  S1 (G1→G2): L=  5.74m  R= 1.98m ✓  climb=  9.2° ✓  Z=[0.50, 1.10]
```

If the targets are not affine-compatible the script warns and tells
you to optimize at the final heights directly instead.

## Choosing `ds_resample` in the consumer

`external_trajectory.py` resamples the CSV before fitting its
B-spline, and its default spacing of **1.0 m is too coarse** to
preserve a 1.9 m turn radius. Measured minimum radius of the
steady-state loop after it round-trips through
`load_external_trajectory()`:

| ds_resample | `triangle` | `wave` | `oval` |
|---|---|---|---|
| 1.00 m | 1.39 m | 0.85 m | 2.74 m |
| 0.50 m | 1.73 m | 1.47 m | 2.76 m |
| 0.25 m | 1.98 m | 1.83 m | 2.77 m |
| *analytic* | *2.05 m* | *1.93 m* | *2.77 m* |

Pass `ds_resample=0.25` for tracks whose radius is near the limit.
Tracks with generous radius (`oval`) are insensitive.

That the numbers **converge** on the analytic value as the spacing
tightens is the check that the exported geometry is genuinely
smooth. The legacy CSVs in this directory diverge instead
(`trajectory_v10.csv`: 1.36 → 1.03 → 0.79 m over the same spacings),
which is the signature of kinks in the source points.

## CLI reference

### Track definition
| Flag | Default | Description |
|---|---|---|
| `--track NAME` | — | preset layout |
| `--gates FILE` | — | gate JSON (overrides `--track`) |
| `--auto-yaw` | off | derive yaws from gate positions |

### Output
| Flag | Default | Description |
|---|---|---|
| `--output PATH` | `trajectory.csv` | output CSV |
| `--n-pts N` | 150 | samples per segment |
| `--plot [PATH]` | off | PNG preview; defaults next to the CSV |
| `--quiet` | off | suppress progress output |

### Drone / arena
| Flag | Default |
|---|---|
| `--min-radius M` | 1.8 |
| `--max-climb DEG` | 20.0 |
| `--max-speed MPS` | 4.0 (lap-time estimate only) |
| `--gate-width M` | 0.80 |
| `--gate-height M` | 0.60 |
| `--drone-margin M` | 0.10 |
| `--flight-box X Y Z` | 7.5 7.5 4.0 |
| `--flight-center X Y Z` | 0 0 1.5 |
| `--min-traj-z M` | 0.15 |

### Optimizer
| Flag | Default | Description |
|---|---|---|
| `--de-iter N` | 60 | DE generations |
| `--de-pop N` | 15 | DE population |
| `--lbfgs-iter N` | 800 | L-BFGS-B iterations |
| `--seed N` | 42 | DE seed |
| `--no-de` | off | never escalate to DE |
| `--force-de` | off | always run DE |

### Post-processing
| Flag | Default | Description |
|---|---|---|
| `--no-reorder` | off | keep the given gate order |
| `--start-gate I` | lowest gate | rotate the loop to start at gate `I` (0-based) |
| `--no-zscale` | off | ignore `z_final` |

## Worked examples

All of these were run and verified.

```bash
# 1. Baseline. 4 gates, alternating heights.
python3 generate_track.py --track oval --output oval.csv --plot
#    → R min 2.83 m, climb 10.6°, loop 19.16 m

# 2. Longest preset loop, 6 gates.
python3 generate_track.py --track hex --output hex.csv
#    → R min 2.46 m, climb 16.5°, loop 20.74 m

# 3. Your own gates, letting the script pick the yaws.
python3 generate_track.py --gates my_gates.json --auto-yaw \
    --output custom.csv --plot

# 4. Optimize tall, export compressed (z_final).
python3 generate_track.py --track zscale --output zscale.csv
#    → z → 0.48·z + 0.332, exact; R min 1.83 m, climb 9.4°

# 5. Tighter radius than the airframe default.
python3 generate_track.py --track oval --min-radius 2.2 --output r22.csv

# 6. Shallower climb limit.
python3 generate_track.py --track oval --max-climb 12 --output c12.csv

# 7. Smaller gates (50 × 40 cm openings).
python3 generate_track.py --track oval \
    --gate-width 0.5 --gate-height 0.4 --output narrow.csv

# 8. Start the loop at a specific gate instead of the lowest.
python3 generate_track.py --track oval --start-gate 2 --output g2.csv

# 9. Skip the DE escalation for a fast iteration loop.
python3 generate_track.py --track wave --no-de --output quick.csv
```

### Recovering a bad layout

Feeding in yaws that do not describe a followable flow:

```
$ python3 generate_track.py --gates hard.json
  Local refinement: cost 62772.3   constraints: violated ✗
  Global search: Differential Evolution — [constraints violated]
  DE cost: 28463.2
  Kept: cost 28463.1653, constraints violated ✗
```

The same gates with `--auto-yaw`:

```
$ python3 generate_track.py --gates hard.json --auto-yaw
  Local refinement: cost 2.0080   constraints: satisfied ✓
  Loop: 20.10m  |  lap ≈ 5.0s @ 4.0m/s  |  ALL OK ✓
```

Note the first run does **not** quietly emit a usable-looking file —
it reports the violation and exits non-zero.

## Outputs

### `trajectory.csv`

```
seg,t,x,y,z,curvature,climb_rate,climb_angle_deg
0,0.0,0.0,3.0,0.325,0.3612,0.0,0.0
...
```

Byte-identical header to the existing `trajectory_*.csv` files, so
it is a drop-in for `external_trajectory.py`.

- `seg` — segment index
- `t` — parameter within the segment, [0, 1]
- `x, y, z` — position in metres
- `curvature` — κ in 1/m
- `climb_rate` — `|dz/ds|` = sin γ
- `climb_angle_deg` — `asin(climb_rate)`

`curvature` and `climb_rate` are recomputed from the **final**
sampled geometry, so they stay consistent with `x,y,z` after Z
rescaling. Curvature uses the circumcircle through each consecutive
triple (Menger curvature), which is exact for any sample spacing —
differentiating twice with `np.gradient` instead spikes wherever the
spacing jumps, i.e. at every segment junction.

`external_trajectory.py` only reads `x,y,z`; the rest is for
inspection and validation.

### `trajectory_gates.json`

Gate positions in export order, with both the optimized and final
heights.

### `trajectory.png` (with `--plot`)

Three panels: 3D path, top view with gate headings, and turn radius
plus climb angle along the loop with the limits marked. Falls back
to a height profile in place of the 3D panel if `mpl_toolkits.mplot3d`
is unavailable — which is the case on this machine, where a pip
matplotlib shadows the system one.

## Python API

```python
from generate_track import (Gate, DroneConfig, ArenaConfig,
                            generate, export_csv, auto_yaw)

gates = [
    Gate(3.0,  1.5, 0.35, -69.7),
    Gate(-0.5, -1.5, 1.60, 175.2, z_final=1.10),
    Gate(-3.0, 1.5, 0.35, 64.9),
]

rows, gates_ordered, opt, ok = generate(
    gates,
    drone=DroneConfig(min_turn_radius=1.8, max_climb_deg=20.0),
    arena=ArenaConfig(flight_box=(7.5, 7.5, 4.0)),
)

if ok:
    export_csv(rows, "trajectory.csv")
```

`generate()` returns `ok=False` if the exported trajectory violates
any constraint — check it before flying.

Lower-level pieces: `TrajectoryOptimizer.optimize()`,
`.analyze(segments)`, `.junction_report(segments)` (turn angle per
gate — 0 by construction), `.feasible(segments)`, and the
post-processing helpers `sample_segments`, `reorder_from_gate`,
`fit_z_map`, `scale_z`, `recompute_derived`, `analyze_rows`.

## What changed vs. the two original scripts

Merged from `optimize_track.py` and `generate_trajectory.py`. Beyond
the union of their features, these are behaviour changes worth
knowing about — each one fixes a defect found while testing.

1. **C1 continuity is structural.** Previously it was a soft
   penalty (weight 50) the optimizer could and did pay. With the
   original presets, 5 of 6 tracks came out with kinks at the
   gates — up to a **177° reversal** on `racetrack`, i.e. the path
   doubled back on itself. Forcing DE did not help: identical cost,
   identical kinks. It was the formulation, not a local minimum.

2. **Junction checking added.** The old `analyze()` swept each
   segment independently and so was structurally blind to
   discontinuities *between* segments — it printed `ALL OK ✓` for
   every one of those kinked tracks. Both report paths now check
   the joins, with a tolerance that scales with sample spacing.

3. **Z rescaling rewritten.** The old `scale_z()` applied a
   *different* scale factor per segment (`new_range/old_range`,
   falling back to 1.0 when a segment's gates were level). Adjacent
   segments therefore got different Z scales, putting a slope
   discontinuity at every gate. Measured on the old `highloop`
   layout: R min collapsed from 2.09 m to **0.57 m** purely from
   rescaling. Replaced by one global affine map, which preserves
   continuity exactly.

4. **Derived CSV columns are recomputed after rescaling.** The old
   pipeline rescaled Z but exported the pre-rescale `curvature` and
   `climb_rate`, so those columns described a curve that was never
   flown.

5. **Climb angle uses `asin`, not `atan`.** `climb_rate` is
   `|dz/ds|`, which is sin γ — but both originals reported
   `atan(climb_rate)` and compared it against `tan(γ_max)`. At a
   20° setting that silently allowed 21.35°. Now consistent
   throughout, and consistent with `external_trajectory.py:290`,
   which already used `asin`. The constraint is very slightly
   tighter than before.

6. **DE escalation replaces the init-cost heuristic.** The old rule
   skipped DE when the initial cost was under 1e6 — a threshold
   uncorrelated with whether the *result* is any good. Now DE runs
   only when the cheap local solve actually fails, which is both
   faster in the common case and more reliable in the hard one.

7. **Presets fixed.** The original yaws were not flow-consistent
   and are infeasible under the now-hard heading constraint. All
   seven presets ship with corrected yaws and are verified above.
   `cloverleaf` became `triangle` (with auto-yaws it is a rounded
   triangle, not three petals); `slalom` had collinear gates
   requiring two 180° turns at R ≥ 1.8 m in a 7.5 m box — replaced
   by `wave`. `hex` and `zscale` are new.

8. **Gate clearance is vectorized** — same arithmetic, no Python
   loop over sample points inside the cost function.

9. **Dead code removed** — the old `_bounds()` was unreachable and
   would have raised had it been called.
