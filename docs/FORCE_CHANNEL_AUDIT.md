# Force-channel audit

**Why this exists.** Four quantities in this project can be called "the pull force", and
during Phase 8 one of them read 17-23 N while the command was 5 N. Rather than assume that
was a bug or assume it was the truth, every channel was measured against the drawer's
equation of motion.

**How.** `python scripts/audit_force_channels.py --headless`, which runs one execution over
six hidden states chosen to isolate the terms (no resistance, damping only, friction only,
both, asymmetric friction, heavy) and compares the channels over the sliding plateau.
Report: `outputs/logs/force_channel_audit.json`. Figure:
`outputs/plots/force_channel_comparison.png`. Run on 2026-09-02.

---

## The four channels

| Channel | What it is | Provenance | Deployability |
|---|---|---|---|
| `commanded_force` | What the controller asked the OSC for | force profile | **deployable** |
| `measured_force` | Pull-axis component of the wrist joint reaction wrench | `body_incoming_joint_wrench_b` at `panda_hand`, rotated out of `panda_link7` | diagnostic |
| `drawer_resistance_force` | The drawer's internal resistance | `ArticulationView.get_dof_projected_joint_forces` on `drawer_top_joint` | sim-only privileged |
| `drawer_external_force` | The axial force delivered to the drawer | `m_total * a - resistance` | sim-only privileged |

Also recorded, and *not* a pull force: `handle_contact_force_w`, the net contact force on the
handle body. It reports normal grip load only.

---

## What each API actually turned out to mean

### `get_dof_projected_joint_forces` is the drawer's internal resistance

Measured directly, four cases:

| Hidden state | Mean drawer speed | Channel reads | `-(mu_d + b*v)` predicts |
|---|---|---|---|
| `b = 0`, `mu = 0` | 0.111 m/s | 0.0000 N | 0.0000 N |
| `b = 8`, `mu = 0` | 0.095 m/s | **-0.7680 N** | -0.7611 N |
| `b = 0`, `mu = 3` | 0.055 m/s | **-3.0000 N** | -3.0000 N |
| `b = 8`, `mu = 3` | 0.063 m/s | **-3.5102 N** | -3.5038 N |

So the channel is exactly `-(mu_dynamic * sign(v) + b * v)`: negative while the drawer
opens, because it opposes motion. It does **not** include the external pull or the inertial
term.

Across the six audit cases the mean residual against the prediction from the hidden state
was at most **0.0099 N**, against forces of 2-3 N. The worst *per-step* residual was
0.1953 N, which is the velocity filter rather than the channel: PhysX applies `b * v` with
its own instantaneous velocity while the prediction uses this project's causally filtered
one, and at `b = 8` a 0.02 m/s ripple is already 0.16 N. The identity is therefore checked
on window means, and that choice is documented in the analysis module.

### The joint reaction wrench cannot give the drawer-axis force

`get_link_incoming_joint_force` at `drawer_top` reads **0.000** along the pull axis in every
case. This is structural, not a bug: a prismatic joint does not constrain its own axis, so
the constraint wrench along that axis is zero by definition. The API is correct and simply
does not answer this question.

### A contact sensor on the handle reports normal load only

Measured earlier (`docs/DECISIONS.md` D006): `net_forces_w` stayed at 0.22-0.37 N whether the
commanded pull was 4 N or 12 N, because the pull is transmitted through *tangential* finger
friction that the net-contact-force report does not include. Kept as a witness of grip
quality; nothing decides anything from it.

### The delivered force follows from the equation of motion

With the resistance channel measured and the mass known,
`F_external = m_total * a - F_resistance`. This is privileged twice over: it needs the
drawer's mass, which is part of the hidden state the robot is meant to be inferring.

---

## Do the channels agree?

Measured over the sliding plateau, `peak_force = 4 N`, `duration = 1.5 s`:

| Case | `F_cmd` | `F_wrist` | `F_resist` | `F_external` | `F_ext / F_cmd` | `|F_ext - F_wrist|` |
|---|---|---|---|---|---|---|
| free (`mu=0, b=0`) | 3.996 | 1.734 | -0.000 | 1.600 | 0.40 | 0.134 |
| damping only (`b=8`) | 3.996 | 2.128 | -0.906 | 2.064 | 0.52 | 0.064 |
| friction only (`mu=2`) | 3.998 | 2.887 | -2.000 | 2.778 | 0.69 | 0.110 |
| friction + damping | 3.998 | 3.066 | -2.360 | 3.012 | 0.75 | 0.054 |
| asymmetric (`mu_s=2.5, mu_d=0.5`) | 3.998 | 2.250 | -0.918 | 2.144 | 0.54 | 0.106 |
| heavy (`m=14`) | 3.998 | 3.259 | -2.295 | 3.210 | 0.80 | 0.049 |

**Delivered force vs wrist sensor: agree to within 0.049-0.134 N**, against a
hand-and-finger inertial bound of 0.342 N. The two bracket the same interaction from either
side of the grasp and the residual is the mass the wrist carries beyond the drawer. This
validates the wrist channel rather than merely restating it.

**Only 40-80 % of the commanded force reaches the drawer**, and the fraction rises with the
drawer's resistance. The rest accelerates the arm's own reflected inertia, so a stiffer
drawer -- which accelerates less -- wastes less. This is real physics, not a control defect,
but it means `F_peak` and the force delivered to the drawer are not interchangeable.

---

## The 17-23 N wrist force, explained

A deliberate end-stop episode (`m = 5`, `mu = 1.5`, `b = 4`, `F_peak = 6 N`, `T = 2.5 s`):

| | |
|---|---|
| Peak commanded force | 6.0 N |
| Peak absolute wrist force | **59.2 N** |
| Ratio to command | **9.87x** |
| Time of the spike | 2.117 s |
| Displacement at the spike | 0.3377 m = **84.4 % of travel** |
| Final displacement | 0.3479 m = 87.0 % of travel |
| Peak velocity | 0.446 m/s |

The spike coincides with the drawer arriving at its mechanical end stop at 0.45 m/s. It is
an impact, not a control fault -- and the operating region already rejects any episode above
80 % of travel (`docs/EXPERIMENT_SPACE.md`), so it cannot contaminate an Oracle label. The
Phase 8 anomaly was a milder version of the same thing.

---

## Decision

| Channel | First ACE | Reason |
|---|---|---|
| `commanded_force` | **required** | The robot always knows the force it asked for. Two drawers that both move 5 mm, one at 4 N and one at 8 N, have different dynamics, and the command is what makes that distinguishable (`docs/DECISIONS.md` D018). |
| `measured_force` | recorded, not required | A real Franka has a wrist sensor, so this is legitimately deployable, but it is noisy (a few newtons of swing at 5 N command level) and the first ACE does not need it. Kept for the ACE-5 ablation. |
| `drawer_resistance_force` | **forbidden** | Simulator-only. Verification, analysis and privileged teachers. |
| `drawer_external_force` | **forbidden** | Simulator-only, and needs the hidden state itself. |

`probe_drawer.observations.validate_model_input()` enforces the last two, and
`tests/unit/test_observation_spec.py` asserts that no privileged channel is flagged for the
model.
