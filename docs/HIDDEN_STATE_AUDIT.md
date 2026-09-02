# Hidden-state capability audit

**What this is.** The main paper's hidden state is four dimensional. That is a choice, and a
choice is only defensible if the alternatives were examined, so every physical quantity of
the cabinet, the drawer and the robot that this installation exposes was enumerated, probed
on the live simulation, and given a role.

**How the measured columns were produced.** `python scripts/audit_hidden_states.py --headless`
writes each candidate, reads it back out of PhysX, restores it, and records the result.
Nothing is stepped, so the audit cannot perturb a later experiment. Report:
`outputs/logs/hidden_state_audit.json`. Run on 2026-09-02, Isaac Sim 5.1.0.0 / Isaac Lab
2.3.0.

**Headline result.** All 15 candidates are writable and every write reads back correctly.
Nothing was ruled out for being unavailable; the four in the paper were chosen on physical
and identifiability grounds.

---

## The audit table

`visible` means a robot could observe the quantity directly at deployment. A hidden state
must be invisible; anything visible belongs in the observation instead.

| Parameter | Simulator API | Target | Writable | Reads back | Visible | Role |
|---|---|---|---|---|---|---|
| `drawer_mass` | `ArticulationView.set_masses` (+ `set_inertias`) | body `drawer_top` | yes | yes | no | **main-paper xi** |
| `joint_static_friction` | `write_joint_friction_coefficient_to_sim` (static) | `drawer_top_joint` | yes | yes | no | **main-paper xi** |
| `joint_dynamic_friction` | `write_joint_friction_coefficient_to_sim` (dynamic) | `drawer_top_joint` | yes | yes | no | **main-paper xi** |
| `joint_damping` | `write_joint_damping_to_sim` | `drawer_top_joint` | yes | yes | no | **main-paper xi** |
| `joint_viscous_friction` | `write_joint_friction_coefficient_to_sim` (viscous) | `drawer_top_joint` | yes | yes | no | held fixed |
| `joint_armature` | `write_joint_armature_to_sim` | `drawer_top_joint` | yes | yes | no | held fixed |
| `handle_contact_friction` | `ArticulationView.set_material_properties` | every cabinet material | yes | yes | no | held fixed |
| `joint_stiffness` | `write_joint_stiffness_to_sim` | `drawer_top_joint` | yes | yes | no | OOD candidate |
| `drawer_center_of_mass` | `ArticulationView.set_coms` | body `drawer_top` | yes | yes | no | OOD candidate |
| `drawer_inertia_tensor` | `ArticulationView.set_inertias` | body `drawer_top` | yes | yes | no | not suitable |
| `restitution` | `set_material_properties` (restitution) | every cabinet material | yes | yes | no | not suitable |
| `drawer_travel_limit` | `ArticulationView.set_dof_limits` | `drawer_top_joint` | yes | yes | **yes** | not suitable |
| `drawer_effort_limit` | `set_dof_max_forces` | `drawer_top_joint` | yes | yes | no | not suitable |
| `robot_actuator_gains` | `set_dof_stiffnesses` / `set_dof_dampings` | `panda_joint1..7` | yes | yes | **yes** | not suitable |
| `gravity` | `set_disable_gravities` | cabinet bodies | yes | yes | **yes** | not suitable |

Observed round trips (a sample): `drawer_mass` 5.175 -> 9.75, `joint_static_friction`
0 -> 4.25, `joint_dynamic_friction` 0 -> 1.75, `joint_damping` 1.0 -> 7.5,
`drawer_travel_limit` 0.4 -> 0.3, `handle_contact_friction` 0.5 -> 0.9.

---

## The four in the paper, and what each one does to a pull

| | Physical meaning | Effect on a force-driven pull | Where it shows in a probe |
|---|---|---|---|
| `drawer_mass` *m* | Mass of the moving drawer assembly (kg) | Sets the inertial term: how much of the applied force becomes acceleration | Peak acceleration: measured 0.133 -> 0.078 m/s² as *m* went 4 -> 12 kg |
| `joint_static_friction` *μ_s* | Coulomb effort resisting the *start* of motion (N) | Sets the breakaway force: below it the drawer does not move at all | Breakaway: 0.150 s at 1.67 N -> 0.400 s at 2.92 N as *μ_s* went 0.5 -> 3.0 N |
| `joint_dynamic_friction` *μ_d* | Coulomb effort resisting *continued* motion (N) | Velocity-independent drag once sliding | Probe duration 0.350 -> 0.467 s as *μ_d* went 0.15 -> 1.25 N |
| `joint_damping` *b* | Viscous damping of the joint drive (N s/m) | Velocity-proportional drag; caps terminal speed | **Barely visible** at the calibrated probe's speeds -- see the limitation below |

Measurements from `python scripts/plot_probe_identifiability.py --headless`
(`outputs/plots/probe_identifiability.png`).

### One honest limitation

The calibrated probe stops after 3 mm, where the drawer is moving at roughly 0.013 m/s.
Viscous drag is then `b * v` = 0.14 N at *b* = 11 N s/m, against a 2 N command -- so it is
close to invisible. Sweeping *b* from 2 to 11 N s/m changed the probe duration from 0.400 s
to 0.400 s and the breakaway force from 1.92 N to 1.92 N. **This probe identifies mass and
both frictions; it does not identify damping.**

That does not invalidate the task: the same sweep shows the *required peak force* is also
only weakly dependent on *b*, so a probe that misses *b* can still predict the force. But it
does mean a damping-identifying probe would need a second phase that reaches higher speed,
and that is recorded as a next-step recommendation rather than papered over.

---

## Why the others are not in the paper

**Held fixed** -- writable, physically real, but they would make `xi` unidentifiable or
would confound the study.

* `joint_viscous_friction`: a second velocity-proportional drag. Along one degree of freedom
  it has exactly the same signature as `joint_damping`; including both would mean two
  parameters no probe could ever separate. Pinned to 0.
* `joint_armature`: added inertia on the joint. Degenerate with `drawer_mass` for a
  prismatic DOF. Pinned to 0.
* `handle_contact_friction`: changes the *grasp*, not the drawer. Varying it would make a
  failed pull ambiguous between a stiff drawer and a slipping grip. Pinned to the midpoints
  of the official environment's randomisation ranges (see `docs/DECISIONS.md` D010).

**Out-of-distribution candidates** -- real, interesting, and deliberately saved.

* `joint_stiffness`: a spring-loaded drawer. Genuinely different physics -- the force needed
  grows with travel -- and the official cabinet ships with 10 N/m, which contributes about
  3 N over full travel. Removed from the main paper because it is a fourth mechanism on top
  of `xi` (D008); a natural OOD axis later.
* `drawer_center_of_mass`: an unevenly loaded drawer. Perturbs the rail contact, but its
  effect on the axial response is far weaker than any element of `xi`.

**Not suitable.**

* `drawer_inertia_tensor`: the drawer translates and does not rotate, so this has almost no
  observable effect. It cannot be inferred, and would only add nuisance dimensions.
* `restitution`: matters only on impact, i.e. at the mechanical end stop, which valid
  episodes never reach.
* `drawer_travel_limit`: **visible**. A drawer's travel is apparent to look at, and it is the
  task's geometry rather than its dynamics.
* `drawer_effort_limit`: inactive here -- the drawer's drive is passive, so the limit is
  never reached and no probe could reveal it.
* `robot_actuator_gains`: **visible**. A robot knows its own controller gains. Varying them
  would confound arm behaviour with drawer dynamics, and the hybrid OSC needs them at zero.
* `gravity`: **visible**, and not a property of an unknown drawer.

---

## What this means for later phases

* The main paper varies `[m, mu_s, mu_d, b]` and nothing else. Training and OOD ranges are
  in `docs/EXPERIMENT_SPACE.md`.
* Two OOD axes are already known to be available and physically meaningful:
  `joint_stiffness` and `drawer_center_of_mass`. Adding either needs no new simulator work,
  only a decision.
* `mu_s >= mu_d` is a hard PhysX constraint, not a modelling choice. See
  `docs/DECISIONS.md` D016.
