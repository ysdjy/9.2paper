# RMA²-inspired Direct Adaptation

One of this paper's baselines, kept in its own folder so that nothing it does can change the
benchmark the other methods are measured on.

```
tau_p  ──►  AdaptationEncoder  ──►  z_probe  ──►  ParameterHead  ──►  p* = [F_peak, T]
                                       ≈
xi     ──►  PrivilegedEncoder   ──►  z_priv                     (training only)
```

Against the paper's other two adaptation methods, the **only** thing that differs is how `p`
is produced after the probe:

| | privileged teacher | latent distillation | what is predicted |
|---|---|---|---|
| Direct Regression | no | no | `p*` |
| **RMA²-inspired (this)** | **yes** | **yes** | `p*` |
| Ours (ACE + PSP + SPC) | yes | yes | `P(success \| z, p, goal)`, then search |

It is called *RMA²-inspired* and not *RMA²* on purpose: RMA² adapts a PPO policy that emits
low-level actions, and this paper does not learn a low-level policy at all. What is carried
over, and what is deliberately dropped, is set out with source references in
[docs/RMA2_TO_DRAWER_MAPPING.md](docs/RMA2_TO_DRAWER_MAPPING.md) §4 and §20.

---

## Status

```
[x] Official RMA² reproduced: environment, PPO, adapter training, evaluation
[x] Official source analysed: privileged vector, encoder, latent, temporal CNN, both stages
[x] Module mapping and the baseline's full design contract
[x] adaptation_premise: is the adaptation problem well posed? (offline audit, 12 tests)
[ ]  --- blocked on three project-level tasks, see "What has to happen first" ---
[ ] Stage A: privileged direct adaptation
[ ] Stage B: latent distillation
[ ] Deployment + evaluation
```

**No model code is written yet, and that is deliberate.** The audit found that on the current
one-dimensional parameter space the success set is a contiguous interval whose midpoint
succeeds for 104 of 105 hidden states — so there is no multi-modality for a success-landscape
model to exploit, and the paper's central claim is not testable. That is what moved the skill
parameter to `p = [F_peak, T]` (**D034**), and it is what blocks this baseline.

### What has to happen first

Three tasks, in order, and **none of them belongs to this baseline** — they change the
benchmark, so they belong to the project and to every method equally:

1. re-sweep the Oracle over a `(F_peak, T)` grid;
2. re-select `MAIN_TASK` against it with the existing scored rule (D024);
3. generalise `adaptation_premise` from bands to regions and re-run it.

Task 3 produces the number that decides whether this comparison is worth running at all: **how
many hidden states have a *disconnected* success region.** If the 2-D regions turn out to be
convex blobs, the framing has to change rather than the measurement.
[docs/RMA2_TO_DRAWER_MAPPING.md](docs/RMA2_TO_DRAWER_MAPPING.md) §3.1.

---

## Layout

```
baseline/rma2_direct/
├── docs/
│   ├── RMA2_REPRODUCTION_REPORT.md   the official code, read line by line, and how far it ran
│   └── RMA2_TO_DRAWER_MAPPING.md     what transfers, the design contract, and the measured risks
├── src/rma2_direct/
│   └── adaptation_premise.py         is the adaptation problem well posed? offline, no Isaac Sim
├── scripts/
│   └── audit_adaptation_premise.py   runs the above and writes the report
├── tests/                            12 tests, no Isaac Sim, ~1 s
├── patches/rma4rma/                  the four fixes the official code needs, plus the installer
└── third_party/rma4rma/              the official clone (git-ignored, not vendored)
```

## Relationship to the main project

This baseline **imports `probe_drawer` and never modifies it.** The environment, the
controllers, the probe, the sequential protocol, the evaluator, the Oracle, the dataset schema
and the splits are all shared, because a comparison in which one method has its own controller
or its own success criterion measures nothing
([docs/RMA2_TO_DRAWER_MAPPING.md](docs/RMA2_TO_DRAWER_MAPPING.md) §14).

Two consequences worth stating plainly:

* If this baseline needs a change in `src/probe_drawer/`, that change belongs to the project
  and to **every** method — not to this folder.
* Anything shared living inside one method's folder is how an unfair comparison starts. The
  parameter space, the oracle target and the metric definitions must end up in
  `probe_drawer`, not here, as soon as a second method needs them.

Project-level artefacts stay at the repository root: this baseline's audit reads
`outputs/logs/sequential_oracle_*.json` and writes `outputs/logs/adaptation_premise.json`,
which `docs/TRAINING_V0.md` already cites by that name.

## Running it

The audit needs only the main project installed — no Isaac Sim, about a second:

```bash
conda activate env_isaaclab
cd /home/zbh/Downloads/IsaacLab/9.2paper
python baseline/rma2_direct/scripts/audit_adaptation_premise.py
python -m pytest baseline/rma2_direct/tests -q          # 12 tests
```

`rma2_direct` is put on the path by `tests/conftest.py` and by the script itself, so nothing
has to be installed. `pip install --no-deps -e baseline/rma2_direct` also works and makes
those bootstraps no-ops; it is not done by default, because this baseline must not add
anything to an environment the other methods and other agents share.

### Reproducing the official RMA²

A **separate** conda environment — SAPIEN 2.2.2 and `numpy < 1.24` are incompatible with
Isaac Sim, so `env_isaaclab` is never touched:

```bash
bash baseline/rma2_direct/patches/rma4rma/install_rma2.sh    # creates conda env `rma2`
```

The official code **does not run as published** — four separate defects, each observed rather
than inferred, fixed by the patches in `patches/rma4rma/` and documented in
[docs/RMA2_REPRODUCTION_REPORT.md](docs/RMA2_REPRODUCTION_REPORT.md) §21. The most serious is
that the ManiSkill2 fork's own HEAD commit misspells `set_drive_target`, so every `env.step`
raises `AttributeError`.

TurnFaucet could not be reproduced on this machine: the ManiSkill2 asset server is unreachable
through this network's proxy. `PegInsertionSide-v1` was used instead, for the reasons in §4 of
that report.
