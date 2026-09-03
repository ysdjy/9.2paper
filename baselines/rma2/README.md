# RMA² baseline — RMA²-inspired Direct Adaptation

One of this paper's baselines, in its own folder so that nothing it does can change the
benchmark the other methods are measured on.

```
tau_p  ──►  AdaptationEncoder  ──►  z_probe  ──►  ParameterHead  ──►  p*
                                       ≈
xi     ──►  PrivilegedEncoder   ──►  z_priv                     (training only)
```

Against the paper's other two adaptation methods, the **only** thing that differs is how `p`
is produced after the probe:

| | privileged teacher | latent distillation | what is predicted | how `p` is chosen |
|---|---|---|---|---|
| Direct Regression | no | no | `p*` | it is the output |
| **RMA²-inspired (this)** | **yes** | **yes** | `p*` | it is the output |
| Ours (ACE + PSP + SPC) | yes | yes | `P(success \| z, p, goal)` | search over candidates |

It is called *RMA²-inspired* and not *RMA²* on purpose: RMA² adapts a PPO policy that emits
low-level joint actions, and this paper does not learn a low-level policy at all. What is
carried over and what is deliberately dropped, with source references,
is in [docs/RMA2_TO_DRAWER_MAPPING.md](docs/RMA2_TO_DRAWER_MAPPING.md) §4 and §20.

---

## 1. Status

```
[x] Official RMA² reproduced: environment, PPO, adapter training, evaluation
[x] Official source analysed: privileged vector, encoder, latent, temporal CNN, both stages
[x] Module mapping and this baseline's full design contract
[x] adaptation_premise: is the adaptation problem well posed? (offline audit, 12 tests)
[x] Setting V1 design audit                    (docs/SETTING_V1_DESIGN_AUDIT.md)
[x] Stage A: privileged direct adaptation      (xi -> z_priv -> ParameterHead -> F_peak*)
[x] Stage B: latent distillation               (tau_p -> z_probe ~= z_priv)
[x] Deployment + evaluation on the shared protocol (in-distribution)
```

**§2 below is obsolete and has not yet been rewritten.** Its blocker -- that a
one-dimensional parameter space cannot test the paper's claim -- was both superseded (Setting
V1 refroze to 1-D, D044) and empirically refuted (the landscape model wins in 1-D anyway).
See [docs/SETTING_V1_DESIGN_AUDIT.md](docs/SETTING_V1_DESIGN_AUDIT.md) §0 for what replaces
it, and §4 there for the list of stale text in this file and in the mapping.

**Stage A and Stage B are done**, and together they decompose the paper's headline gap.
Deployed in one session on 88 unseen drawers with three seeds:

| method | reach |
|---|---|
| Stage A, `xi` -> point | 97.3 % |
| teacher, `xi` -> landscape | 97.0 % |
| ACE + PSP, probe -> landscape | 93.9 % |
| **Stage B, probe -> latent -> point** | **92.0 %** |
| Direct GRU, probe -> point | 79.2 % |

So of ACE + PSP's +14.8 pp over Direct GRU, **latent distillation accounts for +12.9 pp (87 %)
and the success landscape for +1.9 pp (13 %)** -- and at the privileged level the landscape is
worth nothing at all (-0.4 pp). [docs/STAGE_A_RESULTS.md](docs/STAGE_A_RESULTS.md),
[docs/STAGE_B_RESULTS.md](docs/STAGE_B_RESULTS.md).

## 2. What blocks Stage A, and why it is not this baseline's to unblock

The audit found that on the current one-dimensional parameter space each hidden state's
success set is a contiguous interval whose midpoint succeeds for **104 of 105** hidden states.
So there is no multi-modality for a success-landscape model to exploit, and the paper's
central claim — that predicting *where* the successes are beats predicting one answer — is not
testable as posed. That is what moved the skill parameter to `p = [F_peak, T]` (**D034**).

Three tasks follow, in order, and **none of them belongs to this baseline**, because each one
changes the benchmark and therefore has to be identical for every method:

1. re-sweep the Oracle over a `(F_peak, T)` grid;
2. re-select `MAIN_TASK` against it with the existing scored rule (D024);
3. generalise `adaptation_premise` from bands to regions and re-run it.

Task 3 produces the number that decides whether this comparison is worth running at all: **how
many hidden states have a *disconnected* success region.** If the 2-D regions turn out to be
convex blobs, the framing has to change rather than the measurement.
[docs/RMA2_TO_DRAWER_MAPPING.md](docs/RMA2_TO_DRAWER_MAPPING.md) §3.1.

Everything in §4 is written so that the parameter space is a **config value, not an
architectural assumption** — the head's output width and the candidate grid come from the
config, so moving from 1-D to 2-D changes data, not code.

## 3. Layout

```
baselines/rma2/
├── README.md                          this file
├── pyproject.toml                     optional editable install; deliberately not installed by default
├── configs/
│   └── adaptation_premise.yaml        the audit's tunables (read by its script)
├── src/rma2/
│   └── adaptation_premise.py          is the adaptation problem well posed? offline, no Isaac Sim
├── scripts/
│   └── audit_adaptation_premise.py    runs the above, writes the report
├── tests/                             12 tests, no Isaac Sim, ~1 s
├── docs/
│   ├── RMA2_REPRODUCTION_REPORT.md    the official code, read line by line, and how far it ran
│   └── RMA2_TO_DRAWER_MAPPING.md      what transfers, the design contract, the measured risks
├── checkpoints/                       trained weights (git-ignored; manifests tracked)
├── patches/rma4rma/                   the four fixes the official code needs, plus the installer
└── third_party/rma4rma/               the official clone (git-ignored, not vendored)
```

## 4. How this plugs into the main project

**One direction only.** This baseline imports `probe_drawer`; `probe_drawer` never imports
this. `grep -rn "baselines" src/probe_drawer/` returning nothing is the invariant, and it is
what keeps the main method independent of a baseline's fate.

### 4.1 What is shared, and must be

A comparison in which one method has its own controller or its own success criterion measures
nothing. These come from the main project unchanged, by import
([docs/RMA2_TO_DRAWER_MAPPING.md](docs/RMA2_TO_DRAWER_MAPPING.md) §14):

| Shared | From |
|---|---|
| Drawer environment, Franka, grasp reset | `probe_drawer.pull_system.PullSystem` |
| hidden dynamics `xi`, its ranges and sampling | `probe_drawer.envs.DynamicsRandomizer`, `experiment_plan.{TRAINING,OOD}_XI_RANGES` |
| the standardised probe and the pull controller | `probe_drawer.controllers` |
| probe → gap → execution, no reset | `probe_drawer.protocols.SequentialPullProtocol` |
| task, success definition, validity | `probe_drawer.experiment_plan.MAIN_TASK`, `probe_drawer.evaluation` |
| dataset schema, storage, **and the `xi_id` split** | `probe_drawer.dataset` |
| batching, feature scaling, metrics | `probe_drawer.training.{dataloader,metrics}` |
| the Oracle landscape | `probe_drawer.analysis.{sweep,oracle}` |

The `xi_id`-level split in particular is shared *by construction*: `SplitCfg` assigns groups
by hashing the group key rather than by shuffling with a seed, so this baseline and the main
method get the same train/val/test drawers without having to coordinate a seed.

### 4.2 What this baseline owns, and must not take from the main project

Everything that *is* the method:

| Owned here | Why not shared |
|---|---|
| `PrivilegedEncoder` over `xi` | The main project's `models.psp.PrivilegedEncoder` feeds a `SuccessPredictor`. Reusing it would couple this baseline to Ours' latent and make a change to Ours silently move this baseline's numbers. |
| `AdaptationEncoder` (temporal CNN) | Ours' `AdaptationContextEncoder` is a GRU. Same inputs, same data, **different architecture** is the point — RMA²-style should keep RMA²'s architecture. |
| `ParameterHead` → `p*` | Ours has no such head; it has a success head. This is the difference being measured. |
| Stage A / Stage B trainer, config | Different objective and a different two-stage schedule from `training.trainer`. |

`probe_drawer.models.baselines.GruForceRegressor` is the **Direct Regression** baseline and
must stay distinct from this one: it has no privileged teacher and no distillation. If the two
ever end up sharing an encoder *and* a training objective, they have become the same model and
the comparison is empty — [docs/RMA2_TO_DRAWER_MAPPING.md](docs/RMA2_TO_DRAWER_MAPPING.md) §15.

### 4.3 The rule for anything in between

If this baseline needs something every method would need — a parameter space, an oracle
target, a metric — it belongs in `probe_drawer`, **not here**. A shared component living
inside one method's folder is how an unfair comparison starts. Adding it there is a change to
the main project and needs saying out loud first.

## 5. Running it

The audit needs only the main project installed — no Isaac Sim, about a second:

```bash
conda activate env_isaaclab
cd /home/zbh/Downloads/IsaacLab/9.2paper
python baselines/rma2/scripts/audit_adaptation_premise.py
python -m pytest baselines/rma2/tests -q          # 12 tests
```

`rma2` is put on the path by `tests/conftest.py` and by the script itself, so nothing has to
be installed. `pip install --no-deps -e baselines/rma2` also works and makes those bootstraps
no-ops; it is not done by default, because this baseline must not add anything to an
environment that the other methods and other agents share.

### 5.1 Reproducing the official RMA²

A **separate** conda environment — SAPIEN 2.2.2 and `numpy < 1.24` are incompatible with Isaac
Sim, so `env_isaaclab` is never touched:

```bash
bash baselines/rma2/patches/rma4rma/install_rma2.sh    # creates conda env `rma2`
```

The official code **does not run as published** — four separate defects, each observed rather
than inferred, fixed by the patches in `patches/rma4rma/` and documented in
[docs/RMA2_REPRODUCTION_REPORT.md](docs/RMA2_REPRODUCTION_REPORT.md) §21. The most serious is
that the ManiSkill2 fork's own HEAD commit misspells `set_drive_target`, so every `env.step`
raises `AttributeError`.

TurnFaucet could not be reproduced on this machine: the ManiSkill2 asset server is unreachable
through this network's proxy. `PegInsertionSide-v1` was used instead, for the reasons in §4 of
that report.

## 6. Provenance — what came from where

| Component | Origin |
|---|---|
| `MLP` encoder block, `Linear → LayerNorm → ELU` including the output layer | official, `algo/models.py:150-163` |
| Temporal CNN: `Linear(C,C)×2` then `Conv1d` kernels `(9,7,5,3)`, strides `(2,2,1,1)`, LayerNorm + ReLU | official, `algo/models.py:248-293` |
| Distillation loss `((z_hat − stopgrad(z_priv))²).mean()`, Adam 1e-4, freeze all but the adapter | official, `algo/adaptation.py:47-53`, `:81` |
| Two-stage schedule, and deployment by replacing `z_priv` with `z_hat` with no online gradient | official, `algo/policy.py:99-116` |
| **Dropped**: PPO, reward, GAE, ADR, DepthCNN, object identity embeddings, on-policy distillation | see mapping §20 |
| **New adaptation layer**: `ParameterHead` → `p*`, the 90×8 probe window, `d_z = 8`, the max-margin oracle target, Stage A/B against an offline dataset | this project |

The single most useful transfer result: the official kernel stack `(9,7,5,3)` applies to our
probe **unchanged** if the window is the probe's 1.5 s budget (90 steps at 60 Hz), giving
`90 → 41 → 18 → 14 → 12`. At the median probe length of 28 steps it is arithmetically invalid.
Mapping §8.
