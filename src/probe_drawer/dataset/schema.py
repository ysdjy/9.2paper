"""The formal training-sample schema, and the identifiers that keep splits honest.

One sample is one *question*: given what a probe measured, would this candidate peak force
land the drawer on the goal? So a sample carries the probe's recording, the state the probe
left behind, one candidate force, the task, and the outcome::

    (xi, probe_history, probe_summary, post_probe_state,
     candidate_peak_force, task_condition = (d_goal, T_goal),
     d_total(T), v(T), position_error, reach_success, stable_success, validity)

Only ``candidate_peak_force`` varies within a probe's rows: Setting V1 searches the force and
nothing else, and ``T_goal`` is a task condition rather than an adapted parameter
(``docs/DECISIONS.md`` D044).

``xi`` is present and is labelled ``SIM_ONLY_PRIVILEGED``. It is what makes an upper-bound
oracle and a per-dimension error analysis possible, and it must never reach a model's input;
:func:`model_input_fields` is the list that may.

**Why the identifiers exist.** A single probe is expensive, so one probe is naturally paired
with many candidate forces. Those rows are not independent: they share a probe recording, a
hidden state, and a post-probe state. Splitting rows at random puts near-duplicates of a
training row into the test set, and the reported error becomes a measure of memorisation.
Three nested groups are recorded so a split can be taken at the right level:

``xi_id``
    The hidden state. Splitting here answers "does this work on a drawer never seen?", which
    is the question the paper asks. This is the default.
``probe_id``
    One physical probe episode. Every candidate sharing a probe shares this. The *minimum*
    admissible split level -- coarser than ``xi_id`` but still leak-free with respect to the
    probe recording itself.
``candidate_id``
    The row. Never a split level; it exists for deduplication and for joining back to logs.

See ``docs/DATASET_SCHEMA.md`` for the field-by-field reference and
``docs/DECISIONS.md`` D031.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from probe_drawer.observations import validate_model_input

__all__ = [
    "SPLIT_LEVELS",
    "XI_DIMENSIONS",
    "TrainingSample",
    "candidate_id",
    "model_input_fields",
    "probe_id",
    "validate_probe_history",
    "xi_id",
]

#: The group levels a split may legitimately be taken at, from strictest to loosest.
#: ``candidate_id`` is deliberately absent -- see the module docstring.
SPLIT_LEVELS = ("xi_id", "probe_id")

#: Order of the four hidden dimensions in every identifier and serialisation.
XI_DIMENSIONS = ("mass", "static_friction", "dynamic_friction", "damping")


def _digest(*parts: object) -> str:
    """A short, stable, content-addressed identifier.

    Content-addressed rather than a counter so that two datasets built in different runs, or
    in different batch orders, agree on which rows describe the same thing. Twelve hex
    characters is 48 bits, which is far more than the ``10^5``-row datasets this schema is
    for.
    """
    payload = "|".join(f"{part:.6g}" if isinstance(part, float) else str(part) for part in parts)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def xi_id(xi: dict) -> str:
    """Identifier for a hidden state, from its four values."""
    missing = [name for name in XI_DIMENSIONS if name not in xi]
    if missing:
        raise ValueError(f"a hidden state needs all four dimensions; missing {missing}")
    return _digest("xi", *(float(xi[name]) for name in XI_DIMENSIONS))


def probe_id(xi: dict, episode_index: int, probe_task: dict) -> str:
    """Identifier for one probe episode.

    Args:
        xi: The hidden state the probe ran against.
        episode_index: Which repeat this is. Two probes of the same drawer with the same
            parameters are still different episodes -- they differ by simulator noise -- and
            must be distinguishable, or repeats would collapse into one group.
        probe_task: The probe's parameters, so a re-calibrated probe never shares an
            identifier with the old one.
    """
    return _digest("probe", xi_id(xi), episode_index, *sorted(f"{k}={v:.6g}" for k, v in probe_task.items()))


def candidate_id(probe: str, peak_force: float, duration: float, goal_displacement: float) -> str:
    """Identifier for one row: a probe, a candidate force, and the task it is judged against."""
    return _digest("candidate", probe, peak_force, duration, goal_displacement)


@dataclass(frozen=True)
class TrainingSample:
    """One row of the formal dataset.

    Attributes:
        candidate_id: This row. Unique.
        probe_id: The probe episode this row's evidence came from.
        xi_id: The hidden state.
        xi: The four hidden values. ``SIM_ONLY_PRIVILEGED``.
        probe_history: The probe's recorded time series, as ``{channel: [values]}``. Only
            ``DEPLOYABLE`` channels belong here.
        probe_summary: Scalar features of the probe, from
            :func:`~probe_drawer.analysis.probe_features.extract_features`.
        post_probe_state: Where the probe left the drawer, at the moment the execution
            starts: ``displacement`` (m, from the task's start) and ``velocity`` (m/s). This
            is state, not a command, and it is deployable -- a robot knows where it has
            already pulled the handle to.
        candidate_peak_force: :math:`F_\\text{peak}` this row asks about (N).
        branch_index: Where in its probe's candidate sweep this execution ran. Recorded
            because branching drifts slightly with sweep position, and the sweep order is
            shuffled to keep that drift uncorrelated with force; storing the index is what
            lets the audit verify the decorrelation instead of trusting it
            (``docs/COUNTERFACTUAL_BRANCHING.md`` 5.2).
        duration: :math:`T_\\text{goal}` (s).
        goal_displacement: :math:`d_\\text{goal}` (m), measured from *before* the probe.
        final_total_displacement: :math:`d_\\text{total}(T)` (m).
        final_velocity: :math:`v(T)` (m/s).
        success: The strict label -- position, terminal velocity and validity all held.
            Unchanged in name and meaning since Dataset v0, which was generated with it.
        reach_success: **Primary metric** from Setting V1 on: position and validity held,
            regardless of terminal velocity (``docs/DECISIONS.md`` D046). ``None`` in a
            Dataset v0 row, which predates the split -- a v0 negative could have failed on
            either term and the row does not record which, so it is left unknown rather than
            guessed.
        stable_success: **Secondary metric**: ``reach_success`` and the terminal velocity.
            Equal to ``success`` where both are recorded; ``None`` in a v0 row.
        termination_reason: How the execution ended, e.g. ``"duration_completed"`` or
            ``"safety_abort"``. ``None`` in a v0 row.
        valid: Whether the episode stayed inside the operating region. An invalid row is
            evidence about the rig, not about the drawer, and must be dropped before
            training -- it is kept in the file so the drop is auditable.
        invalid_reasons: Why not, if not.
        protocol: ``"sequential"``. A reset row is not a training sample; the field is
            explicit so a mixed file cannot be trained on by accident.
    """

    candidate_id: str
    probe_id: str
    xi_id: str
    xi: dict
    probe_history: dict
    probe_summary: dict
    post_probe_state: dict
    candidate_peak_force: float
    branch_index: int
    duration: float
    goal_displacement: float
    final_total_displacement: float
    final_velocity: float
    success: bool
    valid: bool
    reach_success: bool | None = None
    stable_success: bool | None = None
    termination_reason: str | None = None
    invalid_reasons: list[str] = field(default_factory=list)
    protocol: str = "sequential"

    def __post_init__(self) -> None:
        if self.protocol != "sequential":
            raise ValueError(
                f"a training sample must come from the sequential protocol, not {self.protocol!r}. "
                "The reset Oracle is preliminary verification only (docs/DECISIONS.md D026)."
            )
        for name in ("displacement", "velocity"):
            if name not in self.post_probe_state:
                raise ValueError(f"post_probe_state needs '{name}'; got {sorted(self.post_probe_state)}")
        # Nested by definition (D046). A row that claims otherwise was assembled wrongly, and
        # finding out at training time would mean re-generating the dataset.
        if self.stable_success and self.reach_success is False:
            raise ValueError(
                f"row {self.candidate_id} is stable_success without reach_success; "
                "stable_success is reach_success plus the terminal-velocity condition."
            )
        if self.stable_success is not None and bool(self.stable_success) != bool(self.success):
            raise ValueError(
                f"row {self.candidate_id} has stable_success={self.stable_success} but "
                f"success={self.success}; they are the same label under two names."
            )

    @property
    def position_error(self) -> float:
        r"""Signed :math:`d_\text{total}(T) - d_\text{goal}` (m); positive means overshoot.

        Derived rather than stored, so it cannot disagree with the two values it comes from.
        """
        return self.final_total_displacement - self.goal_displacement

    @property
    def task_condition(self) -> tuple[float, float]:
        """:math:`(d_\text{goal}, T_\text{goal})` -- what the task asks, not what is adapted.

        Setting V1 searches ``candidate_peak_force`` and nothing else; the duration is a
        condition handed to the model, which is why it sits here rather than among the
        candidate's parameters.
        """
        return (self.goal_displacement, self.duration)

    def as_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "probe_id": self.probe_id,
            "xi_id": self.xi_id,
            "xi": dict(self.xi),
            "probe_history": {name: list(values) for name, values in self.probe_history.items()},
            "probe_summary": dict(self.probe_summary),
            "post_probe_state": dict(self.post_probe_state),
            "candidate_peak_force": self.candidate_peak_force,
            "branch_index": self.branch_index,
            "duration": self.duration,
            "goal_displacement": self.goal_displacement,
            "final_total_displacement": self.final_total_displacement,
            "final_velocity": self.final_velocity,
            "success": self.success,
            "valid": self.valid,
            "reach_success": self.reach_success,
            "stable_success": self.stable_success,
            "termination_reason": self.termination_reason,
            "invalid_reasons": list(self.invalid_reasons),
            "protocol": self.protocol,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> TrainingSample:
        """Rebuild a row. Dataset v0 payloads load unchanged, with the newer fields ``None``."""
        return cls(**payload)


def model_input_fields() -> tuple[str, ...]:
    """The sample fields a model may read.

    ``xi`` is excluded because it is privileged. Everything else here is either a deployable
    measurement or a number the task itself hands the robot.
    """
    return (
        "probe_history",
        "probe_summary",
        "post_probe_state",
        "candidate_peak_force",
        "duration",
        "goal_displacement",
    )


def validate_probe_history(history: dict) -> None:
    """Refuse a probe history that carries a channel a robot could not measure.

    The registry already knows each channel's deployability, so this is only the application
    of that rule at the dataset boundary -- the last point before a model sees the data
    (D017).
    """
    validate_model_input(tuple(history))
