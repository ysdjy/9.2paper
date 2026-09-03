"""The success predictor, and the two encoders that feed it.

The shape of the whole thing::

    xi ------------> E_priv ---> z_priv --\\
                                            >--- PSP(z, F_candidate, post_probe) -> P(success)
    probe history -> ACE ------> z_ace ---/

Three deliberate choices.

**The head is shared in structure, not in weights.** The teacher (``E_priv + PSP``) and the
student (``ACE + PSP``) use the same ``SuccessPredictor`` class with the same input contract,
which is what makes their success landscapes comparable and distillation meaningful. They are
separate instances: a shared head would let the teacher's gradients reshape what the student
must fit.

**The student cannot see ``xi``.** Not by convention -- :class:`AdaptationContextEncoder`
takes a history and a length and has no parameter that could carry the hidden state. The
teacher is the only thing in the package that accepts ``xi``, and it says so in its name.

**Small.** ``z`` is 16-dimensional and the hidden layers are 96 wide. The task's spread is
driven almost entirely by one physical quantity (dynamic friction, `docs/ORACLE_LANDSCAPE.md`),
so capacity is not the binding constraint and a large model would only make the comparison
against a scalar baseline harder to interpret.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

__all__ = [
    "CONDITION_DIM",
    "CONDITION_FIELDS",
    "AdaptationContextEncoder",
    "PrivilegedEncoder",
    "PspCfg",
    "PspPrediction",
    "SuccessPredictor",
    "build_student",
    "build_teacher",
]

#: The head's conditioning inputs besides ``z``, in the order they are concatenated.
#:
#: The last two are the **task condition**. Setting V1 searches ``F_peak`` and nothing else --
#: ``T_goal`` is something the task hands the robot, not something the model chooses
#: (``docs/DECISIONS.md`` D044) -- and putting the condition in the input is what makes that
#: distinction structural rather than a convention.
#:
#: **Honestly**: within Dataset v1 both are constant, so they contribute nothing the network
#: can learn from; a constant input is 2 extra weights and no information. They are here so
#: that the model's contract is "given *this* task, would this force work?" rather than
#: "would this force work?", which is the contract a second task distance would need. Phase 11
#: left them out for the opposite and equally defensible reason.
CONDITION_FIELDS = (
    "candidate_peak_force",
    "post_probe_displacement",
    "post_probe_velocity",
    "goal_displacement",
    "duration",
)

#: Width of the conditioning vector.
CONDITION_DIM = len(CONDITION_FIELDS)


@dataclass(frozen=True)
class PspCfg:
    """Sizes for the teacher, the student and the shared head.

    Args:
        z_dim: Context width. Small on purpose -- see the module docstring.
        hidden: Width of the head's hidden layers.
        gru_hidden: Width of the student's recurrent state.
        gru_layers: Recurrent depth. One layer unless an ablation says otherwise.
        dropout: Applied inside the head only.
    """

    z_dim: int = 16
    hidden: int = 96
    gru_hidden: int = 64
    gru_layers: int = 1
    dropout: float = 0.1

    def as_dict(self) -> dict:
        return {
            "z_dim": self.z_dim,
            "hidden": self.hidden,
            "gru_hidden": self.gru_hidden,
            "gru_layers": self.gru_layers,
            "dropout": self.dropout,
            "condition_dim": CONDITION_DIM,
            "condition_fields": list(CONDITION_FIELDS),
        }


class PspPrediction(NamedTuple):
    """What the head produces for one batch.

    ``logit`` is the task output and the only one used to select a force. The other two are
    **auxiliary**: regressing what the execution will actually do gives the shared trunk a
    dense target alongside a binary one, and gives a reader something to inspect when a
    prediction is wrong -- a model that says "will fail" is less informative than one that
    says "will stop 14 mm short".

    Attributes:
        logit: ``(batch,)`` logit of ``P(reach_success)``.
        displacement: ``(batch,)`` predicted :math:`d_\text{total}(T)` (m).
        velocity: ``(batch,)`` predicted :math:`v(T)` (m/s).
    """

    logit: torch.Tensor
    displacement: torch.Tensor
    velocity: torch.Tensor


class PrivilegedEncoder(nn.Module):
    """``xi -> z``. The teacher's eyes, and the only module that takes the hidden state.

    Its job is to establish an upper bound: if a model that is *told* the four hidden values
    cannot predict the success landscape, then the landscape is not learnable from a probe
    either and nothing downstream is worth training (the Phase 11 gate).
    """

    def __init__(self, xi_dim: int = 4, cfg: PspCfg | None = None) -> None:
        super().__init__()
        cfg = cfg or PspCfg()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(xi_dim, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.z_dim),
        )

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """Args: ``xi`` of shape ``(batch, 4)``. Returns ``(batch, z_dim)``."""
        return self.net(xi)


class AdaptationContextEncoder(nn.Module):
    """``probe history -> z``. The deployable encoder.

    A GRU over the variable-length recording, read out at each sequence's **true last step**.
    The lengths are used through ``pack_padded_sequence``, so padding is never visited: the
    returned state is what the network would produce if that one sequence had been run alone.
    That equivalence is asserted in the tests, because a mask applied slightly wrongly is a
    bug that degrades results without failing anything.
    """

    def __init__(self, num_channels: int, cfg: PspCfg | None = None) -> None:
        super().__init__()
        cfg = cfg or PspCfg()
        self.cfg = cfg
        self.num_channels = num_channels
        self.gru = nn.GRU(
            input_size=num_channels,
            hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
        )
        self.project = nn.Linear(cfg.gru_hidden, cfg.z_dim)

    def forward(self, history: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Args:
            history: ``(batch, time, channel)``, zero-padded.
            lengths: ``(batch,)`` true lengths, on the CPU.

        Returns:
            ``(batch, z_dim)``.
        """
        if history.shape[-1] != self.num_channels:
            raise ValueError(
                f"encoder was built for {self.num_channels} channels, got {history.shape[-1]}."
            )
        packed = pack_padded_sequence(
            history, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, final = self.gru(packed)
        return self.project(final[-1])


class SuccessPredictor(nn.Module):
    """``(z, F_candidate, post_probe, d_goal, T_goal) -> logit P(reach_success), d_hat, v_hat``.

    Takes a *logit* rather than a probability so the loss can be the numerically stable
    ``binary_cross_entropy_with_logits``, and so distillation can match logits directly.

    One trunk, two heads. The auxiliary regression shares every parameter with the classifier
    up to the last layer, which is the point: it is there to shape the representation, not to
    be a second model.
    """

    def __init__(self, cfg: PspCfg | None = None) -> None:
        super().__init__()
        cfg = cfg or PspCfg()
        self.cfg = cfg
        self.trunk = nn.Sequential(
            nn.Linear(cfg.z_dim + CONDITION_DIM, cfg.hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(cfg.hidden, 1)
        self.regressor = nn.Linear(cfg.hidden, 2)

    def predict(
        self,
        context: torch.Tensor,
        candidate_force: torch.Tensor,
        post_probe: torch.Tensor,
        task_condition: torch.Tensor,
    ) -> PspPrediction:
        """Args:
            context: ``(batch, z_dim)``.
            candidate_force: ``(batch,)``.
            post_probe: ``(batch, 2)`` -- displacement and velocity where the execution starts.
            task_condition: ``(batch, 2)`` -- ``d_goal`` and ``T_goal``.

        Returns:
            A :class:`PspPrediction`.
        """
        conditions = torch.cat([candidate_force.unsqueeze(-1), post_probe, task_condition], dim=-1)
        if conditions.shape[-1] != CONDITION_DIM:
            raise ValueError(
                f"expected {CONDITION_DIM} conditioning inputs {CONDITION_FIELDS}, "
                f"got {conditions.shape[-1]}."
            )
        features = self.trunk(torch.cat([context, conditions], dim=-1))
        auxiliary = self.regressor(features)
        return PspPrediction(
            logit=self.classifier(features).squeeze(-1),
            displacement=auxiliary[..., 0],
            velocity=auxiliary[..., 1],
        )

    def forward(
        self,
        context: torch.Tensor,
        candidate_force: torch.Tensor,
        post_probe: torch.Tensor,
        task_condition: torch.Tensor,
    ) -> torch.Tensor:
        """The task output alone, ``(batch,)`` logits. See :meth:`predict` for the rest."""
        return self.predict(context, candidate_force, post_probe, task_condition).logit


class TeacherModel(nn.Module):
    """``E_priv + PSP``. Reads ``xi``; never deployed."""

    def __init__(self, cfg: PspCfg | None = None, xi_dim: int = 4) -> None:
        super().__init__()
        cfg = cfg or PspCfg()
        self.cfg = cfg
        self.encoder = PrivilegedEncoder(xi_dim, cfg)
        self.head = SuccessPredictor(cfg)

    def forward(self, batch) -> torch.Tensor:
        return self.predict(batch).logit

    def predict(self, batch) -> PspPrediction:
        return self.head.predict(
            self.context(batch), batch.candidate_force, batch.post_probe, batch.task_condition
        )

    def context(self, batch) -> torch.Tensor:
        return self.encoder(batch.xi)


class StudentModel(nn.Module):
    """``ACE + PSP``. Reads only the probe recording and the deployable post-probe state.

    ``forward`` takes the whole batch for symmetry with the teacher, but touches only
    ``history``, ``lengths``, ``candidate_force``, ``post_probe`` and ``task_condition`` --
    every one of them deployable, the last two being numbers the task itself hands the robot.
    It has no path to ``batch.xi``; the test suite asserts that a batch with corrupted ``xi`` produces
    identical output.
    """

    def __init__(self, num_channels: int, cfg: PspCfg | None = None) -> None:
        super().__init__()
        cfg = cfg or PspCfg()
        self.cfg = cfg
        self.num_channels = num_channels
        self.encoder = AdaptationContextEncoder(num_channels, cfg)
        self.head = SuccessPredictor(cfg)

    def forward(self, batch) -> torch.Tensor:
        return self.predict(batch).logit

    def predict(self, batch) -> PspPrediction:
        return self.head.predict(
            self.context(batch), batch.candidate_force, batch.post_probe, batch.task_condition
        )

    def context(self, batch) -> torch.Tensor:
        return self.encoder(batch.history, batch.lengths)


def build_teacher(cfg: PspCfg | None = None) -> TeacherModel:
    return TeacherModel(cfg=cfg)


def build_student(num_channels: int, cfg: PspCfg | None = None) -> StudentModel:
    return StudentModel(num_channels=num_channels, cfg=cfg)
