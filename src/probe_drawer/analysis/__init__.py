"""Offline analysis of recorded episodes and of the simulator's own capabilities.

The force-channel and sweep analyses are pure: they consume results and histories that were
already recorded, so they need no simulator and can be unit-tested. The hidden-state audit
is the exception -- it probes a live environment by design -- and is kept here because it is
analysis rather than part of the experiment pipeline.
"""

from .force_channel_analysis import (
    AUDIT_CASES,
    END_STOP_CASE,
    ForceAuditCase,
    analyse_end_stop_episode,
    analyse_force_channels,
)
from .hidden_state_audit import CANDIDATES, AuditVerdict, HiddenStateCandidate, PaperRole, run_hidden_state_audit

__all__ = [
    "AUDIT_CASES",
    "CANDIDATES",
    "END_STOP_CASE",
    "AuditVerdict",
    "ForceAuditCase",
    "HiddenStateCandidate",
    "PaperRole",
    "analyse_end_stop_episode",
    "analyse_force_channels",
    "run_hidden_state_audit",
]
