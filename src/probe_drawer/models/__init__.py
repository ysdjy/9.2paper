"""Models: a privileged teacher, a deployable student, and a shared success head.

Nothing here imports Isaac Lab or reads a dataset from disk. The privileged/deployable split
is structural rather than conventional -- see :mod:`probe_drawer.models.psp`.
"""

from probe_drawer.models.psp import (
    AdaptationContextEncoder,
    PrivilegedEncoder,
    PspCfg,
    StudentModel,
    SuccessPredictor,
    TeacherModel,
    build_student,
    build_teacher,
)

__all__ = [
    "AdaptationContextEncoder",
    "PrivilegedEncoder",
    "PspCfg",
    "StudentModel",
    "SuccessPredictor",
    "TeacherModel",
    "build_student",
    "build_teacher",
]
