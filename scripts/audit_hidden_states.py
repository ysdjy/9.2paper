"""Phase 9C -- probe every hidden-state candidate this simulator exposes.

Writes each candidate physical quantity, reads it back, restores it, and prints the table
with the reviewed claim about what the quantity means and whether it belongs in the main
paper's four-dimensional xi. The environment is left exactly as it was found and nothing is
stepped, so the audit cannot perturb any later experiment.

The result is the data behind ``docs/HIDDEN_STATE_AUDIT.md``.

Usage::

    python scripts/audit_hidden_states.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import json  # noqa: E402

from probe_drawer.analysis import run_hidden_state_audit  # noqa: E402
from probe_drawer.envs import XI_FIELDS  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import enable_unbuffered_stdout, project_root  # noqa: E402


def main() -> None:
    enable_unbuffered_stdout()

    system = PullSystem.build(PullSystemCfg(num_envs=1, device=args_cli.device))
    system.reset()
    report = run_hidden_state_audit(system.env)

    print("\n" + "=" * 78)
    print(f"[hidden-audit] candidates probed : {len(report['rows'])}")
    print(f"[hidden-audit] main-paper xi     : {list(XI_FIELDS)}")
    print("[hidden-audit]")
    print(f"[hidden-audit] {'parameter':>26} {'writable':>9} {'readback':>9} {'visible':>8} {'role':>16}")
    for row in report["rows"]:
        print(
            f"[hidden-audit] {row['name']:>26} {str(row['writable']):>9} {str(row['readable_back']):>9} "
            f"{str(row['deployment_visible']):>8} {row['role']:>16}"
        )
    print("[hidden-audit]")
    for role, names in sorted(report["by_role"].items()):
        print(f"[hidden-audit] {role:>16}: {names}")
    print(f"[hidden-audit] verified writable : {len(report['writable_and_verified'])}")
    if report["write_accepted_but_not_read_back"]:
        print(
            "[hidden-audit] WRITE SILENTLY DISCARDED by the simulator: "
            f"{report['write_accepted_but_not_read_back']}"
        )
    print("[hidden-audit]")
    for row in report["rows"]:
        print(f"[hidden-audit] {row['name']}: {row['detail']}")

    output = project_root() / "outputs" / "logs" / "hidden_state_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str))
    print(f"[hidden-audit] report written    : {output}")
    print("=" * 78 + "\n")

    system.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
