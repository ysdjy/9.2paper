"""Report an out-of-distribution deployment, stratified by what the Oracle and the probe say.

One OOD number mixes three situations the feasibility pilot already separated: states the task
cannot reach in range at all, states the probe moves, and states the probe barely moves. This
prints them apart, together with the force each method chose against the force the Oracle says
was needed -- which is what distinguishes "could not act" from "could not tell".

Usage::

    python scripts/report_ood_evaluation.py
    python scripts/report_ood_evaluation.py --report outputs/logs/ood_closed_loop.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probe_drawer.analysis.ood_evaluation import summarise_ood_evaluation
from probe_drawer.utils import enable_unbuffered_stdout, git_commit, project_root

#: Gap definitions, tried in order. A report that carries only some of the method names simply
#: yields the gaps it can compute, so one script serves the main project's deployment and the
#: RMA2 baseline's, whose method labels differ.
GAPS = (
    ("teacher - ACE", "teacher (privileged)", "ACE + PSP"),
    ("ACE - GRU", "ACE + PSP", "D GRU (history)"),
    ("ACE - ridge", "ACE + PSP", "B ridge (summary)"),
    ("teacher - ACE", "teacher (xi -> landscape)", "ACE + PSP (probe -> landscape)"),
    ("ACE - StageB", "ACE + PSP (probe -> landscape)", "RMA2 Stage B (probe -> latent -> point)"),
    ("StageB - GRU", "RMA2 Stage B (probe -> latent -> point)", "D GRU (probe -> point)"),
    ("ACE - GRU", "ACE + PSP (probe -> landscape)", "D GRU (probe -> point)"),
)

ORDER = (
    "teacher (privileged)",
    "teacher (xi -> landscape)",
    "RMA2 Stage A (xi -> point)",
    "ACE + PSP",
    "ACE + PSP (probe -> landscape)",
    "RMA2 Stage B (probe -> latent -> point)",
    "D GRU (history)",
    "D GRU (probe -> point)",
    "B ridge (summary)",
    "A linear (1 feature)",
    "fixed force",
)


def main() -> None:
    enable_unbuffered_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=str, default="outputs/logs/ood_closed_loop.json")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.report)
    if not path.is_absolute():
        path = project_root() / path
    report = json.loads(path.read_text())
    summary = summarise_ood_evaluation(report, GAPS)

    print("\n" + "=" * 100)
    print(f"[oodeval] population : {report.get('population')}")
    print(f"[oodeval] seeds      : {report.get('seeds')}")
    print(f"[oodeval] task       : d_goal={report['task']['goal_displacement'] * 1000:g} mm "
          f"T={report['task']['duration']:g} s, F_peak {report['task']['peak_force_range']}")

    for name, stratum in summary["strata"].items():
        if not stratum["states"]:
            continue
        print("[oodeval]")
        print(f"[oodeval] === {name}  (n = {stratum['states']} states) -- {stratum['description']}")
        print(f"[oodeval] {'method':>38} {'reach':>8} {'+-sd':>6} {'|d-goal| med':>13} "
              f"{'F MAE':>9} {'rho(F,req)':>11} {'F chosen med':>13} {'F bias med':>11}")
        for method in ORDER:
            values = stratum["methods"].get(method)
            if not values or not values.get("episodes"):
                continue
            bias, mae, rho = (
                values["median_force_bias"],
                values.get("force_mae"),
                values.get("force_vs_required_rho"),
            )
            print(
                f"[oodeval] {method:>38} {values['reach_pp']:7.1f}% "
                f"{values['reach_sd_across_seeds']:5.1f} "
                f"{values['median_position_error_mm']:12.2f}mm "
                f"{('n/a' if mae is None else f'{mae:8.3f}N')} "
                f"{('n/a' if rho is None else f'{rho:+11.3f}')} "
                f"{values['median_chosen_force']:12.2f}N "
                f"{('n/a' if bias is None else f'{bias:+10.2f}N')}"
            )
        if stratum["gaps"]:
            print("[oodeval] " + "   ".join(f"{label} {value:+.1f} pp" for label, value in stratum["gaps"].items()))

    output = (
        Path(args.output)
        if args.output
        else project_root() / "outputs" / "logs" / "ood_evaluation_summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"source": str(path), "git_commit": git_commit(), **summary}, indent=2)
    )
    print("[oodeval]")
    print(f"[oodeval] written    : {output}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
