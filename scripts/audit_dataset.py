"""Phase 11E -- audit a dataset, and refuse to bless one that fails a gate.

No simulator. Reads a dataset, takes the grouped split, runs every check in
``probe_drawer.dataset.audit``, writes ``audit.json`` and ``splits.json`` next to the data,
and exits non-zero if a gate fails so a pipeline cannot walk past it.

Distributional findings are reported, not gated: a skewed positive rate is information about
the task. Structural and leakage findings are gates.

Usage::

    python scripts/audit_dataset.py --dataset outputs/dataset_smoke
    python scripts/audit_dataset.py --dataset outputs/dataset_v0 --report docs/DATASET_V0.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from probe_drawer.dataset import DatasetStore, SplitCfg, split_samples
from probe_drawer.dataset.audit import audit_dataset
from probe_drawer.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True, help="Dataset directory.")
    parser.add_argument("--split-level", type=str, default="xi_id", help="Group level to split on.")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--salt", type=str, default="phase11", help="Split salt.")
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.is_absolute():
        root = project_root() / root
    store = DatasetStore(root)

    cfg = SplitCfg(
        level=args.split_level,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        salt=args.salt,
    )
    split = split_samples(store.load_samples(), cfg)
    report = audit_dataset(store, split)
    report["split_cfg"] = {
        "level": cfg.level,
        "train_fraction": cfg.train_fraction,
        "val_fraction": cfg.val_fraction,
        "salt": cfg.salt,
    }

    store.write_splits(
        {
            "cfg": report["split_cfg"],
            "groups": split.groups,
            "counts": split.counts(),
        }
    )
    store.write_audit(report)
    _print(report)

    if not report["all_gates_passed"]:
        print(f"[audit] FAILED gates: {report['failed_gates']}")
        print("[audit] The dataset must not be trained on until these pass.")
        print("=" * 78 + "\n")
        sys.exit(1)
    print("[audit] all gates passed")
    print("=" * 78 + "\n")


def _print(report: dict) -> None:
    distributions = report["distributions"]
    checks = report["checks"]
    print("\n" + "=" * 78)
    print(f"[audit] dataset  : {report['dataset']['root']}")
    print(
        f"[audit] version  : {report['manifest'].get('dataset_version')} "
        f"(schema {report['manifest'].get('schema_version')}, "
        f"commit {str(report['manifest'].get('git_commit'))[:9]})"
    )
    print(
        f"[audit] size     : {distributions['rows']} candidate rows, "
        f"{distributions['probes']} probes, {distributions['hidden_states']} hidden states"
    )
    print(
        f"[audit] labels   : {distributions['positive_fraction'] * 100:.2f} % "
        f"{distributions['label']}, {distributions['strict_positive_fraction'] * 100:.2f} % strict, "
        f"{distributions['invalid_fraction'] * 100:.2f} % invalid {distributions['invalid_reasons'] or ''}"
    )

    lengths = distributions["sequence_length"]
    print(
        f"[audit] histories: {lengths['min']}-{lengths['max']} steps "
        f"(mean {lengths['mean']:.1f}, {lengths['distinct']} distinct lengths)"
    )
    if distributions["xi_ranges"]:
        print(
            "[audit] xi       : "
            + ", ".join(
                f"{name} {low:.2f}-{high:.2f}" for name, (low, high) in distributions["xi_ranges"].items()
            )
        )
    if distributions.get("hidden_states_with_incomplete_xi"):
        print(f"[audit] xi       : {distributions['hidden_states_with_incomplete_xi']} INCOMPLETE hidden states")
    print(f"[audit]            mu_d <= mu_s everywhere: {distributions['dynamic_friction_never_exceeds_static']}")

    print("[audit]")
    print("[audit] positives per probe (the candidate-budget question):")
    print(f"[audit]   {distributions['positives_per_probe']}")
    total = distributions["probes"]
    print(
        f"[audit]   none: {distributions['probes_with_no_positive']} "
        f"({distributions['probes_with_no_positive'] / max(total, 1) * 100:.1f} %)  "
        f">=1: {distributions['probes_with_at_least_one']} "
        f"({distributions['probes_with_at_least_one'] / max(total, 1) * 100:.1f} %)  "
        f">=2: {distributions['probes_with_at_least_two']} "
        f"({distributions['probes_with_at_least_two'] / max(total, 1) * 100:.1f} %)"
    )

    print("[audit]")
    print("[audit] success against force:")
    for row in distributions["success_vs_force"]:
        bar = "#" * int(round(row["positive_fraction"] * 40))
        print(
            f"[audit]   {row['low']:5.2f}-{row['high']:5.2f} N  n={row['rows']:5d}  "
            f"{row['positive_fraction'] * 100:5.1f} % {bar}"
        )

    split = checks["split_has_no_leakage"]
    if "counts" in split:
        print("[audit]")
        print(f"[audit] split ({split['level']}): {json.dumps(split['counts'])}")
        print(
            "[audit]   positive fraction per subset: "
            + ", ".join(f"{name} {value * 100:.2f} %" for name, value in split["positive_fraction"].items())
        )
        if split.get("empty_subsets"):
            print(
                f"[audit]   NOTE: {split['empty_subsets']} empty -- too few groups for these "
                "fractions; metrics on those subsets are meaningless"
            )

    correlation = checks["branch_index_decorrelated_from_force"]
    print("[audit]")
    print(
        f"[audit] force vs branch position over {correlation['probes_checked']} probes "
        f"x {correlation['candidates_per_probe']} candidates:"
    )
    print(
        f"[audit]   mean corr {correlation['mean_correlation']:+.4f} = "
        f"{correlation['sigmas']:.2f} sigma (null SE {correlation['null_standard_error']:.4f}, "
        f"gate {correlation['tolerance']:.4f}); worst single probe "
        f"|corr| {correlation['max_abs_correlation']:.4f}"
    )

    print("[audit]")
    print("[audit] gates:")
    for name in report["gates"]:
        print(f"[audit]   {'pass' if checks[name]['passes'] else 'FAIL':>4}  {name}")


if __name__ == "__main__":
    main()
