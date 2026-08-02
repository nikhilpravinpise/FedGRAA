"""Stage C (FedGRAA-v2 frontier campaign) completeness check.

Asserts every cell result JSON `scripts/run_v2_frontier_sweep.py` is
supposed to have produced for the requested tier(s) exists, loads, and is
internally consistent, exiting non-zero on any gap. Same idiom as
`check_stage_a_complete.py` / `check_stage_b_complete.py`: `fail()`/`ok()`,
aggregate every failure before exiting, `sys.exit(main())`.

Written directly against `run_one_cell`'s own result-dict schema
(`scripts/run_v2_frontier_sweep.py`). As of 2026-08-02 the full compute
campaign has run for real: T1-T7 and TC are complete (830 + 90 cells; see
`docs/current/STAGE_C_MASTER_REPORT.md`), so this checker's per-cell
assertions are exercised against genuine campaign output, not just the
schema `run_one_cell` writes. It is scoped to the per-cell JSON layer only --
presence and shape, not correctness. `scripts/audit_stage_c.py` (2026-08-02)
is the `audit_stage_a.py`/`audit_stage_b.py` analogue this checker itself
used to lack: it independently re-derives Stage C's reported numbers
(accuracy, significance, overhead) from the raw per-cell JSON, including the
consolidated table builders this script doesn't re-check
(`build_v2_frontier_table.py`, `build_v2_significance_table.py`,
`build_overhead_table.py`).

Usage::

    python scripts/check_stage_c_complete.py --tier T1
    python scripts/check_stage_c_complete.py --tier T1 T2 --datasets CIC-IDS-2017
    python scripts/check_stage_c_complete.py --tier T1 T2 T3 T4 T5 T6 TC --t4-m 3
    python scripts/check_stage_c_complete.py --tier T7 --seeds 0 1 2 3 4
    python scripts/check_stage_c_complete.py --tier T7G --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Failure/success banners use an em-dash. On a Windows console defaulting to
# cp1252 that raises UnicodeEncodeError (or silently mojibakes to U+FFFD)
# instead of printing the banner -- same defect class fixed in
# audit_stage_b.py; applied here too since this is the gate every doc points
# to as the authoritative green signal.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover - non-reconfigurable stream
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.check_doc_links import find_broken_links  # noqa: E402
from scripts.run_v2_frontier_sweep import (  # noqa: E402
    DEFAULT_DATASETS,
    DEFAULT_OUT_DIR,
    DEFAULT_SEEDS,
    CellSpec,
    enumerate_cells,
)

REQUIRED_CELL_FIELDS = {
    "dataset", "arm", "gamma", "seed", "grad_refresh_every", "rounds",
    "all_acc", "last_n_acc", "zeta", "final_all_acc", "final_last_n_acc",
    "bytes_sent", "comm_rounds",
}

FAILURES: list = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


# --------------------------------------------------------------------------- #
# Per-cell result JSON
# --------------------------------------------------------------------------- #
def check_cell_json(cell: CellSpec) -> bool:
    label = f"{cell.dataset}/{cell.arm}/g{cell.gamma}/seed{cell.seed}"
    if not os.path.exists(cell.path):
        fail(f"{label}: missing {cell.path}")
        return False
    try:
        with open(cell.path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        fail(f"{label}: {cell.path} failed to load ({exc})")
        return False

    missing = REQUIRED_CELL_FIELDS - set(data.keys())
    if missing:
        fail(f"{label}: {cell.path} missing fields {sorted(missing)}")
        return False

    if not isinstance(data["final_all_acc"], (int, float)):
        fail(f"{label}: final_all_acc is not numeric ({data['final_all_acc']!r})")
        return False
    if not isinstance(data["final_last_n_acc"], (int, float)):
        fail(f"{label}: final_last_n_acc is not numeric ({data['final_last_n_acc']!r})")
        return False

    # ⚠️ Corrected: this check originally asserted v1_projected must NOT carry
    # bytes_sent/comm_rounds, written before run_one_cell's "Billed for BOTH
    # arms now" fix landed (see that function's own comment in
    # run_v2_frontier_sweep.py) -- v1_projected's FederatedTrainer instruments
    # its own bytes_sent (fedgraa/trainer.py), and every T3/T5 external
    # baseline gets an analytically-computed value (same file, the
    # BASELINE_ARM_CLASSES branch), since none of them do SecAgg either. Every
    # arm must have both populated -- the frontier's x-axis needs a value at
    # every point, not only the v2 arms.
    if data["bytes_sent"] is None or data["comm_rounds"] is None:
        fail(f"{label}: cell missing bytes_sent/comm_rounds instrumentation")
        return False

    if data["dataset"] != cell.dataset or data["arm"] != cell.arm or data["seed"] != cell.seed:
        fail(f"{label}: {cell.path} identity fields don't match its own file name/cell spec")
        return False

    ok(f"{label}: complete and internally consistent")
    return True


def check_tier_coverage(tiers: list, datasets: list, seeds: list, out_dir: str,
                         t4_m: int = 1) -> None:
    cells = enumerate_cells(tiers, datasets=datasets, seeds=seeds, out_dir=out_dir,
                             t4_m=t4_m)
    n_complete = 0
    for cell in cells:
        if check_cell_json(cell):
            n_complete += 1
    arms = sorted({c.arm for c in cells})
    print(f"\n  {n_complete}/{len(cells)} cells complete for tier(s) {tiers}, "
          f"datasets={datasets}, arms={arms}, seeds={seeds}")


def check_doc_links() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    broken = find_broken_links(repo_root)
    if broken:
        for md_file, target in broken:
            fail(f"doc link: {md_file} -> {target} does not resolve")
    else:
        ok("doc links: all relative markdown links resolve")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", nargs="+", default=["T1"],
                    choices=["T1", "T2", "T3", "T4", "T5", "T6", "TC", "T7", "T7G"])
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--t4-m", type=int, default=1,
        help="Which v2 refresh interval T4 was run at -- must match the "
             "--t4-m value the campaign itself was launched with "
             "(scripts/run_v2_frontier_sweep.py's own default), or every T4 "
             "cell will be reported missing under the wrong filename.",
    )
    args = ap.parse_args()

    print("=" * 80)
    print(f"  Stage C completeness check -- tier(s)={args.tier}")
    print("=" * 80)

    preview_cells = enumerate_cells(args.tier, datasets=args.datasets, seeds=args.seeds,
                                     out_dir=args.out_dir, t4_m=args.t4_m)
    preview_arms = sorted({c.arm for c in preview_cells})
    print(f"\n-- Per-cell result JSONs ({len(preview_arms)} arm(s): "
          f"{', '.join(preview_arms)}) --")
    check_tier_coverage(args.tier, args.datasets, args.seeds, args.out_dir, t4_m=args.t4_m)

    print("\n-- Documentation links --")
    check_doc_links()

    print("\n" + "=" * 80)
    if FAILURES:
        print(f"  STAGE C ({'/'.join(args.tier)}) INCOMPLETE — {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"    - {f}")
        print("=" * 80)
        return 1
    print(f"  STAGE C ({'/'.join(args.tier)}) COMPLETE — all checks passed.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
