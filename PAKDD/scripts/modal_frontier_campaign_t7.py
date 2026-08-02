"""Modal driver for FedGRAA-v2's T7 tier (adversarial robustness) -- a
genuinely separate file/App from ``modal_frontier_campaign.py``, following
``modal_frontier_campaign_t6.py``'s own precedent
(docs/research_c/STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md §2.4
item 4): "never bolting a second dataset/workload onto an existing App" --
``PRE_COMPUTE_AUDIT_2026-07-28.md``'s "second new defect" box is the named
reason this pattern exists. T7 doesn't carry a large new dataset the way T6
does, but it does add a genuinely new cell shape (attack/defense/attacker-
fraction, not just dataset/arm/gamma/seed) and a new module
(``fedgraa.baselines_t7_adversarial``) -- isolating it here keeps a T1-T6/TC
launch's image build untouched by T7's own iteration.

**What this script duplicates from `modal_frontier_campaign.py`, and why.**
``PINNED``, ``LOCAL_REPO``/``REMOTE_REPO``/``RESULTS_VOLUME``/``REMOTE_RESULTS``,
``_bootstrap``, ``_assert_pinned_env``, ``_run``, ``_local_git_state``, and
``_completed_on_volume`` are near-identical to the other two files' -- copied
rather than imported, for the same reason ``modal_frontier_campaign_t6.py``
gives: importing the other module would construct ITS ``app = modal.App(...)``
as a side effect, and coupling this file's correctness to "that happens to be
side-effect-free" is the wrong kind of implicit dependency.

**On Windows, set UTF-8 first** (same reason as the other two drivers)::

    $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"

Usage::

    modal run scripts/modal_frontier_campaign_t7.py --verify-only
    modal run scripts/modal_frontier_campaign_t7.py --dry-run
    modal run scripts/modal_frontier_campaign_t7.py --smoke       # 2 cells, 2 rounds
    modal run --detach scripts/modal_frontier_campaign_t7.py

    # Gamma-sweep addendum (STAGE_C_MASTER_REPORT.md Sec 14 item 1 -- "gamma
    # was not swept in the T7 campaign... the most direct next measurement,
    # named but not performed"). Runs T7G instead of T7: the undefended
    # mechanism-specific-attack cross product at gamma in {0, 0.5} (T7 itself
    # already covers gamma=1.0), same 5-seed grid, same 2 attacker fractions.
    modal run --detach scripts/modal_frontier_campaign_t7.py --gamma-sweep

Then pull results down exactly as for T1/T2/T4/T6 (same Volume, same
``v2_frontier`` path)::

    modal volume get fedgraa-v2-results v2_frontier results/
    python scripts/check_stage_c_complete.py --tier T7
    python scripts/check_stage_c_complete.py --tier T7G --seeds 0 1 2 3 4
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

PINNED = {
    "tensorflow": "2.21.0",
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "pandas": "3.0.3",
    "scikit-learn": "1.9.0",
}
_IMPORT_NAME = {"scikit-learn": "sklearn"}

LOCAL_REPO = str(Path(__file__).resolve().parent.parent)
REMOTE_REPO = "/repo"
RESULTS_VOLUME = "fedgraa-v2-results"
REMOTE_RESULTS = "/results"

EXCLUDE_PARTS = {
    ".venv313", ".cache", ".pytest_cache", "__pycache__", "archive",
    ".git", ".claude", "results",
    # T7 never touches the real cross-silo dataset -- exclude it from this
    # image exactly as the main T1/T2/T4 driver excludes it, for the same
    # reason (12.79 GB, no T7 cell reads it).
    "NF-UQ-NIDS-v2",
}


def _ignore(path) -> bool:
    from pathlib import Path as _P

    p = _P(path)
    return any(part in EXCLUDE_PARTS for part in p.parts)


image_t7 = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(*[f"{k}=={v}" for k, v in PINNED.items()], "pytest>=7.0")
    .add_local_dir(LOCAL_REPO, remote_path=REMOTE_REPO, copy=True, ignore=_ignore)
)

app_t7 = modal.App("fedgraa-v2-frontier-t7")
results_volume = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)


def _bootstrap() -> None:
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    sys.path.insert(0, REMOTE_REPO)
    os.chdir(REMOTE_REPO)


def _assert_pinned_env() -> dict:
    import importlib

    resolved, wrong = {}, []
    for pkg, want in PINNED.items():
        mod = importlib.import_module(_IMPORT_NAME.get(pkg, pkg))
        got = getattr(mod, "__version__", "?")
        resolved[pkg] = got
        if got != want:
            wrong.append(f"{pkg}: pinned {want}, container has {got}")
    if wrong:
        raise RuntimeError(
            "Modal container environment does not match the pinned campaign "
            "environment:\n  " + "\n  ".join(wrong)
            + "\nRefusing to run cells against an unrecorded environment."
        )
    return resolved


def _run(payload: dict) -> dict:
    _bootstrap()
    _assert_pinned_env()

    for key in ("FEDGRAA_GIT_COMMIT", "FEDGRAA_GIT_DIRTY"):
        if payload.get(key) is not None:
            os.environ[key] = str(payload[key])

    from scripts.run_v2_frontier_sweep import (
        CellSpec,
        run_one_cell,
        should_skip_cell,
    )

    cell = CellSpec(**payload["cell"])
    label = (f"{cell.dataset} {cell.arm} g={cell.gamma} seed={cell.seed}"
             f"{cell.config_tag}")

    try:
        results_volume.reload()
    except Exception:  # noqa: BLE001 -- never fail a cell on a reload hiccup
        pass

    if should_skip_cell(cell, force=bool(payload.get("force"))):
        print(f"--- skip (already complete) {label}", flush=True)
        return {"ok": True, "skipped": True, "label": label, "path": cell.path,
                "final_all_acc": None, "final_last_n_acc": None,
                "bytes_sent": None, "failed_secagg_rounds": 0}

    print(f">>> {label}", flush=True)
    try:
        result = run_one_cell(cell)
    except Exception as exc:  # noqa: BLE001 -- one bad cell must not kill .map()
        import traceback
        traceback.print_exc()
        results_volume.commit()
        return {"ok": False, "label": label, "path": cell.path, "error": repr(exc)}

    results_volume.commit()
    return {
        "ok": True,
        "label": label,
        "path": cell.path,
        "final_all_acc": result["final_all_acc"],
        "final_last_n_acc": result["final_last_n_acc"],
        "bytes_sent": result["bytes_sent"],
        "failed_secagg_rounds": result["failed_secagg_rounds"],
    }


# Memory/timeout: T7 cells train the same model architecture as every other
# tier (no new model, no new dataset), so the same cpu=1.0/memory ceiling
# the main driver measured (COMPUTE_INFRASTRUCTURE_2026-07-27.md §5-6)
# applies; timeout is generous (SOS's O(K^2) affinity calibration and the
# Weibull fit add a small, bounded per-round cost over plain FedAvg, not a
# qualitatively different one).
@app_t7.function(image=image_t7, cpu=1.0, memory=4096, timeout=3600,
                  volumes={REMOTE_RESULTS: results_volume}, max_containers=20)
def run_cell_t7(payload: dict) -> dict:
    return _run(payload)


def _completed_on_volume() -> set:
    try:
        return {
            os.path.basename(e.path)
            for e in results_volume.listdir("v2_frontier")
            if e.path.endswith(".json")
        }
    except Exception:  # noqa: BLE001 -- volume empty / not yet created
        return set()


def _local_git_state():
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=LOCAL_REPO,
            capture_output=True, text=True, timeout=15,
        ).stdout.strip() or None
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=LOCAL_REPO,
            capture_output=True, text=True, timeout=15,
        ).stdout.strip())
        return commit, dirty
    except Exception:  # noqa: BLE001 -- never block a launch on provenance
        return None, None


@app_t7.function(image=image_t7, cpu=1.0, memory=4096, timeout=900,
                  volumes={REMOTE_RESULTS: results_volume})
def verify_env_t7() -> dict:
    _bootstrap()
    resolved = _assert_pinned_env()

    import platform

    import fedgraa  # noqa: F401
    import fedgraa.attacks  # noqa: F401
    import fedgraa.baselines_t7_adversarial  # noqa: F401

    flags = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = line.split(":", 1)[1]
                    break
    except OSError:
        pass

    probe = os.path.join(REMOTE_RESULTS, ".write_probe_t7")
    with open(probe, "w") as f:
        f.write("ok")
    results_volume.commit()
    os.remove(probe)
    results_volume.commit()

    return {
        "resolved_versions": resolved,
        "python": platform.python_version(),
        "avx512": "avx512f" in flags,
        "volume_writable": True,
    }


@app_t7.local_entrypoint()
def main(
    seeds: str = "",
    datasets: str = "",
    arm: str = "",
    rounds: int = 200,
    eval_every: int = 20,
    secagg_threshold: int = -1,
    dry_run: bool = False,
    smoke: bool = False,
    verify_only: bool = False,
    force: bool = False,
    gamma_sweep: bool = False,
):
    """T7 (default) or T7G (``--gamma-sweep``) only -- no general ``--tier``
    flag, this script only ever runs one of those two. T7 is the defense
    cross-product + the D1c undefended-v2 trend addendum, at gamma=1.0. T7G
    is the gamma-sweep addendum (:data:`scripts.run_v2_frontier_sweep.
    T7_GAMMA_SWEEP_EXTRA`, gamma in ``{0.0, 0.5}``): the same undefended
    mechanism-specific-attack cross product, at the two gammas T7 doesn't
    already cover -- answers STAGE_C_MASTER_REPORT.md Sec 14 item 1 (does
    higher gamma widen the attack surface?) without re-running the full
    defense cross-product at every gamma. Per JC-T7-18's final grid, the
    recommended ``--seeds`` for a real campaign is ``0,1,2,3,4`` (5 seeds --
    the base grid's own seed count; the D1c trend addendum's "10 seeds
    total" comes from splitting those same 5 seeds across its 2 attacker-
    fraction points x 5 = still 5 seeds each, per
    :func:`scripts.run_v2_frontier_sweep.enumerate_cells`'s own T7/T7G
    branches -- passing more than 5 seeds scales every arm uniformly, not
    just the addendum). ``--arm`` is ignored under ``--gamma-sweep``: T7G
    only ever generates the raresig arm.
    """
    sys.path.insert(0, LOCAL_REPO)
    from scripts.run_v2_frontier_sweep import (  # noqa: E402
        DEFAULT_DATASETS,
        DEFAULT_SEEDS,
        T7_ARM_NAMES,
        T7_GAMMA_SWEEP_EXTRA,
        T7_RARESIG_ARM,
        enumerate_cells,
    )

    tier = "T7G" if gamma_sweep else "T7"

    print("=" * 78)
    print("  Pre-flight: verifying the T7 image")
    print("=" * 78)
    info = verify_env_t7.remote()
    for k, v in info.items():
        print(f"  {k}: {v}")
    if not info["avx512"]:
        print("\n  NOTE: this container has no AVX512 (expected on Modal).")
    if verify_only:
        return

    sd = [int(s) for s in seeds.replace(",", " ").split()] or DEFAULT_SEEDS
    ds = [d.strip() for d in datasets.split(",") if d.strip()] or DEFAULT_DATASETS
    thr = None if secagg_threshold < 0 else secagg_threshold
    # T7G only ever generates the raresig arm (see enumerate_cells's T7G
    # branch) -- an --arm filter under --gamma-sweep would silently produce
    # zero cells for any value other than v2_raresig, so it's ignored rather
    # than left as a footgun.
    arms_filter = [] if gamma_sweep else ([arm] if arm else [])

    if smoke:
        rounds, eval_every, sd = 2, 1, [0]

    cells = enumerate_cells(
        [tier], datasets=ds, seeds=sd, rounds=rounds, eval_every=eval_every,
        data_dir=f"{REMOTE_REPO}/data", cache_dir="/tmp/.cache",
        secagg_threshold=thr, out_dir=f"{REMOTE_RESULTS}/v2_frontier",
        arms=arms_filter,
    )
    if smoke:
        keep = {"secfednids_labelflip", "v2_raresig"}
        cells = [c for c in cells if c.arm in keep][:2]

    displayed_arms = [T7_RARESIG_ARM] if gamma_sweep else (T7_ARM_NAMES if not arm else [arm])
    print("\n" + "=" * 78)
    print(f"  FedGRAA-v2 frontier campaign -- {tier}"
          + (" gamma-sweep addendum" if gamma_sweep else " (adversarial robustness)"))
    print(f"  datasets={ds}  seeds={sd}  rounds={rounds}  "
          f"secagg_threshold={thr or 'majority'}  arms={displayed_arms}"
          + (f"  gammas={T7_GAMMA_SWEEP_EXTRA}" if gamma_sweep else "  gamma=1.0"))
    print(f"  {len(cells)} cells -> Modal (cpu=1.0, <=20 concurrent)")
    print("=" * 78)

    if not force:
        done = _completed_on_volume()
        if done:
            before = len(cells)
            cells = [c for c in cells if os.path.basename(c.path) not in done]
            print(f"  resuming: {before - len(cells)} of {before} cells already "
                  f"complete on the volume, {len(cells)} to run")

    if dry_run:
        for c in cells:
            print(f"  WOULD RUN: {c.dataset} {c.arm} g={c.gamma} "
                  f"seed={c.seed}{c.config_tag} -> "
                  f"{c.path.replace(os.sep, '/')}")
        print(f"\n  {len(cells)} cell(s) would run.")
        return

    if not cells:
        print("\n  Nothing to do -- every requested cell already has a result.")
        print(f"  Pull results:  modal volume get {RESULTS_VOLUME} v2_frontier results/")
        return

    from dataclasses import asdict

    git_commit, git_dirty = _local_git_state()
    if git_commit is None:
        print("  WARNING: could not resolve a git commit locally -- results "
              "will carry no code identity.")
    else:
        print(f"  commit={git_commit[:12]}"
              + ("  (DIRTY TREE)" if git_dirty else "  (clean)"))

    payloads = [
        {
            "cell": asdict(c),
            "force": force,
            "FEDGRAA_GIT_COMMIT": git_commit,
            "FEDGRAA_GIT_DIRTY": "1" if git_dirty else "0",
        }
        for c in cells
    ]

    done, failed, skipped = 0, [], 0
    for res in run_cell_t7.map(payloads, order_outputs=False):
        done += 1
        if not res["ok"]:
            failed.append(res)
            print(f"  [FAIL {done}/{len(cells)}] {res['label']}: "
                  f"{res['error']}", flush=True)
        elif res.get("skipped"):
            skipped += 1
            print(f"  [SKIP {done}/{len(cells)}] {res['label']} "
                  f"-- already complete", flush=True)
        else:
            print(f"  [OK  {done}/{len(cells)}] {res['label']}  "
                  f"all={res['final_all_acc']:.3f} "
                  f"rare={res['final_last_n_acc']:.3f} "
                  f"bytes={res['bytes_sent']:,}"
                  + (f" aborted_rounds={res['failed_secagg_rounds']}"
                     if res["failed_secagg_rounds"] else ""),
                  flush=True)

    print("\n" + "=" * 78)
    if failed:
        print(f"  {len(failed)}/{len(cells)} cells FAILED:")
        for r in failed:
            print(f"    - {r['label']}: {r['error']}")
        print("\n  Re-run the same command; completed cells are skipped.")
    else:
        print(f"  All {len(cells)} cells completed."
              + (f"  ({skipped} were already done and skipped.)"
                 if skipped else ""))
    seeds_str = " ".join(str(s) for s in sd)
    print(f"\n  Pull results:  modal volume get {RESULTS_VOLUME} v2_frontier results/")
    print(f"  Then verify :  python scripts/check_stage_c_complete.py --tier {tier}"
          f" --seeds {seeds_str}")
    print("=" * 78)
