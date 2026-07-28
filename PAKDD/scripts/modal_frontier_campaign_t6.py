"""Modal driver for FedGRAA-v2's T6 tier (the real NF-UQ-NIDS-v2 cross-silo
benchmark) -- a genuinely separate file/App from
[`modal_frontier_campaign.py`](modal_frontier_campaign.py), not a variant of it.

**Why a separate file, not a second image/function on the T1/T2/T4 App.**
The first version of this put T6's image (which includes the 12.79 GB
NF-UQ-NIDS-v2 file) on the SAME ``modal.App`` as T1/T2/T4. That was wrong,
confirmed by two direct measurements rather than assumed correct after the
fact:

1. A plain ``--tier T1 --smoke`` run on that combined App, which should
   finish in ~2 minutes, instead sat for **15+ minutes** building an image
   containing the 12.79 GB directory -- even though the T1 smoke test never
   touches NF-UQ-NIDS-v2. Modal's ``app.run()`` (which ``modal run`` enters)
   syncs **every function registered on the App it activates**, not only the
   ones the current command's tier actually needs.
2. A minimal two-``App`` experiment then isolated the fix: invoking one
   App's entrypoint while a *second*, unrelated App (defined in the same
   file) held a 12.79 GB ``add_local_dir`` on its own image completed in
   **12 seconds** -- the second App's image was never built. A
   ``local_entrypoint`` also cannot call ``.remote()`` on a function
   belonging to a different, non-running App, so "one file, two ``App``
   objects" cannot share a single ``main()`` either -- hence this is a
   wholly separate file with its own ``main()``, not merely a second
   ``App`` object appended to the existing one.

Full writeup: ``docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md`` (B6 closure
pass), where this fix is recorded alongside B6 itself (the loader's
bounded-memory rewrite, measured at 2.65 GB peak against the real file).

**What this script duplicates from `modal_frontier_campaign.py`, and why.**
``PINNED``, ``LOCAL_REPO``/``REMOTE_REPO``/``RESULTS_VOLUME``/``REMOTE_RESULTS``,
``_bootstrap``, ``_assert_pinned_env``, ``_run``, ``_local_git_state``, and
``_completed_on_volume`` are near-identical to the other file's. They are
copied rather than imported: importing the other module would execute ITS
``app = modal.App(...)`` and ``@app.function`` decorators as a Python-level
side effect (harmless on its own, per the two-App experiment above -- merely
constructing another App object and decorating functions on it does not
trigger a build unless THAT App is the one `modal run` activates), but
coupling this file's correctness to "importing the other file happens to be
side-effect-free" is exactly the kind of implicit dependency this bug came
from in the first place. A small amount of duplication is the more robust
choice here.

Usage::

    $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"   # Windows, see the other file's docstring

    modal run scripts/modal_frontier_campaign_t6.py --verify-only
    modal run scripts/modal_frontier_campaign_t6.py --dry-run
    modal run scripts/modal_frontier_campaign_t6.py --smoke       # 2 cells, 2 rounds
    modal run --detach scripts/modal_frontier_campaign_t6.py

Then pull results down exactly as for T1/T2/T4 (same Volume, same
``v2_frontier`` path)::

    modal volume get fedgraa-v2-results v2_frontier results/
    python scripts/check_stage_c_complete.py --tier T6
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
}


def _ignore(path) -> bool:
    """This image DOES include ``data/NF-UQ-NIDS-v2`` -- the whole point of
    this file existing separately -- so, unlike the T1/T2/T4 image, that
    directory is deliberately absent from the exclusion list."""
    from pathlib import Path as _P

    p = _P(path)
    return any(part in EXCLUDE_PARTS for part in p.parts)


image_t6 = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(*[f"{k}=={v}" for k, v in PINNED.items()], "pytest>=7.0")
    .add_local_dir(LOCAL_REPO, remote_path=REMOTE_REPO, copy=True, ignore=_ignore)
)

app_t6 = modal.App("fedgraa-v2-frontier-t6")
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


# Memory/timeout: the two-pass bounded loader was measured at 2.65 GB peak
# against the real 12.79 GB / 75,987,976-row file
# (scripts/verify_nf_uq_nids_v2_memory.py, PRE_COMPUTE_AUDIT_2026-07-28.md B6)
# -- comfortably under CIC-IDS-2017's own 8,630 MB ceiling in the other
# script. Timeout is longer than T1/T2/T4's, since streaming the real file
# twice (tally pass + selection pass, ~17 min measured locally) adds real
# wall-clock beyond the training loop itself. max_containers is lower than
# T1/T2/T4's 99: T6 has only ~50 cells (5 arms x 10 seeds), and running many
# containers each streaming the same 12.79 GB file concurrently has no
# established concurrency ceiling the way §6.1's 99-container Starter-tier
# measurement covers T1/T2/T4's workload -- kept conservative rather than
# assumed safe at the same scale.
@app_t6.function(image=image_t6, cpu=1.0, memory=8192, timeout=5400,
                  volumes={REMOTE_RESULTS: results_volume}, max_containers=20)
def run_cell_t6(payload: dict) -> dict:
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


@app_t6.function(image=image_t6, cpu=1.0, memory=4096, timeout=900,
                  volumes={REMOTE_RESULTS: results_volume})
def verify_env_t6() -> dict:
    """Pre-flight for the T6 image specifically: pinned versions, and the
    CSV **file** present (not just the directory -- the directory always
    exists once ``add_local_dir`` runs; the file's presence is what actually
    matters and is what a broken ``_ignore`` would silently drop)."""
    _bootstrap()
    resolved = _assert_pinned_env()

    import platform

    import fedgraa  # noqa: F401

    from scripts.run_v2_frontier_sweep import NF_UQ_NIDS_V2_CSV_RELPATH

    csv_path = os.path.join(REMOTE_REPO, "data", NF_UQ_NIDS_V2_CSV_RELPATH)
    present = os.path.isfile(csv_path)
    size_gb = os.path.getsize(csv_path) / 1e9 if present else 0.0

    flags = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = line.split(":", 1)[1]
                    break
    except OSError:
        pass

    probe = os.path.join(REMOTE_RESULTS, ".write_probe_t6")
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
        "nf_uq_csv_present": present,
        "nf_uq_csv_path": csv_path,
        "nf_uq_csv_size_gb": round(size_gb, 2),
    }


@app_t6.local_entrypoint()
def main(
    seeds: str = "",
    rounds: int = 200,
    eval_every: int = 20,
    eta_prime: float = -1.0,
    secagg_threshold: int = -1,
    nf_uq_max_per_class: int = -1,
    dry_run: bool = False,
    smoke: bool = False,
    verify_only: bool = False,
    force: bool = False,
):
    """T6 only: all five arms, gamma=1.0, on NF-UQ-NIDS-v2. No ``--tier``
    flag -- this script only ever runs T6, by construction."""
    sys.path.insert(0, LOCAL_REPO)
    from scripts.run_v2_frontier_sweep import (  # noqa: E402
        DEFAULT_ETA_PRIME,
        DEFAULT_SEEDS,
        NF_UQ_NIDS_V2_DEFAULT_MAX_PER_CLASS,
        enumerate_cells,
    )

    print("=" * 78)
    print("  Pre-flight: verifying the T6 image (includes NF-UQ-NIDS-v2)")
    print("=" * 78)
    info = verify_env_t6.remote()
    for k, v in info.items():
        print(f"  {k}: {v}")
    if not info["nf_uq_csv_present"]:
        raise SystemExit(
            "\nABORT: data/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv is not in the "
            "image. Check _ignore() and that the real file exists locally "
            "at data/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv."
        )
    if not info["avx512"]:
        print("\n  NOTE: this container has no AVX512 (expected on Modal).")
    if verify_only:
        return

    sd = [int(s) for s in seeds.replace(",", " ").split()] or DEFAULT_SEEDS
    ep = DEFAULT_ETA_PRIME if eta_prime < 0 else eta_prime
    thr = None if secagg_threshold < 0 else secagg_threshold
    cap = (NF_UQ_NIDS_V2_DEFAULT_MAX_PER_CLASS if nf_uq_max_per_class < 0
           else nf_uq_max_per_class)

    if smoke:
        rounds, eval_every, sd = 2, 1, [0]

    cells = enumerate_cells(
        ["T6"], seeds=sd, rounds=rounds, eval_every=eval_every,
        data_dir=f"{REMOTE_REPO}/data", cache_dir="/tmp/.cache",
        eta_prime=ep, secagg_threshold=thr, nf_uq_max_per_class=cap,
        out_dir=f"{REMOTE_RESULTS}/v2_frontier",
    )
    if smoke:
        keep = {"v1_projected", "v2_m1"}
        cells = [c for c in cells if c.arm in keep][:2]

    print("\n" + "=" * 78)
    print("  FedGRAA-v2 frontier campaign -- T6 (NF-UQ-NIDS-v2)")
    print(f"  seeds={sd}  rounds={rounds}  eta'={ep}  "
          f"secagg_threshold={thr or 'majority'}  nf_uq_max_per_class={cap}")
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
    for res in run_cell_t6.map(payloads, order_outputs=False):
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
    print(f"\n  Pull results:  modal volume get {RESULTS_VOLUME} v2_frontier results/")
    print("  Then verify :  python scripts/check_stage_c_complete.py --tier T6")
    print("=" * 78)
