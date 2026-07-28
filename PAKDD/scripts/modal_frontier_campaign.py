"""Modal driver for the FedGRAA-v2 frontier campaign (T1 / T2 / T4).

**T6 lives in a separate file,** [`scripts/modal_frontier_campaign_t6.py`](modal_frontier_campaign_t6.py),
**on purpose.** The first version of this driver defined T6's image (which
includes the 12.79 GB NF-UQ-NIDS-v2 file) as a second ``@app.function`` on
this SAME ``modal.App``. That made a plain ``--tier T1`` launch build and
upload *both* images, because Modal's ``app.run()`` (which ``modal run``
enters) syncs every function registered on the App it activates, not just
the ones the requested tier touches -- confirmed directly: a
``--tier T1 --smoke`` run that should finish in ~2 minutes instead sat for
15+ minutes building an image containing the 12.79 GB directory. A minimal
two-App experiment then confirmed the fix: invoking App A's entrypoint
while App B (in the same file) held a 12.79 GB ``add_local_dir`` completed
in **12 seconds**, and App B's image was never built. A ``local_entrypoint``
also cannot call ``.remote()`` on a function belonging to a different,
non-running App, so "one file, two Apps" cannot share a single ``main()``
either -- hence a genuinely separate file with its own ``main()``, see
``docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md`` for the full writeup.

This is the piece that ``docs/research_c/REMAINING_COMPUTE_WORK.md`` §3.1 and
``docs/research_c/COMPUTE_INFRASTRUCTURE_2026-07-27.md`` §7 both describe as
"the next step" and which, until now, did not exist -- see
``docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md`` **B1**. It fans
:func:`scripts.run_v2_frontier_sweep.run_one_cell` out over
:func:`~scripts.run_v2_frontier_sweep.enumerate_cells`'s output, one cell per
container, following the pattern validated in
``COMPUTE_INFRASTRUCTURE_2026-07-27.md`` §4.5.

Design decisions, each traceable to a measurement rather than a guess:

* **``cpu=1.0``.** §6.2 measured ``cpu=2.0`` *slower* than ``cpu=1.0`` and
  ``cpu=4.0`` within noise, because ``fedgraa/trainer.py`` pins
  ``TF_NUM_INTRAOP_THREADS=1``. Extra cores buy nothing and cost linearly.
* **Two functions, not one, split by dataset.** Modal bills memory on
  ``max(requested, actual)``. §5.2 measured peak RSS of **8,630 MB** for
  CIC-IDS-2017 and **2,086 MB** for UNSW-NB15. Running every cell in one
  12 GiB container would bill UNSW's 200 cells at 12 GiB for ~11 minutes
  each -- roughly **3x** their measured memory cost. Splitting recovers that.
* **Results on a ``modal.Volume``, committed per cell.** Preemption is
  expected and auto-retried (§6.4); a cell that finished must survive it.
  ``run_one_cell`` writes its own JSON, so the only addition here is the
  commit. Combined with ``missing_cells()``, a from-scratch relaunch redoes
  only what is genuinely missing.
* **No ``nonpreemptible=True``.** §6.4: a 3x surcharge to avoid a failure
  mode that costs $0.015-$0.049 to retry.
* **Exact version pins.** ``requirements.txt`` carries lower bounds only, so
  an unpinned image is not reproducible across rebuilds
  (``PRE_COMPUTE_AUDIT_2026-07-28.md`` **H3**). These pin the exact
  environment the 252-test suite is green in. ``verify_env`` below asserts
  the container actually resolved them, before any cell runs.
* **``data/NF-UQ-NIDS-v2`` excluded from this image entirely.** It is
  12.79 GB and no T1/T2/T4 cell touches it
  (``PRE_COMPUTE_AUDIT_2026-07-28.md`` **H2**); a ``--tier T1`` launch that
  built it anyway is the eager-cross-App-build bug documented above and is
  exactly why T6 now lives in its own file/App instead of a second image on
  this one.
* **B6's memory fix, run for real** (T6's own file, not this one).
  ``fedgraa.data.load_nf_uq_nids_v2_silos``'s ``max_per_class`` cap was
  measured against the actual 12.79 GB / 76M-row release, not estimated --
  see ``docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md`` **B6** for the
  peak-working-set number and ``scripts/verify_nf_uq_nids_v2_memory.py`` for
  the script that produced it.

**On Windows, set UTF-8 first.** Modal's own CLI progress output contains
``✓``, which the default ``cp1252`` console codec cannot encode -- the
run dies with ``'charmap' codec can't encode character '✓'`` before any
container starts. This is a console-encoding fault, not a Modal or campaign
fault, and it costs nothing to avoid::

    $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"

Usage::

    modal run scripts/modal_frontier_campaign.py --verify-only     # env check
    modal run scripts/modal_frontier_campaign.py --tier T1 --dry-run
    modal run scripts/modal_frontier_campaign.py --tier T1 --smoke  # 2 cells, 2 rounds
    modal run --detach scripts/modal_frontier_campaign.py --tier T1
    modal run --detach scripts/modal_frontier_campaign.py --tier T1,T2

    # T6 -- separate file/App, see modal_frontier_campaign_t6.py's own docstring.
    modal run scripts/modal_frontier_campaign_t6.py --smoke
    modal run --detach scripts/modal_frontier_campaign_t6.py

Then pull the results down::

    modal volume get fedgraa-v2-results v2_frontier results/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Environment -- pinned to the exact versions the local suite is green in.
# --------------------------------------------------------------------------- #
PINNED = {
    "tensorflow": "2.21.0",
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "pandas": "3.0.3",
    "scikit-learn": "1.9.0",
}
# ``sklearn`` imports under a different name than it installs under.
_IMPORT_NAME = {"scikit-learn": "sklearn"}

LOCAL_REPO = str(Path(__file__).resolve().parent.parent)
REMOTE_REPO = "/repo"
RESULTS_VOLUME = "fedgraa-v2-results"
REMOTE_RESULTS = "/results"

# Everything the campaign does not need inside the image. ``data`` is NOT
# excluded wholesale -- T1/T2/T4 need data/CIC-IDS-2017 and data/UNSW-NB15 --
# but NF-UQ-NIDS-v2's 12.79 GB merged CSV IS, unconditionally: this App never
# runs a T6 cell (that lives entirely in modal_frontier_campaign_t6.py), so
# there is no case in which this image should ever contain that file.
EXCLUDE_PARTS = {
    ".venv313", ".cache", ".pytest_cache", "__pycache__", "archive",
    ".git", ".claude", "results",
}


def _ignore(path) -> bool:
    from pathlib import Path as _P

    p = _P(path)
    if any(part in EXCLUDE_PARTS for part in p.parts):
        return True
    s = str(p).replace("\\", "/")
    return s.endswith("data/NF-UQ-NIDS-v2") or "/data/NF-UQ-NIDS-v2/" in f"/{s}/"


image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(*[f"{k}=={v}" for k, v in PINNED.items()], "pytest>=7.0")
    .add_local_dir(LOCAL_REPO, remote_path=REMOTE_REPO, copy=True, ignore=_ignore)
)

app = modal.App("fedgraa-v2-frontier")
results_volume = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)


# --------------------------------------------------------------------------- #
# Container-side bootstrap, shared by every function below.
# --------------------------------------------------------------------------- #
def _bootstrap() -> None:
    """Make ``/repo`` importable and force the ``spawn`` start method.

    ``COMPUTE_INFRASTRUCTURE_2026-07-27.md`` §4.3: Linux defaults to ``fork``,
    and forking a process that has already imported TensorFlow deadlocks.
    ``fedgraa/experiment.py`` now forces ``spawn`` at both of its own
    ``ProcessPoolExecutor`` sites, but setting it here too costs nothing and
    covers anything added later. Each cell is single-process regardless --
    §6.4's "do not nest process pools inside a Modal container".
    """
    import multiprocessing
    import os

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    sys.path.insert(0, REMOTE_REPO)
    os.chdir(REMOTE_REPO)


def _assert_pinned_env() -> dict:
    """Fail loudly if the container resolved versions other than :data:`PINNED`.

    A campaign that silently ran half its cells on a different NumPy is not
    reproducible, and the failure is invisible in the results. Checking costs
    milliseconds.
    """
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
            + "\nRefusing to run cells against an unrecorded environment. "
              "Update PINNED in this file (and re-run the local suite against "
              "those versions) if the change is intended."
        )
    return resolved


# --------------------------------------------------------------------------- #
# The unit of work -- one (dataset, arm, gamma, seed, +config) cell.
# --------------------------------------------------------------------------- #
def _run(payload: dict) -> dict:
    _bootstrap()
    _assert_pinned_env()

    import os

    # The container has no ``git`` binary and no ``.git`` directory (the image
    # ships a copy of the tree, not a clone), so ``provenance()`` cannot
    # resolve the commit here -- it returned ``null`` on every cell of the
    # 2026-07-28 smoke test. The launcher resolves it locally and forwards it;
    # ``provenance()`` reads these before attempting git.
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

    # Authoritative resumability check. The launcher also filters against a
    # Volume listing, but that listing is a snapshot taken before the map
    # started; this is the check that actually holds when two launches race or
    # when a preempted input is retried after its result was already written.
    # Without it a relaunch re-runs every completed cell -- the "resumable"
    # property COMPUTE_INFRASTRUCTURE_2026-07-27.md §6.4 relies on to make
    # preemption cost one cell rather than the campaign.
    # A Volume mount is a snapshot taken at container start; writes committed
    # by *other* containers are not visible until reloaded. Without this the
    # skip check would only ever see results that existed when this container
    # booted.
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

    # Persist immediately: a cell that finished must survive a later
    # preemption of a *different* container (§6.4).
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


# Memory is sized from COMPUTE_INFRASTRUCTURE_2026-07-27.md §5.2's MEASURED
# peak RSS (CIC 8,630 MB / UNSW 2,086 MB), with headroom, and split into two
# functions so UNSW cells are not billed at CIC's ceiling.
@app.function(image=image, cpu=1.0, memory=12288, timeout=5400,
              volumes={REMOTE_RESULTS: results_volume}, max_containers=99)
def run_cell_large(payload: dict) -> dict:
    return _run(payload)


@app.function(image=image, cpu=1.0, memory=4096, timeout=5400,
              volumes={REMOTE_RESULTS: results_volume}, max_containers=99)
def run_cell_small(payload: dict) -> dict:
    return _run(payload)


LARGE_MEMORY_DATASETS = {"CIC-IDS-2017"}


def _completed_on_volume() -> set:
    """Basenames of result files already on the results Volume.

    Runs client-side so a relaunch does not pay a container start for every
    already-finished cell. Best-effort: an empty set simply means "filter
    nothing here," and the in-container check in :func:`_run` remains the
    authoritative one.
    """
    try:
        return {
            os.path.basename(e.path)
            for e in results_volume.listdir("v2_frontier")
            if e.path.endswith(".json")
        }
    except Exception:  # noqa: BLE001 -- volume empty / not yet created
        return set()


def _local_git_state():
    """``(commit, dirty)`` resolved on the launching machine.

    Runs client-side on purpose: the Modal image contains a *copy* of the
    working tree, not a git clone, and the container has no ``git`` binary --
    so this cannot be resolved remotely, and every cell of the 2026-07-28
    smoke test wrote ``git_commit: null`` before this was added.
    """
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


@app.function(image=image, cpu=1.0, memory=4096, timeout=900,
              volumes={REMOTE_RESULTS: results_volume})
def verify_env() -> dict:
    """Pre-flight: pinned versions, importability, CPU features, volume write.

    Run this before committing to a campaign. It catches the whole class of
    failure that only shows up 40 minutes in.
    """
    _bootstrap()
    resolved = _assert_pinned_env()

    import os
    import platform

    import fedgraa  # noqa: F401

    flags = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = line.split(":", 1)[1]
                    break
    except OSError:
        pass

    probe = os.path.join(REMOTE_RESULTS, ".write_probe")
    with open(probe, "w") as f:
        f.write("ok")
    results_volume.commit()
    os.remove(probe)
    results_volume.commit()

    datasets = {
        d: os.path.isdir(os.path.join(REMOTE_REPO, "data", d))
        for d in ("CIC-IDS-2017", "UNSW-NB15", "NF-UQ-NIDS-v2")
    }
    return {
        "resolved_versions": resolved,
        "python": platform.python_version(),
        "avx512": "avx512f" in flags,
        "volume_writable": True,
        "datasets_present": datasets,
    }


# --------------------------------------------------------------------------- #
# Local entrypoint.
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main(
    tier: str = "T1",
    datasets: str = "",
    seeds: str = "",
    rounds: int = 200,
    eval_every: int = 20,
    eta_prime: float = -1.0,
    secagg_threshold: int = -1,
    t4_m: int = 1,
    dry_run: bool = False,
    smoke: bool = False,
    verify_only: bool = False,
    force: bool = False,
):
    """Enumerate cells locally, then fan them out over Modal.

    Enumeration happens on the *client* so ``--dry-run`` costs nothing and so
    resumability (``missing_cells``) can consult the Volume through a single
    listing rather than 300 container starts.
    """
    sys.path.insert(0, LOCAL_REPO)
    from scripts.run_v2_frontier_sweep import (  # noqa: E402
        DEFAULT_DATASETS,
        DEFAULT_ETA_PRIME,
        DEFAULT_SEEDS,
        enumerate_cells,
    )

    tiers = [t.strip() for t in tier.replace(",", " ").split() if t.strip()]
    if "T6" in tiers:
        raise SystemExit(
            "\nT6 is not run from this script -- use "
            "scripts/modal_frontier_campaign_t6.py (its own file/App; see "
            "this module's docstring for why). "
            "`modal run scripts/modal_frontier_campaign_t6.py --smoke`"
        )

    print("=" * 78)
    print("  Pre-flight: verifying the Modal container environment")
    print("=" * 78)
    info = verify_env.remote()
    for k, v in info.items():
        print(f"  {k}: {v}")
    missing_data = [d for d, present in info["datasets_present"].items()
                    if not present and d != "NF-UQ-NIDS-v2"]
    if missing_data:
        raise SystemExit(
            f"\nABORT: dataset(s) {missing_data} are not in the image. "
            "Every cell would fail. Check the _ignore() filter."
        )
    if not info["avx512"]:
        print("\n  NOTE: this container has no AVX512 (expected on Modal). "
              "COMPUTE_INFRASTRUCTURE_2026-07-27.md §6.5 -- G3/G4 were "
              "re-verified on exactly this container class, 34/34 green.")

    if verify_only:
        return

    ds = [d.strip() for d in datasets.split(",") if d.strip()] or DEFAULT_DATASETS
    sd = [int(s) for s in seeds.replace(",", " ").split()] or DEFAULT_SEEDS
    ep = DEFAULT_ETA_PRIME if eta_prime < 0 else eta_prime
    thr = None if secagg_threshold < 0 else secagg_threshold

    if smoke:
        rounds, eval_every, sd = 2, 1, [0]
        ds = ds[:1]

    cells = enumerate_cells(
        tiers, datasets=ds, seeds=sd, rounds=rounds, eval_every=eval_every,
        data_dir=f"{REMOTE_REPO}/data", cache_dir="/tmp/.cache",
        eta_prime=ep, secagg_threshold=thr, t4_m=t4_m,
        out_dir=f"{REMOTE_RESULTS}/v2_frontier",
    )
    if smoke:
        # One cell per arm shape: the v1 reference plus the cheapest v2 arm.
        keep = {"v1_projected", "v2_m1"}
        cells = [c for c in cells if c.arm in keep][:2]

    print("\n" + "=" * 78)
    print(f"  FedGRAA-v2 frontier campaign -- tiers={tiers}")
    print(f"  datasets={ds}  seeds={sd}  rounds={rounds}")
    print(f"  eta'={ep}  secagg_threshold={thr or 'majority'}"
          + (f"  t4_m={t4_m}" if "T4" in tiers else ""))
    print(f"  {len(cells)} cells -> Modal (cpu=1.0, <=99 concurrent)")
    print("=" * 78)

    # Resumability, launcher side: don't even dispatch cells whose result is
    # already on the Volume. Each container re-checks authoritatively (that is
    # what handles races and preemption retries), but filtering here avoids
    # paying a container start per already-done cell on a relaunch.
    #
    # This runs BEFORE --dry-run on purpose: a dry run that lists cells the
    # campaign would actually skip is a dry run that misreports the work.
    if not force:
        done = _completed_on_volume()
        if done:
            before = len(cells)
            cells = [c for c in cells if os.path.basename(c.path) not in done]
            print(f"  resuming: {before - len(cells)} of {before} cells already "
                  f"complete on the volume, {len(cells)} to run")

    if dry_run:
        for c in cells:
            # ``CellSpec.path`` is a property, so it is recomputed *inside* the
            # container with Linux separators -- only this client-side preview
            # would otherwise show a Windows one and misdescribe the remote
            # path. (``asdict`` serializes fields, not properties, so nothing
            # platform-specific is ever shipped.)
            print(f"  WOULD RUN: {c.dataset} {c.arm} g={c.gamma} "
                  f"seed={c.seed}{c.config_tag} -> "
                  f"{c.path.replace(os.sep, '/')}")
        print(f"\n  {len(cells)} cell(s) would run.")
        return

    from dataclasses import asdict

    # Resolve the commit HERE, where git exists, and forward it into every
    # container. Without this the campaign's results carry no code identity.
    git_commit, git_dirty = _local_git_state()
    if git_commit is None:
        print("  WARNING: could not resolve a git commit locally -- results "
              "will carry no code identity.")
    else:
        print(f"  commit={git_commit[:12]}"
              + ("  (DIRTY TREE)" if git_dirty else "  (clean)"))

    def _payload(c) -> dict:
        return {
            "cell": asdict(c),
            "force": force,
            "FEDGRAA_GIT_COMMIT": git_commit,
            "FEDGRAA_GIT_DIRTY": "1" if git_dirty else "0",
        }

    if not cells:
        print("\n  Nothing to do -- every requested cell already has a result.")
        print(f"  Pull results:  modal volume get {RESULTS_VOLUME} v2_frontier results/")
        return

    large = [_payload(c) for c in cells if c.dataset in LARGE_MEMORY_DATASETS]
    small = [_payload(c) for c in cells
             if c.dataset not in LARGE_MEMORY_DATASETS]
    print(f"  {len(large)} cells at memory=12288 | "
          f"{len(small)} at memory=4096\n")

    done, failed, skipped = 0, [], 0
    for fn, batch in ((run_cell_large, large), (run_cell_small, small)):
        if not batch:
            continue
        for res in fn.map(batch, order_outputs=False):
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
    print(f"  Then verify :  python scripts/check_stage_c_complete.py "
          f"--tier {' '.join(tiers)}")
    print("=" * 78)
