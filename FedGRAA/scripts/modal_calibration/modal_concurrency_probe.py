"""Empirically measure the real concurrent-container ceiling for this Modal
account, instead of trusting the pricing page's tier table blindly.

Launches N trivial (sleep-only, no TF/data) containers via .map() with
max_containers set above the documented Starter-tier limit (100), has each
one record its own start/end wall-clock timestamp, then reconstructs the
maximum number of overlapping (i.e. truly concurrent) containers from those
timestamps. Cheap: trivial image, ~20s of compute per container.

Measured result (2026-07-28, see modal_concurrency_result.json): 99 of 130
launched containers ran concurrently -- confirms the Starter-tier 100-container
ceiling directly on this account (docs/research_c/COMPUTE_INFRASTRUCTURE_2026-07-27.md
Sec 6.1), rather than assuming the pricing page's tier table applies unchanged.

Usage: modal run scripts/modal_calibration/modal_concurrency_probe.py
"""

from __future__ import annotations

import modal

app = modal.App("fedgraa-concurrency-probe")

N = 130          # above the documented Starter limit of 100
SLEEP_SECONDS = 20

image = modal.Image.debian_slim(python_version="3.13")


@app.function(image=image, cpu=0.125, memory=128, timeout=120, max_containers=N)
def probe(i: int) -> dict:
    import time

    t0 = time.time()
    time.sleep(SLEEP_SECONDS)
    t1 = time.time()
    return {"i": i, "start": t0, "end": t1}


@app.local_entrypoint()
def main():
    import json
    import os
    import time

    print(f"Launching {N} concurrent probes (max_containers={N}) ...")
    t_launch0 = time.time()
    results = list(probe.map(range(N)))
    wall = time.time() - t_launch0

    # Reconstruct max overlap from (start, end) intervals -- a sweep-line count.
    events = []
    for r in results:
        events.append((r["start"], 1))
        events.append((r["end"], -1))
    events.sort()
    cur = 0
    peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)

    print(f"Total wall-clock for all {N} probes: {wall:.1f}s (single-batch would be ~{SLEEP_SECONDS}s if N <= ceiling)")
    print(f"Empirically measured peak concurrent containers: {peak} / {N} launched")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modal_concurrency_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n_launched": N, "wall_clock_sec": wall, "peak_concurrent": peak, "raw": results}, f, indent=2)
    print(f"Wrote {out_path}")
