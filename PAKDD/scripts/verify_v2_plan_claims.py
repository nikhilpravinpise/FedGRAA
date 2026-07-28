"""Re-derive every numeric claim in docs/research_c/FEDGRAA_V2_MASTER_PLAN.md.

The plan's findings F3, F5, F10a and F13 all rest on numbers that must come from
the live ``fedgraa/config.py`` and ``fedgraa/partition.py`` rather than from hand
arithmetic -- the plan's own §4.1 documents a 2.4% error that arose in a source
document exactly that way. This script recomputes all of them and prints a
PASS/FAIL line per claim, so the plan can be re-verified after any config change.

Checks
------
1.  F5  -- parameter counts, message sizes, frontier multipliers and byte totals.
2.  F10a -- pooled ``X_val`` is bit-identical to the per-client concatenation.
3.  F3  -- full-shard vs. held-out rarity histograms: determinism, zero-count
           hazard, and ``BottomN`` recovery under Laplace DP noise.
4.  F13 -- the UNSW-NB15 ``n>=5`` oracle set is not count-derivable.

Runs on synthetic data shaped to ``spec.used_counts``; needs only numpy, and
deliberately loads ``config``/``partition`` directly so it works even when the
TensorFlow import path is unavailable (see the plan's §13).

    python scripts/verify_v2_plan_claims.py
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DP_TRIALS = 2000
SEEDS = range(10)
MINUTES_PER_RUN = 17.0  # Stage B measured 15-20 min/run on CPU


def _load_tf_free() -> types.ModuleType:
    """Load fedgraa.config + fedgraa.partition without importing the package.

    ``fedgraa/__init__.py`` imports ``model``, which imports TensorFlow. Both
    modules needed here are pure numpy, so they are loaded into a synthetic
    package instead -- keeping this script runnable in environments where the
    TF import chain is blocked.
    """
    pkg = types.ModuleType("_fgpure")
    pkg.__path__ = [str(REPO / "fedgraa")]
    sys.modules["_fgpure"] = pkg
    for name in ("config", "partition"):
        spec = importlib.util.spec_from_file_location(
            f"_fgpure.{name}", REPO / "fedgraa" / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"_fgpure.{name}"] = mod
        spec.loader.exec_module(mod)
    return pkg


_fg = _load_tf_free()
from _fgpure.config import CIC_IDS_2017, UNSW_NB15, FedConfig  # noqa: E402
from _fgpure.partition import partition_non_iid  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def make_dataset(spec, seed: int = 0):
    """Synthesize a dataset with exactly ``spec.used_counts`` rows per class."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for cid, cnt in enumerate(spec.used_counts):
        if cnt:
            Xs.append(rng.normal(size=(cnt, spec.n_features)).astype(np.float32))
            ys.append(np.full(cnt, cid, dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(ys)


def hist(y, C):
    return np.bincount(y, minlength=C).astype(np.int64)


def bottom_n(counts, attack_ids, n):
    """The n rarest attack classes -- mirrors rare_client_ids' fallback path."""
    return sorted(attack_ids, key=lambda i: counts[i])[:n]


def split_quiet(X, y, spec, n_rare, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # UNSW n>=7 truncation warning
        return partition_non_iid(X, y, spec, n_rare=n_rare,
                                 val_fraction=0.10, seed=seed)


def oracle_quiet(spec, n_rare):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return set(spec.rare_client_ids(n_rare))


def histograms(split, C):
    """Return (full-shard, held-out) SecAgg-summed label histograms."""
    hf = np.zeros(C, dtype=np.int64)
    hh = np.zeros(C, dtype=np.int64)
    for c in split.clients:
        hf += hist(np.concatenate([c.y_train, c.y_val]), C)
        hh += hist(c.y_val, C)
    return hf, hh


# --------------------------------------------------------------------------- #
def check_f5_communication(cfg) -> None:
    """F5 / §4.1 -- parameter counts, message sizes, frontier byte totals."""
    print("\nF5  communication frontier (plan sec 4.1)")
    expected_gb = {  # the plan's §4.1 table
        "CIC-IDS-2017": {1: 5.606, 2: 3.738, 3: 2.803, 5: 1.869},
        "UNSW-NB15": {1: 2.908, 2: 1.939, 3: 1.454, 5: 0.969},
    }
    for spec in (CIC_IDS_2017, UNSW_NB15):
        sizes = spec.model_config().layer_sizes
        params = sum(sizes[i] * sizes[i + 1] + sizes[i + 1]
                     for i in range(len(sizes) - 1))
        msg = params * 4  # float32
        base = 2 * spec.n_clients * cfg.rounds * msg
        print(f"  {spec.name}: layer_sizes={sizes}  params={params:,}  "
              f"msg={msg:,} B  baseline={base / 1e9:.3f} GB")
        for m, want in expected_gb[spec.name].items():
            mult = 1 + math.ceil(cfg.weight_epochs / m)
            got = mult * base / 1e9
            check(f"{spec.name} m={m} -> {mult}x = {got:.3f} GB",
                  abs(got - want) < 0.002, f"(plan says {want:.3f} GB)")


def check_f10a_pooled_val() -> None:
    """F10a -- pooled X_val is exactly the concatenation of per-client splits."""
    print("\nF10a  pooled D_V == concat(per-client held-out splits) (plan sec 2 F10a)")
    for spec in (CIC_IDS_2017, UNSW_NB15):
        X, y = make_dataset(spec)
        s = split_quiet(X, y, spec, 3, 0)
        ok = (np.array_equal(np.concatenate([c.X_val for c in s.clients]), s.X_val)
              and np.array_equal(np.concatenate([c.y_val for c in s.clients]), s.y_val))
        check(f"{spec.name} bit-identical", ok, f"({len(s.y_val):,} rows)")


def check_f3_histograms() -> None:
    """F3 -- determinism, zero-count hazard, and DP recovery of BottomN."""
    print("\nF3  rarity histogram: full shard vs. held-out (plan sec 2 F3, sec 5.2)")
    for spec in (CIC_IDS_2017, UNSW_NB15):
        X, y = make_dataset(spec)
        C, oracle = spec.n_classes, oracle_quiet(spec, 3)
        uc = np.array(spec.used_counts, dtype=np.int64)
        deterministic = True
        full_ok = held_ok = zero_seeds = 0
        for s in SEEDS:
            hf, hh = histograms(split_quiet(X, y, spec, 3, s), C)
            deterministic &= np.array_equal(hf, uc)
            full_ok += set(bottom_n(hf, spec.attack_ids, 3)) == oracle
            held_ok += set(bottom_n(hh, spec.attack_ids, 3)) == oracle
            zero_seeds += any(hh[c] == 0 for c in oracle)
        check(f"{spec.name} full-shard histogram == used_counts on every seed",
              deterministic)
        check(f"{spec.name} full-shard BottomN recovers oracle",
              full_ok == len(SEEDS), f"({full_ok}/{len(SEEDS)})")
        check(f"{spec.name} held-out BottomN recovers oracle (n=3)",
              held_ok == len(SEEDS), f"({held_ok}/{len(SEEDS)})")
        print(f"         seeds with >=1 oracle-rare class ABSENT from D_V^n: "
              f"{zero_seeds}/{len(SEEDS)}"
              + ("   <-- v1 fragility, plan sec 2 F3" if zero_seeds else ""))

    print("\n     under Laplace DP noise, sigma = (2/eps)*sqrt(2K)  "
          f"({DP_TRIALS} trials)")
    for spec in (CIC_IDS_2017, UNSW_NB15):
        X, y = make_dataset(spec)
        s = split_quiet(X, y, spec, 3, 0)
        K, C, oracle = s.n_clients, spec.n_classes, oracle_quiet(spec, 3)
        hf, hh = histograms(s, C)
        for eps in (1.0, 4.0, 10.0):
            b, rng = 2.0 / eps, np.random.default_rng(0)
            okf = okh = 0
            for _ in range(DP_TRIALS):
                okf += set(bottom_n(hf + rng.laplace(0, b, (K, C)).sum(0),
                                    spec.attack_ids, 3)) == oracle
                okh += set(bottom_n(hh + rng.laplace(0, b, (K, C)).sum(0),
                                    spec.attack_ids, 3)) == oracle
            rf, rh = 100 * okf / DP_TRIALS, 100 * okh / DP_TRIALS
            check(f"{spec.name} eps={eps:<4} sigma={b * math.sqrt(2 * K):5.2f} "
                  f"full={rf:6.2f}% held={rh:6.2f}%", rf >= 99.0 and rh >= 99.0)


def check_f13_uncoverable_oracle() -> None:
    """F13 -- UNSW n>=5 oracle set is not derivable from as-used counts."""
    print("\nF13  oracle rare set recoverability by n (plan sec 2 F13)")
    expect = {("CIC-IDS-2017", 3): True, ("CIC-IDS-2017", 5): True,
              ("CIC-IDS-2017", 7): True, ("UNSW-NB15", 3): True,
              ("UNSW-NB15", 5): False, ("UNSW-NB15", 7): False}
    for spec in (CIC_IDS_2017, UNSW_NB15):
        X, y = make_dataset(spec)
        C = spec.n_classes
        for n_rare in (3, 5, 7):
            oracle = oracle_quiet(spec, n_rare)
            ok = sum(set(bottom_n(histograms(split_quiet(X, y, spec, n_rare, s),
                                             C)[0], spec.attack_ids,
                                  len(oracle))) == oracle for s in SEEDS)
            recoverable = ok == len(SEEDS)
            want = expect[(spec.name, n_rare)]
            check(f"{spec.name} n={n_rare} recoverable={recoverable} "
                  f"({ok}/{len(SEEDS)})", recoverable == want,
                  "" if want else "<-- expected: capped-count tie, plan sec 2 F13")


def check_compute_budget() -> None:
    """§7.2 -- the tiered campaign's CPU-hour estimates."""
    print(f"\n§7.2  compute budget at {MINUTES_PER_RUN:.0f} min/run")
    tiers = [("T1 frontier", 5 * 2 * 10), ("T2 gamma", 5 * 2 * 10 * 2),
             ("T3 approx baselines", 6 * 2 * 10), ("T5 novelty grid", 8 * 2 * 10)]
    must = 0.0
    for name, runs in tiers:
        h = runs * MINUTES_PER_RUN / 60
        if not name.startswith("T5"):
            must += h
        print(f"  {name:22s} runs={runs:4d}  {h:5.1f} CPU-h")
    check(f"T1+T2+T3 = {must:.0f} CPU-h "
          f"({must / 24:.1f} d @1 worker, {must / 3 / 24:.1f} d @3)",
          abs(must - 119.0) < 1.0, "(plan says 119 CPU-h)")


def main() -> int:
    cfg = FedConfig()
    print("=" * 74)
    print("FEDGRAA_V2_MASTER_PLAN -- numeric claim verification")
    print(f"FedConfig: T={cfg.rounds}  E'={cfg.weight_epochs}  "
          f"eta'={cfg.eta_prime}  val_fraction={cfg.val_fraction}")
    print("=" * 74)

    check_f5_communication(cfg)
    check_f10a_pooled_val()
    check_f3_histograms()
    check_f13_uncoverable_oracle()
    check_compute_budget()

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"{len(FAILURES)} CLAIM(S) NO LONGER HOLD -- update the plan:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All plan claims verified against the live codebase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
