"""Empirically settle FEDGRAA_V2_MASTER_PLAN finding F12.

F12 claims that the federated backward pass specified in
``privacy_compliant_fedgraa_redesign.md`` §4.1 -- a plain SecAgg-*sum* of
per-client composite-loss gradients, described there as "mathematically
identical" to the pooled computation -- is **not** equal to ``grad_P L'``, for
two independent reasons:

  A. ``MLP.loss_and_grad`` reduces with ``tf.reduce_mean``, so per-client
     gradients must be **count-weighted**, not summed.
  B. ``L_a = min(L_h, L_m)`` is selected from the **pooled** scalars; a client
     choosing its own local arm forms a gradient the pooled path never forms.

Until this script was written, F12 was an argument from reading the source. It
is now a measurement. Four variants of the same quantity are computed from
identical inputs and compared against the v1 pooled reference:

  V0  pooled                -- the v1 reference (FedGRAA._composite_grad_P)
  V1  naive, local tags     -- exactly what both design documents specify
  V2  naive, global tags    -- isolates obstruction A (B is partly removed)
  V3  corrected             -- global tags, global arm, count-weighted sum-form
                               (the plan's §3 two-phase protocol)

Expected: V3 matches V0 to float32 summation-order tolerance; V1 and V2 do not.
A V3 mismatch would refute the plan's corrected design; a V1 match would refute
F12 itself. Prints PASS/FAIL and exits non-zero on either.

    python scripts/verify_v2_exactness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedgraa.aggregation import (  # noqa: E402
    FedGRAA,
    _composite_min_arm,
    _grad_wrt_weights,
    _loss_and_grad_P,
    static_reparam_weights,
)
from fedgraa.config import CIC_IDS_2017, UNSW_NB15, FedConfig  # noqa: E402
from fedgraa.data import make_synthetic_dataset  # noqa: E402
from fedgraa.model import MLP, Adam, aggregate, init_params  # noqa: E402
from fedgraa.partition import ValidationSplit, partition_non_iid  # noqa: E402

TOL_EXACT = 1e-6      # float32 summation-order slack on ~4e4 parameters
TOL_DISTINCT = 1e-3   # below this, two variants are "the same answer"
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def rel_err(a, b) -> float:
    """Relative L2 distance between two flattened parameter-shaped lists."""
    fa = np.concatenate([x.ravel() for x in a]).astype(np.float64)
    fb = np.concatenate([x.ravel() for x in b]).astype(np.float64)
    den = np.linalg.norm(fb)
    return float(np.linalg.norm(fa - fb) / den) if den else float(np.linalg.norm(fa - fb))


# --------------------------------------------------------------------------- #
# The plan's §3 primitives, prototyped here (they become W1.1 in fedgraa/).
# --------------------------------------------------------------------------- #
def stratum_tags(counts: np.ndarray, rare_ids) -> np.ndarray:
    """Server-side length-C tag vector in {'h','m','n'}, from a histogram only.

    Carries median_frequency_split's arithmetic -- including its gap-2.6
    non-empty-D_V^h fallback -- but consumes counts instead of raw labels, so a
    client can tag its own rows without seeing anyone else's.
    """
    C = len(counts)
    tags = np.full(C, "m", dtype="<U1")
    rare = set(int(r) for r in rare_ids)
    present = [c for c in range(C) if counts[c] > 0 and c not in rare]
    for r in rare:
        tags[r] = "n"
    if not present:
        return tags
    freqs = np.array([counts[c] for c in present], dtype=float)
    median = float(np.median(freqs))
    high = {c for c, f in zip(present, freqs) if f > median}
    if not high and len(present) > 1:                     # gap-2.6 fallback
        order = [present[i] for i in np.argsort(freqs)]
        high = set(order[len(order) // 2:])
    for c in high:
        tags[c] = "h"
    return tags


def apply_tags(X, y, tags) -> ValidationSplit:
    """Client-side: split one client's own rows using the broadcast tags."""
    lab = tags[y] if len(y) else np.empty(0, dtype="<U1")
    m_h, m_m, m_n = (lab == "h"), (lab == "m"), (lab == "n")
    return ValidationSplit(X[m_h], y[m_h], X[m_m], y[m_m], X[m_n], y[m_n],
                           [int(c) for c in np.where(tags == "n")[0]])


def local_split(X, y, rare_ids) -> ValidationSplit:
    """The fully-naive path: each client runs median_frequency_split alone."""
    from fedgraa.partition import median_frequency_split
    return median_frequency_split(X, y, list(rare_ids))


# --------------------------------------------------------------------------- #
# The four variants
# --------------------------------------------------------------------------- #
def v0_pooled(fg: FedGRAA, params, val) -> list:
    """v1 reference: one pooled call, exactly as aggregation.py does today."""
    return fg._composite_grad_P(params, val)[1]


def naive_sum(fg: FedGRAA, params, per_client_vals) -> list:
    """Plain sum of per-client composite gradients -- what the docs specify."""
    total = [np.zeros_like(p) for p in params]
    for v in per_client_vals:
        g = fg._composite_grad_P(params, v)[1]
        total = [t + gi for t, gi in zip(total, g)]
    return total


def v3_corrected(model: MLP, params, per_client_vals, gamma) -> list:
    """Two-phase, count-weighted, sum-form -- the plan's §3 protocol."""
    # ---- phase 1: unnormalized scalars -> pooled L_h/L_m -> global arm ----
    S = {"h": 0.0, "m": 0.0, "n": 0.0}
    N = {"h": 0, "m": 0, "n": 0}
    for v in per_client_vals:
        for key, (Xs, ys) in (("h", (v.X_h, v.y_h)), ("m", (v.X_m, v.y_m)),
                              ("n", (v.X_n, v.y_n))):
            if len(ys):
                S[key] += model.loss(params, Xs, ys) * len(ys)   # = sum of l_i
                N[key] += len(ys)
    L = {k: (S[k] / N[k] if N[k] else float("inf")) for k in S}   # 0-count -> inf
    _, arm = _composite_min_arm(L["h"], L["m"])

    # ---- phase 2: count-weighted sum-form gradients ----------------------
    g_a = [np.zeros_like(p) for p in params]
    g_u = [np.zeros_like(p) for p in params]
    n_a = N[arm] if arm else 0
    n_u = N["n"]
    for v in per_client_vals:
        if arm:
            Xs, ys = (v.X_h, v.y_h) if arm == "h" else (v.X_m, v.y_m)
            if len(ys):
                _, g = _loss_and_grad_P(model, params, Xs, ys)
                g_a = [t + len(ys) * gi for t, gi in zip(g_a, g)]
        if len(v.y_n):
            _, g = _loss_and_grad_P(model, params, v.X_n, v.y_n)
            g_u = [t + len(v.y_n) * gi for t, gi in zip(g_u, g)]
    g_a = [t / n_a for t in g_a] if n_a else g_a
    g_u = [t / n_u for t in g_u] if n_u else g_u
    return [(1.0 - gamma) * a + gamma * u for a, u in zip(g_a, g_u)]


def w_after_step(grad_P, client_params, w0, eta_prime) -> np.ndarray:
    """One Adam step on w, so divergence is shown where it actually bites."""
    opt = Adam(lr=eta_prime)
    w = np.array(w0, dtype=np.float32).copy()
    opt.step([w], [_grad_wrt_weights(grad_P, client_params)])
    return w


# --------------------------------------------------------------------------- #
def run(spec, gamma: float) -> None:
    cfg = FedConfig(gamma=gamma, seed=0)
    X, y, spec = make_synthetic_dataset(spec, seed=0)
    split = partition_non_iid(X, y, spec, n_rare=cfg.n_rare,
                              val_fraction=cfg.val_fraction, seed=cfg.seed)
    layer_sizes = spec.model_config().layer_sizes
    model = MLP(layer_sizes, seed=0)
    fg = FedGRAA(gamma=gamma, eta_prime=cfg.eta_prime,
                 weight_epochs=cfg.weight_epochs, layer_sizes=layer_sizes)

    # A realistic aggregate: one local step per client from a common init.
    base = init_params(layer_sizes, seed=0)
    rng = np.random.default_rng(0)
    client_params = [[p + 0.01 * rng.normal(size=p.shape).astype(np.float32)
                      for p in base] for _ in split.clients]
    w0 = static_reparam_weights([c.n_train for c in split.clients])
    params = aggregate(client_params, w0)

    rare_ids = split.val_split.rare_class_ids
    counts = np.bincount(split.y_val, minlength=spec.n_classes)
    tags = stratum_tags(counts, rare_ids)

    global_vals = [apply_tags(c.X_val, c.y_val, tags) for c in split.clients]
    local_vals = [local_split(c.X_val, c.y_val, rare_ids) for c in split.clients]

    V0 = v0_pooled(fg, params, split.val_split)
    V1 = naive_sum(fg, params, local_vals)     # as both design docs specify
    V2 = naive_sum(fg, params, global_vals)    # global tags, still mean+local arm
    V3 = v3_corrected(model, params, global_vals, gamma)

    print(f"\n  {spec.name}  gamma={gamma}  K={split.n_clients}  "
          f"|D_V|={len(split.y_val):,}  arm(pooled)="
          f"{_composite_min_arm(*[_loss_and_grad_P(model, params, a, b)[0] for a, b in ((split.val_split.X_h, split.val_split.y_h), (split.val_split.X_m, split.val_split.y_m))])[1]}")
    e1, e2, e3 = rel_err(V1, V0), rel_err(V2, V0), rel_err(V3, V0)
    print(f"    rel. L2 error vs pooled grad_P L':  "
          f"V1 naive={e1:.3e}   V2 global-tags={e2:.3e}   V3 corrected={e3:.3e}")

    tag = f"{spec.name} g={gamma}"
    check(f"{tag}: V3 corrected protocol MATCHES pooled", e3 < TOL_EXACT,
          f"(rel err {e3:.2e} < {TOL_EXACT:.0e})")
    check(f"{tag}: V1 naive sum DIFFERS from pooled  [F12 confirmed]",
          e1 > TOL_DISTINCT, f"(rel err {e1:.2e})")

    w_ref = w_after_step(V0, client_params, w0, cfg.eta_prime)
    w_nav = w_after_step(V1, client_params, w0, cfg.eta_prime)
    w_cor = w_after_step(V3, client_params, w0, cfg.eta_prime)
    print(f"    downstream w after one Adam step:  "
          f"||w_naive-w_ref||={np.linalg.norm(w_nav - w_ref):.3e}   "
          f"||w_corrected-w_ref||={np.linalg.norm(w_cor - w_ref):.3e}")
    check(f"{tag}: corrected w matches reference w",
          np.allclose(w_cor, w_ref, atol=1e-6))


def _drop_stratum(v: ValidationSplit, which: str) -> ValidationSplit:
    """Return a copy of ``v`` with one stratum emptied, preserving dtypes."""
    e_X, e_y = v.X_h[:0], v.y_h[:0]
    parts = {"h": (v.X_h, v.y_h), "m": (v.X_m, v.y_m), "n": (v.X_n, v.y_n)}
    parts[which] = (e_X, e_y)
    return ValidationSplit(*parts["h"], *parts["m"], *parts["n"], v.rare_class_ids)


def run_degenerate(spec, gamma: float) -> None:
    """The empty-stratum corner the plan's sec 3 warns about (n_s = 0 -> L_s = inf).

    v1's ``_loss_and_grad_P`` returns ``(inf, zeros)`` for an empty split, which
    is what fires ``_composite_min_arm``'s NaN-symmetric fallback. v2 must
    reproduce that from counts alone. This was specified but never tested until
    now; F3 measured it as reachable (2/10 CIC seeds have an empty ``D_V^n``).
    """
    cfg = FedConfig(gamma=gamma, seed=0)
    X, y, spec = make_synthetic_dataset(spec, seed=0)
    split = partition_non_iid(X, y, spec, n_rare=cfg.n_rare,
                              val_fraction=cfg.val_fraction, seed=cfg.seed)
    layer_sizes = spec.model_config().layer_sizes
    model = MLP(layer_sizes, seed=0)
    fg = FedGRAA(gamma=gamma, eta_prime=cfg.eta_prime,
                 weight_epochs=cfg.weight_epochs, layer_sizes=layer_sizes)
    base = init_params(layer_sizes, seed=0)
    rng = np.random.default_rng(0)
    client_params = [[p + 0.01 * rng.normal(size=p.shape).astype(np.float32)
                      for p in base] for _ in split.clients]
    w0 = static_reparam_weights([c.n_train for c in split.clients])
    params = aggregate(client_params, w0)

    counts = np.bincount(split.y_val, minlength=spec.n_classes)
    tags = stratum_tags(counts, split.val_split.rare_class_ids)
    global_vals = [apply_tags(c.X_val, c.y_val, tags) for c in split.clients]

    for which in ("n", "h", "m"):
        pooled = _drop_stratum(split.val_split, which)
        per_client = [_drop_stratum(v, which) for v in global_vals]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # expected empty-split warning
            V0 = v0_pooled(fg, params, pooled)
            V3 = v3_corrected(model, params, per_client, gamma)
        e = rel_err(V3, V0)
        check(f"{spec.name} g={gamma}: V3 matches pooled with stratum '{which}' EMPTY",
              e < TOL_EXACT, f"(rel err {e:.2e})")


def main() -> int:
    print("=" * 78)
    print("F12 -- is the federated backward pass exact? (plan sec 2 F12, sec 3)")
    print("=" * 78)
    for spec in (CIC_IDS_2017, UNSW_NB15):
        for gamma in (1.0, 0.5, 0.0):
            run(spec, gamma)

    print("\n" + "-" * 78)
    print("Degenerate strata -- the empty-split corner (plan sec 3 edge-case note)")
    print("-" * 78)
    for spec in (CIC_IDS_2017, UNSW_NB15):
        run_degenerate(spec, 0.5)
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("F12 confirmed by measurement: the naive sum is wrong, the two-phase")
    print("count-weighted protocol reproduces the pooled gradient exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
