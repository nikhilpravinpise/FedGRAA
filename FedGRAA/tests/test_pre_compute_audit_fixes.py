"""Regression guards for every defect in
``docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md``.

Each test names the finding it prevents and states, in its own docstring, what
would go wrong without it. These are deliberately kept in one module rather
than scattered, because they share a single purpose: **the campaign path
(driver -> config -> trainer -> persisted result) had never been executed
end-to-end, and the unit tests that were green did not cover it.** Four of the
six blocking defects were silent -- they produced plausible numbers rather
than errors -- so each guard below asserts on the *mechanism*, not on an
output that would look fine either way.

Findings covered:

* **B2** -- the campaign must run at Stage B's combined-fix ``eta'=0.01``,
  not ``FedConfig``'s literal ``0.1``.
* **B3** -- cells differing in any experimental field must not share a path.
* **B4** -- an unrecoverable SecAgg round aborts that round; it does not kill
  the run.
* **B5** -- DP noise is added per-client *before* SecAgg, never server-side
  after it.
* **H4** -- the v1 arm bills bytes, so the frontier's ``1x`` anchor is
  measured.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedgraa.config import DatasetSpec, FedConfig
from fedgraa.data import make_synthetic_dataset
from fedgraa.partition import local_label_histogram, partition_non_iid
from fedgraa.privacy import aggregate_noise_std
from fedgraa.secagg import SecAggChannel, SecAggDropoutError
from fedgraa.trainer import FederatedTrainer, FederatedTrainerV2
from scripts.run_v2_frontier_sweep import (
    DEFAULT_ETA_PRIME,
    T4_DP_EPSILONS,
    T4_P_DROPS,
    CellSpec,
    enumerate_cells,
    provenance,
)


def _toy_spec() -> DatasetSpec:
    return DatasetSpec(
        name="TOY_AUDIT",
        n_features=6,
        labels=["Benign", "A", "B", "C", "Rare"],
        used_counts=[300, 120, 90, 60, 10],
        benign_id=0,
        rare_label_order=["Rare", "C", "B"],
    )


def _make_split(seed: int = 0):
    spec = _toy_spec()
    X, y, spec = make_synthetic_dataset(spec, seed=seed, class_separation=3.0)
    return partition_non_iid(X, y, spec, n_rare=1, val_fraction=0.10, seed=seed)


def _pooled_histogram(split, n_classes: int) -> np.ndarray:
    total = np.zeros(n_classes, dtype=np.float64)
    for shard in split.clients:
        total += local_label_histogram(shard, n_classes, scope="full")
    return total


# =========================================================================== #
# B2 -- the frontier's eta' is the Stage B combined-fix value
# =========================================================================== #
def test_campaign_eta_prime_is_combined_fix_not_fedconfig_default():
    """Without this, the campaign silently inherits ``FedConfig.eta_prime =
    0.1``. With projection, that is Stage B's ``projection-alone`` arm, which
    ``docs/current/fedgraa_convergence_defect.md`` §5.3 measured at **0.0%**
    rare-class accuracy on CIC-IDS-2017 at gamma <= 0.5. The frontier's
    privacy-violating reference point would be an arm this project already
    established as broken -- and every "v2 beats v1" delta would look
    enormous for the wrong reason.
    """
    assert DEFAULT_ETA_PRIME == 0.01
    # The strict-reproduction default must NOT move -- only the campaign's.
    assert FedConfig().eta_prime == 0.1
    assert CellSpec("CIC-IDS-2017", "v2_m1", 1.0, 0).eta_prime == 0.01
    for cell in enumerate_cells(["T1", "T2", "T4"], datasets=["UNSW-NB15"],
                                seeds=[0]):
        assert cell.eta_prime == 0.01, f"{cell.arm} escaped the pin"


def test_eta_prime_is_held_equal_across_both_arms():
    """Holding eta' fixed is what makes the frontier measure the refresh
    interval and the protocol change, rather than confounding them with a
    step-size difference between the arms."""
    cells = enumerate_cells(["T1"], datasets=["UNSW-NB15"], seeds=[0])
    assert len({c.eta_prime for c in cells}) == 1
    assert {c.arm for c in cells} == {
        "v1_projected", "v2_m1", "v2_m2", "v2_m3", "v2_m5",
    }


# =========================================================================== #
# B3 -- experimentally-distinct cells must not collide on disk
# =========================================================================== #
_VARIANTS = [
    {},
    {"p_drop": 0.1},
    {"p_drop": 0.3},
    {"dp_epsilon": 1.0},
    {"dp_epsilon": 4.0},
    {"dp_epsilon": 10.0},
    {"eta_prime": 0.1},
    {"n_rare": 5},
    {"secagg_threshold": 3},
    {"rounds": 60},
    {"p_drop": 0.3, "dp_epsilon": 1.0},
]


def test_cells_differing_in_any_experimental_field_have_distinct_paths():
    """Before this guard the path was ``{ds}_{arm}_g{gamma}_seed{seed}.json``
    only. A T4 run therefore **overwrote** its T1 counterpart -- and
    ``missing_cells()`` then skipped every T4 cell as "already complete",
    because a T1 file sat exactly where the T4 output belonged. A T4 campaign
    launched after T1 would have run nothing and reported success.
    """
    paths = [CellSpec("CIC-IDS-2017", "v2_m1", 1.0, 0, **v).path
             for v in _VARIANTS]
    dupes = {p for p in paths if paths.count(p) > 1}
    assert not dupes, f"colliding paths: {dupes}"


def test_default_configuration_keeps_the_documented_t1_filename():
    """The config tag must be empty for defaults, or every filename recorded
    in the campaign documentation silently changes."""
    cell = CellSpec("CIC-IDS-2017", "v1_projected", 1.0, 0)
    assert cell.config_tag == ""
    assert cell.path.endswith("CIC_IDS_2017_v1_projected_g1.0_seed0.json")


def test_t4_paths_never_collide_with_t1_or_t2_paths():
    t12 = {c.path for c in enumerate_cells(["T1", "T2"])}
    t4 = {c.path for c in enumerate_cells(["T4"])}
    assert not (t12 & t4)


def test_t4_enumerates_the_plans_own_hundred_cells():
    """Master plan §7.2's T4 row: 100 runs = 2 datasets x 10 seeds x
    (2 p_drop + 3 epsilon)."""
    cells = enumerate_cells(["T4"])
    assert len(cells) == 100
    assert len({c.path for c in cells}) == 100
    assert {c.p_drop for c in cells if c.dp_epsilon is None} == set(T4_P_DROPS)
    assert {c.dp_epsilon for c in cells
            if c.dp_epsilon is not None} == set(T4_DP_EPSILONS)
    assert {c.gamma for c in cells} == {1.0}


def test_t4_m_selects_which_arm_the_robustness_sweep_runs_on():
    cells = enumerate_cells(["T4"], datasets=["UNSW-NB15"], seeds=[0], t4_m=3)
    assert {c.arm for c in cells} == {"v2_m3"}


def test_every_tier_together_enumerates_without_a_single_collision():
    cells = enumerate_cells(["T1", "T2", "T4"])
    assert len(cells) == 400
    assert len({c.path for c in cells}) == 400


# =========================================================================== #
# B5 -- DP is per-client and pre-SecAgg
# =========================================================================== #
def test_dp_noise_is_drawn_once_per_client_not_once_for_the_server():
    """The trainer used to apply **one** Laplace draw to the already-exact
    SecAgg-summed histogram, server-side. That is wrong twice:

    1. **Magnitude.** One draw gives ``sigma = (D1/eps)*sqrt(2)``; K
       per-client draws give ``(D1/eps)*sqrt(2K)`` -- the quantity
       :func:`aggregate_noise_std` models, that ``test_privacy.py`` asserts
       against the plan's §5.2 table, that T-9 specifies, and that
       ``verify_v2_plan_claims.py`` measured F3's recovery rates under. The
       server-side version is ``sqrt(K)`` times weaker than all of them, so a
       reported epsilon would over-claim privacy by that factor.
    2. **Trust model.** F4's adversary *is* the server. Noising a sum the
       server already holds exactly protects nothing from it.

    Counting draws is the sharpest available signal: ``K`` means per-client,
    ``1`` means server-side.
    """
    split = _make_split()
    K = split.n_clients
    cfg = FedConfig(rounds=1, gamma=0.5, n_rare=1, seed=0, dp_epsilon=0.5)
    trainer = FederatedTrainerV2(split, cfg, secagg_threshold=1)

    import fedgraa.trainer as trainer_mod

    calls = {"n": 0}
    real = trainer_mod.laplace_mechanism

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    trainer_mod.laplace_mechanism = counting
    try:
        trainer._histograms_and_tags(np.random.default_rng(0),
                                     bill_bytes=False, force_full=True)
    finally:
        trainer_mod.laplace_mechanism = real

    assert calls["n"] == K, (
        f"expected one Laplace draw per client ({K}), got {calls['n']}; "
        "a single draw means server-side noising of the aggregate"
    )


def test_no_dp_draws_happen_when_dp_is_disabled():
    split = _make_split()
    cfg = FedConfig(rounds=1, gamma=0.5, n_rare=1, seed=0, dp_epsilon=None)
    trainer = FederatedTrainerV2(split, cfg, secagg_threshold=1)

    import fedgraa.trainer as trainer_mod

    calls = {"n": 0}
    real = trainer_mod.laplace_mechanism

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    trainer_mod.laplace_mechanism = counting
    try:
        trainer._histograms_and_tags(np.random.default_rng(0),
                                     bill_bytes=False, force_full=True)
    finally:
        trainer_mod.laplace_mechanism = real
    assert calls["n"] == 0


@pytest.mark.parametrize("epsilon", [0.5, 1.0])
def test_trainer_end_to_end_dp_sigma_matches_the_closed_form(epsilon):
    """Asserts the noise the **trainer** actually delivers against T-9's own
    formula. ``test_privacy.py`` already asserted the standalone
    :func:`aggregate_noise_std` -- and it was correct the whole time the
    trainer was not. Testing the function instead of the code path is exactly
    how this defect survived.
    """
    split = _make_split()
    K = split.n_clients
    n_classes = split.spec.n_classes
    clean = _pooled_histogram(split, n_classes)

    cfg = FedConfig(rounds=1, gamma=0.5, n_rare=1, seed=0, dp_epsilon=epsilon)
    trainer = FederatedTrainerV2(split, cfg, secagg_threshold=1)

    import fedgraa.trainer as trainer_mod

    captured: list = []
    real_bottom = trainer_mod.bottom_n_rare_classes

    def capture(counts, attack_ids, n):
        captured.append(np.asarray(counts, dtype=np.float64).copy())
        return real_bottom(counts, attack_ids, n)

    trainer_mod.bottom_n_rare_classes = capture
    try:
        for trial in range(400):
            trainer._histograms_and_tags(
                np.random.default_rng(5000 + trial),
                bill_bytes=False, force_full=True,
            )
    finally:
        trainer_mod.bottom_n_rare_classes = real_bottom

    noised = np.stack(captured)
    expected = aggregate_noise_std(epsilon, K)
    # The server-side clip at 0 biases classes sitting near the boundary, so
    # measure sigma only where the true count is comfortably above it.
    far = clean > 5 * expected
    assert far.any(), "fixture has no class far enough from the clip boundary"

    empirical = float(np.std(noised[:, far] - clean[far]))
    assert empirical == pytest.approx(expected, rel=0.20), (
        f"trainer delivers sigma={empirical:.3f}, closed form says "
        f"{expected:.3f} (ratio {empirical / expected:.3f}); a ratio near "
        f"{1 / math.sqrt(K):.3f} means one server-side draw instead of K "
        "per-client draws"
    )


# =========================================================================== #
# B4 -- an unrecoverable SecAgg round aborts the round, not the run
# =========================================================================== #
def test_secagg_abort_skips_the_round_and_is_counted_rather_than_raised():
    """Under the production majority threshold, ``p_drop=0.3`` at K=9-14
    makes ~9-10% of rounds unrecoverable, so letting ``SecAggDropoutError``
    escape ``train()`` gives a 200-round run a ~1e-9 chance of finishing --
    all 40 ``p_drop=0.3`` cells in T4 would have failed. The real protocol
    aborts the round and continues; the abort *rate* is what a dropout study
    should report.
    """
    split = _make_split()
    K = split.n_clients
    cfg = FedConfig(rounds=6, gamma=0.5, n_rare=1, seed=0, p_drop=0.5)
    # threshold = K forces an abort on any round losing even one client.
    trainer = FederatedTrainerV2(split, cfg, secagg_threshold=K)

    hist = trainer.train(eval_every=2)  # must not raise

    assert len(hist.rounds) >= 3, "training did not run to completion"
    assert 0 < trainer.failed_secagg_rounds <= cfg.rounds, (
        "fixture never triggered an abort; this guard would be vacuous"
    )


def test_aborted_round_records_nan_zeta_not_a_stale_measurement():
    split = _make_split()
    K = split.n_clients
    cfg = FedConfig(rounds=4, gamma=0.5, n_rare=1, seed=0, p_drop=0.9)
    trainer = FederatedTrainerV2(split, cfg, secagg_threshold=K,
                                 measure_zeta=True)
    hist = trainer.train(eval_every=1)
    assert trainer.failed_secagg_rounds > 0
    assert any(math.isnan(z) for z in hist.zeta[1:]), (
        "an aborted round has no reconstructed gradient, so its zeta must be "
        "NaN rather than a value carried over from a different participant set"
    )


def test_weights_are_carried_forward_unchanged_across_an_aborted_round():
    """An aborted round yields no gradient, so it must yield no weight update
    -- not a partial or stale one."""
    split = _make_split()
    K = split.n_clients
    cfg = FedConfig(rounds=3, gamma=0.5, n_rare=1, seed=0, p_drop=0.99)
    trainer = FederatedTrainerV2(split, cfg, secagg_threshold=K)
    hist = trainer.train(eval_every=1)
    # With p_drop=0.99 essentially every round aborts, so w never moves off
    # its Static-Reparam initialization.
    assert trainer.failed_secagg_rounds == cfg.rounds
    for w in hist.weights[1:]:
        np.testing.assert_allclose(w, hist.weights[0], rtol=0, atol=0)


def test_bytes_billed_even_when_round_aborts():
    """Masked histogram shares are genuinely on the wire before the server
    attempts SecAgg reconstruction, so an aborted round (too few survivors
    to reconstruct) must still bill that traffic. Billing only after a
    successful ``aggregate()`` call would silently undercount
    ``bytes_sent`` on every T4 ``p_drop>0`` cell, where aborts are frequent
    by design."""
    split = _make_split()
    K = split.n_clients
    cfg = FedConfig(rounds=1, gamma=0.5, n_rare=1, seed=0, p_drop=0.99)
    trainer = FederatedTrainerV2(split, cfg, secagg_threshold=K)
    assert trainer.bytes_sent == 0
    with pytest.raises(SecAggDropoutError):
        trainer._histograms_and_tags(np.random.default_rng(0))
    assert trainer.bytes_sent > 0, (
        "an aborted round's already-submitted histogram shares must still "
        "be billed rather than silently dropped"
    )


def test_secagg_channel_contract_is_unweakened_by_the_trainer_side_handling():
    """T-7's guarantee lives in the channel: never silently return an
    under-counted sum. The abort handling is in the trainer, and must not
    have softened it."""
    ch = SecAggChannel(n_clients=5, threshold=4)
    ch.submit([np.array(1.0)])
    with pytest.raises(SecAggDropoutError):
        ch.aggregate()


def test_full_participation_reports_zero_aborted_rounds():
    split = _make_split()
    cfg = FedConfig(rounds=3, gamma=0.5, n_rare=1, seed=0, p_drop=0.0)
    trainer = FederatedTrainerV2(split, cfg)
    trainer.train(eval_every=3)
    assert trainer.failed_secagg_rounds == 0


def test_round_zero_uses_full_participation_so_it_cannot_abort():
    """Round 0 is a labelling step, explicitly not billed as protocol
    traffic. Letting it abort would fail a run before training begins."""
    split = _make_split()
    K = split.n_clients
    cfg = FedConfig(rounds=1, gamma=0.5, n_rare=1, seed=0, p_drop=0.95)
    trainer = FederatedTrainerV2(split, cfg, secagg_threshold=K)
    rare_ids, _tags, participants = trainer._histograms_and_tags(
        np.random.default_rng(0), bill_bytes=False, force_full=True,
    )
    assert len(participants) == K
    assert len(rare_ids) == cfg.n_rare


# =========================================================================== #
# H4 -- the frontier's "1x" anchor is a measured byte count
# =========================================================================== #
def test_v1_trainer_bills_bytes_so_the_frontier_anchor_is_measured():
    """The frontier figure's x-axis is total traffic. Billing only the v2
    arms left its leftmost point with no measured x-coordinate, so the ``1x``
    denominator every multiplier is expressed against would have been
    asserted rather than observed."""
    split = _make_split()
    cfg = FedConfig(rounds=3, gamma=1.0, n_rare=1, seed=0)
    v1 = FederatedTrainer(split, cfg, method="fedgraa")
    v1.train(eval_every=3)

    assert v1.comm_rounds == cfg.rounds
    assert v1.bytes_sent == 4 * 2 * v1.n_params * split.n_clients * cfg.rounds


@pytest.mark.parametrize("m,multiplier", [(1, 6), (5, 2)])
def test_measured_v2_over_v1_byte_ratio_matches_the_frontier_formula(
    m, multiplier
):
    """T-12's ``(1 + ceil(E'/m))x`` asserted as a ratio of two **measured**
    counters, rather than against a hand-derived baseline -- which is exactly
    the failure mode T-12 exists to prevent (the redesign doc's 2.4% UNSW
    overstatement)."""
    split = _make_split()

    v1 = FederatedTrainer(
        split, FedConfig(rounds=2, gamma=1.0, n_rare=1, seed=0),
        method="fedgraa",
    )
    v1.train(eval_every=2)

    v2 = FederatedTrainerV2(
        split,
        FedConfig(rounds=2, gamma=1.0, n_rare=1, seed=0, grad_refresh_every=m),
    )
    v2.train(eval_every=2)

    ratio = v2.bytes_sent / v1.bytes_sent
    assert ratio == pytest.approx(multiplier, rel=0.01), (
        f"m={m}: measured {ratio:.4f}x, formula says {multiplier}x"
    )


def test_v1_byte_instrumentation_does_not_touch_the_strict_reproduction_path():
    """The counters are additive only. T-19's guarantee -- that
    ``FederatedTrainer(method='fedgraa')`` still builds the **unprojected**
    weight learner -- must be untouched."""
    from fedgraa.aggregation import FedGRAA, ProjectedFedGRAA

    split = _make_split()
    trainer = FederatedTrainer(split, FedConfig(rounds=1, seed=0),
                               method="fedgraa")
    assert isinstance(trainer.fedgraa, FedGRAA)
    assert not isinstance(trainer.fedgraa, ProjectedFedGRAA)


# =========================================================================== #
# M6 -- results must be self-describing
# =========================================================================== #
def test_provenance_records_code_and_library_versions():
    """``requirements.txt`` pins lower bounds only, so the versions a run
    resolved are not recoverable from the repository -- they have to be
    captured per run or they are lost."""
    p = provenance()
    for key in ("git_commit", "git_dirty", "python", "platform",
                "tensorflow", "numpy", "scipy", "pandas", "sklearn"):
        assert key in p, f"provenance() is missing {key!r}"
    assert p["python"].startswith("3.")


def test_provenance_is_best_effort_and_never_raises(monkeypatch):
    """Provenance must never be the reason a multi-hour campaign cell fails
    -- e.g. inside a Modal container with no ``git`` binary."""
    import subprocess

    def _boom(*args, **kwargs):
        raise OSError("no git in this container")

    monkeypatch.setattr(subprocess, "run", _boom)
    p = provenance()
    assert p["git_commit"] is None
    assert p["tensorflow"] is not None


# =========================================================================== #
# M2 -- the secagg flag must be enforced, not silently ignored
# =========================================================================== #
def test_secagg_false_is_rejected_rather_than_silently_ignored():
    """``FedConfig.secagg`` was documented as "F4: SecAgg ... is mandatory"
    and then never read by any code path -- ``FederatedTrainerV2`` builds a
    ``SecAggChannel`` unconditionally. Setting it ``False`` changed nothing,
    while reading as though it disabled SecAgg."""
    FedConfig(secagg=True)  # the only supported value
    with pytest.raises(ValueError, match="secagg=False is not supported"):
        FedConfig(secagg=False)


# =========================================================================== #
# Verification finding, 2026-07-28: provenance must survive a container with
# no git (the Modal case) -- otherwise results carry no code identity at all.
# =========================================================================== #
def test_provenance_reads_the_commit_from_the_environment_first(monkeypatch):
    """Found by verifying the real Modal smoke test's output, not by reading.

    Both result JSONs came back with ``"git_commit": null``. The container
    ships a *copy* of the working tree, not a clone, and has no ``git``
    binary -- so the subprocess fallback could never succeed there, and
    ``provenance()``'s best-effort design turned that into a silent null on
    exactly the platform the campaign runs on. The launcher now resolves the
    commit locally and forwards it through the environment; this pins that
    the environment is consulted **before** git, so it still works where git
    cannot.
    """
    import subprocess

    def _no_git(*args, **kwargs):
        raise FileNotFoundError("git: command not found")

    monkeypatch.setattr(subprocess, "run", _no_git)
    monkeypatch.setenv("FEDGRAA_GIT_COMMIT", "deadbeefcafe1234")
    monkeypatch.setenv("FEDGRAA_GIT_DIRTY", "1")

    p = provenance()
    assert p["git_commit"] == "deadbeefcafe1234"
    assert p["git_dirty"] is True


def test_provenance_env_dirty_flag_parses_false(monkeypatch):
    monkeypatch.setenv("FEDGRAA_GIT_COMMIT", "abc123")
    monkeypatch.setenv("FEDGRAA_GIT_DIRTY", "0")
    p = provenance()
    assert p["git_commit"] == "abc123"
    assert p["git_dirty"] is False


def test_provenance_still_falls_back_to_git_when_no_env_is_set(monkeypatch):
    """Local runs have git and no env var -- that path must keep working."""
    monkeypatch.delenv("FEDGRAA_GIT_COMMIT", raising=False)
    monkeypatch.delenv("FEDGRAA_GIT_DIRTY", raising=False)
    p = provenance()
    assert p["git_commit"] is not None, "local run should resolve a commit"
    assert len(p["git_commit"]) == 40


# =========================================================================== #
# Verification finding V3, 2026-07-28: the Modal path enumerated cells but
# never consulted missing_cells(), so a relaunch re-ran every completed cell.
# =========================================================================== #
def test_should_skip_cell_is_the_one_shared_definition_of_already_done(tmp_path):
    """Found by reading the Modal launcher after the smoke test, not by a
    failing test: it called ``enumerate_cells`` and dispatched everything,
    never ``missing_cells``. Its own docstring claimed the campaign was
    resumable. It was, locally, and was not on Modal -- so a relaunch after a
    partial failure would have re-run every finished cell, defeating the
    property ``COMPUTE_INFRASTRUCTURE_2026-07-27.md`` §6.4 relies on to make
    preemption cost one cell rather than the campaign.

    Both drivers now route through this one function, so the definition of
    "already done" cannot drift between them again.
    """
    from scripts.run_v2_frontier_sweep import should_skip_cell

    cell = CellSpec("CIC-IDS-2017", "v2_m1", 1.0, 0, out_dir=str(tmp_path))

    assert not should_skip_cell(cell), "nothing written yet -- must not skip"

    os.makedirs(os.path.dirname(cell.path), exist_ok=True)
    with open(cell.path, "w", encoding="utf-8") as f:
        json.dump({"final_all_acc": 0.5, "final_last_n_acc": 0.4}, f)
    assert should_skip_cell(cell), "a complete result must be skipped"

    assert not should_skip_cell(cell, force=True), "--force must override"


def test_should_skip_cell_reruns_a_truncated_or_partial_result(tmp_path):
    """A half-written JSON (preemption mid-write) must be re-run, not
    mistaken for a finished cell."""
    from scripts.run_v2_frontier_sweep import should_skip_cell

    cell = CellSpec("UNSW-NB15", "v2_m2", 0.5, 3, out_dir=str(tmp_path))
    os.makedirs(os.path.dirname(cell.path), exist_ok=True)

    with open(cell.path, "w", encoding="utf-8") as f:
        f.write('{"final_all_acc": 0.5, "trunca')
    assert not should_skip_cell(cell), "invalid JSON must not count as done"

    with open(cell.path, "w", encoding="utf-8") as f:
        json.dump({"final_all_acc": 0.5}, f)  # missing final_last_n_acc
    assert not should_skip_cell(cell), "missing keys must not count as done"


def test_skip_decision_is_per_cell_not_per_tier(tmp_path):
    """Resumability must be cell-granular: completing some T1 cells must not
    cause the rest to be skipped."""
    from scripts.run_v2_frontier_sweep import should_skip_cell

    cells = enumerate_cells(["T1"], datasets=["UNSW-NB15"], seeds=[0, 1],
                            out_dir=str(tmp_path))
    os.makedirs(os.path.dirname(cells[0].path), exist_ok=True)
    with open(cells[0].path, "w", encoding="utf-8") as f:
        json.dump({"final_all_acc": 0.5, "final_last_n_acc": 0.4}, f)

    skipped = [c for c in cells if should_skip_cell(c)]
    assert len(skipped) == 1 and skipped[0].path == cells[0].path
