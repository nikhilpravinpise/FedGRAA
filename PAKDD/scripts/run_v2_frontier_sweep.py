"""Resumable driver for the FedGRAA-v2 exactness-communication frontier
campaign (FEDGRAA_V2_MASTER_PLAN.md §7.2 tiers T1/T2, §4).

**Status (2026-07-28).** The full campaign (T1: ~28 CPU-h; T1+T2: ~85 CPU-h)
has **not** been run. What *has* been verified end-to-end, on Modal, on real
CIC-IDS-2017 data, is the **campaign path** itself: two real cells (one per
arm, 2 rounds) executed in containers, wrote self-describing JSON to the
results Volume, and returned a measured byte ratio of ``6.0007x`` -- see
``docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md``, which also records the six
blocking defects found in this path before that was true. Every mechanism
this script exercises (``FederatedTrainerV2``, the two-phase exact protocol,
the byte/round instrumentation) is separately unit- and integration-tested
(T-4/T-4a/T-4b/T-12/T-13, ``tests/test_trainer_v2.py``). Those are two
different claims and this docstring keeps them apart deliberately.
This script is the harness that turns "the code works" into "the frontier
figure," structured so each ``(dataset, arm, gamma, seed)`` cell is one call
to :func:`run_one_cell` -- directly usable as the function body of a Modal
``.map()`` call, per the pattern validated in
``docs/research_c/COMPUTE_INFRASTRUCTURE_2026-07-27.md`` §6.

Arms:

* ``v1_projected`` -- the privacy-*violating* reference point (§4.2's
  leftmost figure point): ``FederatedTrainer(method="fedgraa")`` with its
  weight learner swapped for :class:`~fedgraa.aggregation.ProjectedFedGRAA`.
* ``v2_m{1,2,3,5}`` -- :class:`~fedgraa.trainer.FederatedTrainerV2` at
  ``grad_refresh_every in {1, 2, 3, 5}`` (F5's frontier knob), SecAgg always
  on (F4 mandatory), no DP/dropout unless explicitly requested via CLI flags.

**Both arms run at ``eta' = 0.01``** (:data:`DEFAULT_ETA_PRIME`), the Stage B
*combined-fix* value FEDGRAA_V2_MASTER_PLAN.md F1 specifies as v2's
comparison reference -- deliberately **not** ``FedConfig``'s own ``0.1``
default. See that constant's comment for why the difference is not cosmetic.

Tiers (§7.2):

* **T1 -- frontier.** Both arms above x 2 datasets x 10 seeds x gamma=1.0.
* **T2 -- gamma interaction.** T1's arms at gamma in {0, 0.5} as well.
* **T4 -- robustness controls.** One v2 arm (``--t4-m``) x 2 datasets x 10
  seeds x (``p_drop in {0.1, 0.3}`` + ``dp_epsilon in {1, 4, 10}``) = 100
  cells. T4 results never collide with T1/T2's because
  :attr:`CellSpec.config_tag` encodes the varied field into the filename.
* **T6 -- real non-IID cross-silo benchmark.** All five arms x 10 seeds x
  gamma=1.0, on NF-UQ-NIDS-v2 (a real 4-silo dataset, loaded and partitioned
  by a different path than CIC/UNSW -- see :data:`NF_UQ_NIDS_V2_DATASET`).
  Needs the loader's ``max_per_class`` cap (B6,
  ``docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md``) to avoid an ~88 GB load of
  the real ~76M-row release.

Usage::

    # Dry run: show what would be launched without running anything.
    python scripts/run_v2_frontier_sweep.py --tier T1 --dry-run

    # Actually run T1 locally, 3 worker processes (spawn context forced).
    python scripts/run_v2_frontier_sweep.py --tier T1 --workers 3

    # T1+T2 for one dataset only.
    python scripts/run_v2_frontier_sweep.py --tier T1 T2 --datasets CIC-IDS-2017

    # T4 robustness, once T1 has identified the best refresh interval.
    python scripts/run_v2_frontier_sweep.py --tier T4 --t4-m 2

To run this on Modal -- the intended path for the real campaign -- use
``scripts/modal_frontier_campaign.py``, which imports :func:`run_one_cell`
and :func:`enumerate_cells` from this module unchanged.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_OUT_DIR = "results/v2_frontier"
DEFAULT_DATASETS = ["CIC-IDS-2017", "UNSW-NB15"]
DEFAULT_SEEDS = list(range(10))
T1_GAMMAS = [1.0]
T2_EXTRA_GAMMAS = [0.0, 0.5]
FRONTIER_MS = [1, 2, 3, 5]
ARM_NAMES = ["v1_projected"] + [f"v2_m{m}" for m in FRONTIER_MS]

# FEDGRAA_V2_MASTER_PLAN.md F1: "Everywhere this plan says 'v1' as v2's
# comparison reference, it means v1-projected (projection + eta'=0.01, the
# Stage B *combined-fix* arm)". FedConfig's own default is 0.1 -- the paper's
# literal value -- which with projection is Stage B's *projection-alone* arm,
# measured at 0.0% rare-class accuracy on CIC-IDS-2017 at gamma <= 0.5
# (docs/current/fedgraa_convergence_defect.md Sec 5.3). Running the campaign
# at that value would anchor the frontier on an arm this project already
# established as broken, so the campaign pins the combined-fix value
# explicitly for BOTH arms -- holding eta' fixed across the frontier so the
# only variable is the refresh interval m and the federated protocol.
DEFAULT_ETA_PRIME = 0.01

# T4 (robustness controls, FEDGRAA_V2_MASTER_PLAN.md Sec 7.2): p_drop in
# {0.1, 0.3} and DP at eps in {1, 4, 10}, on the best m. 2 datasets x 10 seeds
# x (2 p_drop + 3 eps) = 100 runs, matching the plan's own T4 row exactly.
T4_P_DROPS = [0.1, 0.3]
T4_DP_EPSILONS = [1.0, 4.0, 10.0]
T4_DEFAULT_M = 1

# T6 (real non-IID cross-silo benchmark, master plan Sec 7.2/7.4). NF-UQ-NIDS-v2
# is loaded and partitioned by a DIFFERENT code path than CIC/UNSW --
# load_nf_uq_nids_v2_silos + partition_by_silo, not load_dataset +
# partition_non_iid -- so it is never mixed into the ``datasets`` list T1/T2/T4
# iterate over; it gets its own tier branch in enumerate_cells and its own
# dataset-detection branch in run_one_cell.
#
# B6 (docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md): an uncapped load of the
# real ~76M-row release measured ~88 GB peak memory. 10_000 matches
# DatasetSpec.frequent_cap's existing convention and is what
# fedgraa.data.load_nf_uq_nids_v2_silos's own safety guard recommends in its
# error message.
NF_UQ_NIDS_V2_DATASET = "NF-UQ-NIDS-v2"
NF_UQ_NIDS_V2_CSV_RELPATH = os.path.join(
    "NF-UQ-NIDS-v2", "NF-UQ-NIDS-v2.csv",
)
NF_UQ_NIDS_V2_DEFAULT_MAX_PER_CLASS = 10_000
# T6 is "T1 repeated on the cross-silo benchmark" (master plan Sec 7.2), so it
# reuses T1's arm grid and gamma=1.0 -- but at K=4 silos rather than T1's
# K=9/14 synthetic clients, since NF-UQ-NIDS-v2 has exactly 4 constituent
# datasets (docs/research_c/REMAINING_COMPUTE_WORK.md Sec 5.4's own note that
# this makes T6 a different experiment, not a literal repeat).

# T3/T5 (external-baseline reproductions, REMAINING_COMPUTE_WORK.md Sec 4):
# ~14 other papers' methods, implemented as (split, cfg) -> TrainHistory
# classes with the exact same shape as fedgraa/baselines.py's FedProxTrainer
# et al. Each entry maps an arm name to the (module, class) that implements
# it; imported lazily inside run_one_cell so this module stays cheap to
# import/enumerate without pulling in every baseline's dependencies.
#
# FedPGO -- Bikila & Capek, "FedPGO: A personalized federated learning
# framework with gradient orthogonalization for IoT intrusion detection,"
# Computer Networks 280:112210, 2026-03-10, DOI 10.1016/j.comnet.2026.112210
# -- IS implemented, in fedgraa/baselines_t5_fedpgo.py, as a faithful
# implementation from the paper's PRIMARY TEXT, local at
# docs/research_c/FedPGO/FedPGO.md since 2026-07-30 (citation triple-confirmed
# via DBLP journals/cn/BikilaC26, OpenAlex W7134986038, Semantic Scholar
# CorpusId:286471251). The earlier "disclosed RECONSTRUCTION" proxy (from the
# literature review's mechanism summary only) was replaced 2026-07-31 per
# STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md Path 1 (D1); its 20 frontier
# cells are archived at
# results/v2_frontier/_archive_fedpgo_proxy_2026-07-30/.
T3_ARM_CLASSES = {
    "celm": ("fedgraa.baselines_t3_contrib", "CELMTrainer"),
    "spectralfed": ("fedgraa.baselines_t3_contrib", "SpectralFedTrainer"),
    "coast": ("fedgraa.baselines_t3_contrib", "CoAstTrainer"),
    "fltrust": ("fedgraa.baselines_t3_trust", "FLTrustTrainer"),
    "raga": ("fedgraa.baselines_t3_trust", "RAGATrainer"),
    "fedtruth": ("fedgraa.baselines_t3_trust", "FedTruthTrainer"),
}
T5_ARM_CLASSES = {
    "fedpgo": ("fedgraa.baselines_t5_fedpgo", "FedPGOTrainer"),
    "fedmade": ("fedgraa.baselines_t5_nids", "FedMADETrainer"),
    "sbfednids": ("fedgraa.baselines_t5_nids", "SBFedNIDSTrainer"),
    "fedapa": ("fedgraa.baselines_t5_nids", "FedAPATrainer"),
    "fedawa": ("fedgraa.baselines_t5_adaptive", "FedAWATrainer"),
    "dapfl": ("fedgraa.baselines_t5_adaptive", "DAPFLTrainer"),
    "fedaware": ("fedgraa.baselines_t5_adaptive", "FedAWARETrainer"),
    "adaaggrl": ("fedgraa.baselines_t5_adaptive", "AdaAggRLTrainer"),
}
# TC -- classic (non-personalization, non-NIDS-specific) FL optimizer
# baselines (docs/research_c/STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md
# §4): FedAvg/FedProx/FedOpt already existed elsewhere in this codebase but
# had never run through this v2-harness dispatch path; SCAFFOLD/FedNova are
# new, from-scratch reproductions (fedgraa/baselines_classic.py).
TC_ARM_CLASSES = {
    "fedavg": ("fedgraa.baselines_classic", "FedAvgTrainer"),
    "fedprox": ("fedgraa.baselines", "FedProxTrainer"),
    "fedopt": ("fedgraa.baselines", "FedOptServer"),
    "scaffold": ("fedgraa.baselines_classic", "SCAFFOLDTrainer"),
    "fednova": ("fedgraa.baselines_classic", "FedNovaTrainer"),
}
BASELINE_ARM_CLASSES: Dict[str, Tuple[str, str]] = {
    **T3_ARM_CLASSES, **T5_ARM_CLASSES, **TC_ARM_CLASSES,
}

# T7 -- adversarial robustness
# (docs/research_c/STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md §2;
# judgment calls in STAGE_C_T7_JUDGMENT_CALLS_2026-08-02.md). Cross-product
# arms are named "{defense}_{attack}"; each defense is one of
# T7_DEFENSE_CLASSES's (module, class) pairs, constructed with the extra
# attack_fn/attacker_fraction kwargs run_one_cell wires in (not a plain
# 2-arg BASELINE_ARM_CLASSES dispatch, since these need attack injection).
# JC-T7-16: AdaAggRL only ever pairs with labelflip (its own T5
# implementation stays unmodified). JC-T7-18: rare_class_signature (D1c) is
# never part of the defense cross-product -- it is its own arm,
# "v2_raresig", wired directly into FederatedTrainerV2 (undefended, D3).
T7_DEFENSE_CLASSES: Dict[str, Tuple[str, str]] = {
    "secfednids": ("fedgraa.baselines_t7_adversarial", "SecFedNIDSTrainer"),
    "weidetect": ("fedgraa.baselines_t7_adversarial", "WeiDetectTrainer"),
    "adaaggrl": ("fedgraa.baselines_t5_adaptive", "AdaAggRLTrainer"),
}
T7_LABEL_ATTACKS: Tuple[str, ...] = ("labelflip",)
T7_PARAM_ATTACKS: Tuple[str, ...] = ("modelpoison", "adaptive")
T7_DEFAULT_ATTACKER_FRACTION = 0.3
T7_SECOND_ATTACKER_FRACTION = 0.5
T7_RARESIG_ARM = "v2_raresig"


def t7_cross_product_arms() -> List[str]:
    arms = []
    for defense in ("secfednids", "weidetect"):
        for attack in T7_LABEL_ATTACKS + T7_PARAM_ATTACKS:
            arms.append(f"{defense}_{attack}")
    arms.append("adaaggrl_labelflip")
    return arms


T7_ARM_NAMES: List[str] = t7_cross_product_arms() + [T7_RARESIG_ARM]


def _t7_param_attack_fn(attack: str, rng, n_classes: int):
    """Build the uniform ``(P_current, P_global) -> P_poisoned`` attack_fn
    for a T7 param-space attack name (``modelpoison``/``adaptive``).

    **JC-T7-19 (disclosed here, not the master JC table, since it's a
    campaign-wiring detail rather than a mechanism-fidelity one).** The
    adaptive attacker (D1d) optimizes against an L2-magnitude proxy for
    detectability -- smaller perturbations look more benign to both
    SecFedNIDS's SOS-distance-based score and WeiDetect's F1-based score --
    not each specific defense's own full scoring function, which needs
    cross-client context (the whole round's client population) a single
    attacker's local optimization does not have access to in this
    per-client attack_fn's call signature.
    """
    from fedgraa.attacks import adaptive_attack, model_poison_attack, wrap_model_poison_attack

    if attack == "modelpoison":
        return wrap_model_poison_attack(rng, n_classes=n_classes, scale=10.0)
    if attack == "adaptive":
        def _fn(P_current, P_global):
            delta = [c - g for c, g in zip(P_current, P_global)]

            def base(scale):
                return model_poison_attack(delta, P_global, n_classes, scale, rng)

            def defense_proxy(candidate_delta):
                return sum(float((x ** 2).sum()) for x in candidate_delta)

            best_delta, _ = adaptive_attack(
                base, {}, [("scale", [2.0, 5.0, 10.0, 20.0])], defense_proxy,
                n_restarts=4, rng=rng,
            )
            return [d + g for d, g in zip(best_delta, P_global)]
        return _fn
    raise ValueError(f"unrecognized T7 param attack {attack!r}")


@dataclass(frozen=True)
class CellSpec:
    """One ``(dataset, arm, gamma, seed, +config)`` cell -- the atomic unit of
    work, designed to be the exact input a Modal ``.map()`` call would fan out
    over (one cell = one container, per
    ``COMPUTE_INFRASTRUCTURE_2026-07-27.md`` §6.2's guidance to scale via
    containers, not per-container concurrency, for this CPU-bound workload).
    """

    dataset: str
    arm: str
    gamma: float
    seed: int
    rounds: int = 200
    n_rare: int = 3
    eval_every: int = 20
    data_dir: str = "data"
    cache_dir: str = ".cache"
    p_drop: float = 0.0
    dp_epsilon: Optional[float] = None
    eta_prime: float = DEFAULT_ETA_PRIME
    secagg_threshold: Optional[int] = None
    # T6 only (dataset == NF_UQ_NIDS_V2_DATASET): the per-(silo,class) cap
    # fedgraa.data.load_nf_uq_nids_v2_silos bounds memory with (B6). Ignored
    # for CIC-IDS-2017/UNSW-NB15 cells.
    nf_uq_max_per_class: Optional[int] = None
    out_dir: str = DEFAULT_OUT_DIR
    # T7 only: fraction of clients that are attackers. Part of config_tag
    # (below) since T7_RARESIG_ARM's two attacker-fraction points would
    # otherwise collide on the same (dataset, arm, seed) filename.
    attacker_fraction: float = 0.0

    @property
    def grad_refresh_every(self) -> Optional[int]:
        if self.arm.startswith("v2_m"):
            return int(self.arm.split("_m")[1])
        # v1_projected and every T3/T5 external-baseline arm: not a FedGRAA-v2
        # refresh-interval arm, so this field is inert for them.
        return None

    @property
    def config_tag(self) -> str:
        """Filename suffix distinguishing cells that differ only in an
        *experimental* field.

        Without this, a T4 cell (``p_drop=0.3``) and its T1 counterpart share
        a path: the T4 run would overwrite the T1 result, and
        :func:`missing_cells` would then skip every T4 cell as "already
        complete" because a T1 file sits where its output belongs. Empty for
        the default configuration, so T1/T2 filenames stay exactly as the
        campaign documents record them.
        """
        parts: List[str] = []
        if self.p_drop:
            parts.append(f"pdrop{self.p_drop}")
        if self.dp_epsilon is not None:
            parts.append(f"eps{self.dp_epsilon}")
        if self.eta_prime != DEFAULT_ETA_PRIME:
            parts.append(f"etap{self.eta_prime}")
        if self.n_rare != 3:
            parts.append(f"n{self.n_rare}")
        if self.secagg_threshold is not None:
            parts.append(f"thr{self.secagg_threshold}")
        if self.rounds != 200:
            parts.append(f"r{self.rounds}")
        if self.nf_uq_max_per_class is not None:
            parts.append(f"cap{self.nf_uq_max_per_class}")
        if self.attacker_fraction:
            parts.append(f"af{self.attacker_fraction}")
        return ("_" + "_".join(parts)) if parts else ""

    @property
    def path(self) -> str:
        ds_tag = self.dataset.replace("-", "_")
        return os.path.join(
            self.out_dir,
            f"{ds_tag}_{self.arm}_g{self.gamma}_seed{self.seed}"
            f"{self.config_tag}.json",
        )


def enumerate_cells(
    tiers: Sequence[str],
    datasets: Sequence[str] = DEFAULT_DATASETS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    rounds: int = 200,
    n_rare: int = 3,
    eval_every: int = 20,
    data_dir: str = "data",
    cache_dir: str = ".cache",
    p_drop: float = 0.0,
    dp_epsilon: Optional[float] = None,
    eta_prime: float = DEFAULT_ETA_PRIME,
    secagg_threshold: Optional[int] = None,
    t4_m: int = T4_DEFAULT_M,
    nf_uq_max_per_class: int = NF_UQ_NIDS_V2_DEFAULT_MAX_PER_CLASS,
    out_dir: str = DEFAULT_OUT_DIR,
    arms: Sequence[str] = (),
) -> List[CellSpec]:
    """Every cell for the requested tiers, in a stable order.

    T2 does not duplicate T1's gamma=1.0 cells -- passing both tiers together
    yields exactly gamma in {0, 0.5, 1.0}, not {1.0} twice.

    **T4** is a different shape from T1/T2: it is not "all arms at another
    gamma" but a single v2 arm (``t4_m``, which should be set to whichever m
    T1 identifies as best) swept over ``p_drop`` and ``dp_epsilon``
    separately, at gamma=1.0. Its 100 cells never collide with T1/T2's
    because :attr:`CellSpec.config_tag` puts the varied field in the
    filename.

    **T6** is a different shape again: one dataset
    (:data:`NF_UQ_NIDS_V2_DATASET`, never mixed into ``datasets``), all five
    arms, gamma=1.0, over ``seeds`` -- T1's own grid, applied to the one
    real-silo dataset instead of the two synthetic-partition ones.
    ``nf_uq_max_per_class`` bounds the loader's memory (B6); it has no effect
    on any other tier's cells.

    **T3/T5/TC** are external-baseline reproductions, not FedGRAA-v2 arms:
    each is one of :data:`T3_ARM_CLASSES`/:data:`T5_ARM_CLASSES`/
    :data:`TC_ARM_CLASSES`'s (module, class) pairs x ``datasets`` x
    ``seeds``, at a fixed gamma=1.0 (gamma is a FedGRAA-specific
    composite-gradient knob these methods don't have; it is carried on the
    cell only so :class:`CellSpec` stays one uniform shape). TC is the
    classic-FL-optimizer set (FedAvg/FedProx/FedOpt/SCAFFOLD/FedNova, §4).

    ``arms`` optionally restricts every tier to the named arm(s) -- e.g.
    ``arms=["fedpgo"]`` for a single-arm rerun. The filter is applied
    uniformly across tiers *after* enumeration, so an empty tuple (the
    default) means "all arms of the tier" and existing callers such as
    ``check_stage_c_complete.py`` are byte-for-byte unaffected.
    """
    def _common(**overrides) -> dict:
        base = dict(
            rounds=rounds, n_rare=n_rare, eval_every=eval_every,
            data_dir=data_dir, cache_dir=cache_dir, eta_prime=eta_prime,
            secagg_threshold=secagg_threshold, out_dir=out_dir,
        )
        base.update(overrides)
        return base

    cells: List[CellSpec] = []

    gammas: List[float] = []
    if "T1" in tiers:
        gammas += list(T1_GAMMAS)
    if "T2" in tiers:
        gammas += T2_EXTRA_GAMMAS

    if gammas:
        for dataset in datasets:
            for arm in ARM_NAMES:
                for gamma in gammas:
                    for seed in seeds:
                        cells.append(CellSpec(
                            dataset=dataset, arm=arm, gamma=gamma, seed=seed,
                            **_common(p_drop=p_drop, dp_epsilon=dp_epsilon),
                        ))

    if "T4" in tiers:
        arm = f"v2_m{t4_m}"
        for dataset in datasets:
            for pd_ in T4_P_DROPS:
                for seed in seeds:
                    cells.append(CellSpec(
                        dataset=dataset, arm=arm, gamma=1.0, seed=seed,
                        **_common(p_drop=pd_, dp_epsilon=None),
                    ))
            for eps in T4_DP_EPSILONS:
                for seed in seeds:
                    cells.append(CellSpec(
                        dataset=dataset, arm=arm, gamma=1.0, seed=seed,
                        **_common(p_drop=0.0, dp_epsilon=eps),
                    ))

    if "T6" in tiers:
        for arm in ARM_NAMES:
            for seed in seeds:
                cells.append(CellSpec(
                    dataset=NF_UQ_NIDS_V2_DATASET, arm=arm, gamma=1.0,
                    seed=seed, nf_uq_max_per_class=nf_uq_max_per_class,
                    **_common(p_drop=0.0, dp_epsilon=None),
                ))

    if "T3" in tiers:
        for dataset in datasets:
            for arm in T3_ARM_CLASSES:
                for seed in seeds:
                    cells.append(CellSpec(
                        dataset=dataset, arm=arm, gamma=1.0, seed=seed,
                        **_common(p_drop=0.0, dp_epsilon=None),
                    ))

    if "T5" in tiers:
        for dataset in datasets:
            for arm in T5_ARM_CLASSES:
                for seed in seeds:
                    cells.append(CellSpec(
                        dataset=dataset, arm=arm, gamma=1.0, seed=seed,
                        **_common(p_drop=0.0, dp_epsilon=None),
                    ))

    if "TC" in tiers:
        for dataset in datasets:
            for arm in TC_ARM_CLASSES:
                for seed in seeds:
                    cells.append(CellSpec(
                        dataset=dataset, arm=arm, gamma=1.0, seed=seed,
                        **_common(p_drop=0.0, dp_epsilon=None),
                    ))

    if "T7" in tiers:
        # Defense cross-product: 1 attacker fraction (JC-T7-18's final grid).
        for dataset in datasets:
            for arm in t7_cross_product_arms():
                for seed in seeds:
                    cells.append(CellSpec(
                        dataset=dataset, arm=arm, gamma=1.0, seed=seed,
                        attacker_fraction=T7_DEFAULT_ATTACKER_FRACTION,
                        **_common(p_drop=0.0, dp_epsilon=None),
                    ))
            # D1c trend addendum: undefended FedGRAA-v2 only, 2 fractions.
            for frac in (T7_DEFAULT_ATTACKER_FRACTION, T7_SECOND_ATTACKER_FRACTION):
                for seed in seeds:
                    cells.append(CellSpec(
                        dataset=dataset, arm=T7_RARESIG_ARM, gamma=1.0, seed=seed,
                        attacker_fraction=frac,
                        **_common(p_drop=0.0, dp_epsilon=None),
                    ))

    if arms:
        wanted = set(arms)
        cells = [c for c in cells if c.arm in wanted]

    return cells


def provenance() -> Dict[str, Optional[str]]:
    """Everything needed to say *which code and which libraries* produced a
    result, recorded into every cell's JSON.

    ``requirements.txt`` pins lower bounds only, so the library versions a
    container resolves are not fixed by the repository -- they have to be
    captured per run or they are lost. The commit hash matters for the same
    reason: a campaign launched from a dirty tree has no other way to say
    what it ran.
    """
    import platform

    def _version(mod_name: str) -> Optional[str]:
        try:
            import importlib
            return getattr(importlib.import_module(mod_name), "__version__", None)
        except Exception:  # noqa: BLE001 -- provenance must never fail a run
            return None

    # Environment first, git second.
    #
    # A Modal container has no ``git`` binary and no ``.git`` directory -- the
    # image ships a copy of the working tree, not a clone. Relying on the
    # subprocess call alone therefore produced ``git_commit: null`` on every
    # real campaign cell, silently dropping the one field that says *which
    # code* generated a result. (Verified: the 2026-07-28 Modal smoke test's
    # two result JSONs both carried a null commit.) The launcher runs on a
    # machine that *does* have git, so it resolves these once and forwards
    # them through the environment; the git path below remains for local runs.
    commit: Optional[str] = os.environ.get("FEDGRAA_GIT_COMMIT") or None
    dirty_env = os.environ.get("FEDGRAA_GIT_DIRTY")
    dirty: Optional[bool] = (
        dirty_env.lower() in {"1", "true", "yes"} if dirty_env is not None
        else None
    )

    if commit is None:
        try:
            import subprocess
            repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                text=True, timeout=10,
            ).stdout.strip() or None
            dirty = bool(subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo, capture_output=True,
                text=True, timeout=10,
            ).stdout.strip())
        except Exception:  # noqa: BLE001 -- provenance must never fail a run
            pass

    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": _version("tensorflow"),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "pandas": _version("pandas"),
        "sklearn": _version("sklearn"),
    }


def _is_complete(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return "final_all_acc" in data and "final_last_n_acc" in data


def should_skip_cell(cell: "CellSpec", force: bool = False) -> bool:
    """Whether this cell's work is already done and can be skipped.

    Factored out so the local driver and the Modal driver
    (``scripts/modal_frontier_campaign.py``) share one definition of
    "already done" and it can be tested without a Modal container. The Modal
    path originally enumerated cells but never consulted
    :func:`missing_cells`, so a relaunch re-ran every completed cell --
    silently defeating the resumability that
    ``COMPUTE_INFRASTRUCTURE_2026-07-27.md`` §6.4 relies on to make preemption
    cost one cell instead of the campaign
    (``docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md``, verification finding V3).
    """
    return (not force) and _is_complete(cell.path)


def missing_cells(cells: Sequence[CellSpec]) -> List[CellSpec]:
    """Resumability: skip any cell whose result JSON is already complete --
    matching ``scripts/run_ablation_seed_sweep.py``'s own contract, so an
    interrupted Modal campaign (preemption is expected and auto-retried per
    ``COMPUTE_INFRASTRUCTURE_2026-07-27.md`` §6.4, but a whole-job restart
    should still not redo completed cells)."""
    return [c for c in cells if not _is_complete(c.path)]


# --------------------------------------------------------------------------- #
# Execution -- one cell, in-process (the Modal `.map()` function body)
# --------------------------------------------------------------------------- #
def run_one_cell(cell: CellSpec) -> Dict:
    """Run exactly one ``(dataset, arm, gamma, seed)`` cell and persist its
    result to ``cell.path``. Import-safe for use as a Modal function body:
    heavy imports (TensorFlow via ``fedgraa``) happen inside this function,
    not at module scope, so the module itself stays importable (and cheap to
    inspect/enumerate) without a TensorFlow install.
    """
    from fedgraa.aggregation import ProjectedFedGRAA
    from fedgraa.config import FedConfig
    from fedgraa.model import init_params
    from fedgraa.trainer import FederatedTrainer, FederatedTrainerV2

    os.makedirs(cell.out_dir, exist_ok=True)

    if cell.dataset == NF_UQ_NIDS_V2_DATASET:
        # T6: a real multi-silo dataset, loaded/partitioned by an entirely
        # different path than CIC/UNSW's synthetic one-attack-per-client
        # construction -- see NF_UQ_NIDS_V2_DATASET's module-level comment.
        import numpy as np

        from fedgraa.config import DatasetSpec
        from fedgraa.data import load_nf_uq_nids_v2_silos
        from fedgraa.partition import partition_by_silo

        csv_path = os.path.join(
            cell.data_dir, NF_UQ_NIDS_V2_CSV_RELPATH,
        )
        silos, labels = load_nf_uq_nids_v2_silos(
            csv_path,
            max_per_class=cell.nf_uq_max_per_class
            or NF_UQ_NIDS_V2_DEFAULT_MAX_PER_CLASS,
            seed=cell.seed,
            chunksize=2_000_000,
            dtype=np.float32,
        )
        # Measured, not hardcoded (F3/F7's own discipline, applied to a real
        # dataset this time instead of the synthetic one-attack-per-client
        # construction): used_counts and benign_id both come from what the
        # loader actually returned.
        used_counts = [0] * len(labels)
        for _X, y_silo, _name in silos:
            for cid in y_silo:
                used_counts[cid] += 1
        benign_id = next(
            (i for i, lbl in enumerate(labels) if lbl.strip().lower() == "benign"),
            0,
        )
        spec = DatasetSpec(
            name=NF_UQ_NIDS_V2_DATASET,
            n_features=silos[0][0].shape[1],
            labels=labels,
            used_counts=used_counts,
            benign_id=benign_id,
            # No rare_label_order -- there is no oracle attack list for a
            # real dataset, so rare_client_ids() falls back to size-ranking
            # used_counts (F7's non-oracle path), exactly as a genuine
            # deployment must.
        )
        cfg = FedConfig(
            rounds=cell.rounds, gamma=cell.gamma, n_rare=cell.n_rare,
            seed=cell.seed, eta_prime=cell.eta_prime,
            grad_refresh_every=cell.grad_refresh_every or 1,
            p_drop=cell.p_drop, dp_epsilon=cell.dp_epsilon,
        )
        split = partition_by_silo(
            silos, spec, n_rare=cfg.n_rare, val_fraction=cfg.val_fraction,
            seed=cfg.seed,
        )
    else:
        from fedgraa.data import load_dataset
        from fedgraa.partition import partition_non_iid

        data_dir = (
            os.path.join(cell.data_dir, cell.dataset)
            if cell.data_dir and os.path.isdir(os.path.join(cell.data_dir, cell.dataset))
            else cell.data_dir
        )
        X, y, spec = load_dataset(
            cell.dataset, data_dir=data_dir, seed=cell.seed,
            cache_dir=cell.cache_dir,
        )
        cfg = FedConfig(
            rounds=cell.rounds, gamma=cell.gamma, n_rare=cell.n_rare,
            seed=cell.seed, eta_prime=cell.eta_prime,
            grad_refresh_every=cell.grad_refresh_every or 1,
            p_drop=cell.p_drop, dp_epsilon=cell.dp_epsilon,
        )
        split = partition_non_iid(
            X, y, spec, n_rare=cfg.n_rare, val_fraction=cfg.val_fraction,
            seed=cfg.seed,
        )

    failed_secagg_rounds = 0
    attacker_weight_history: List[float] = []
    if cell.arm == "v1_projected":
        trainer = FederatedTrainer(split, cfg, method="fedgraa")
        layer_sizes = spec.model_config().layer_sizes
        trainer.fedgraa = ProjectedFedGRAA(
            gamma=cfg.gamma, eta_prime=cfg.eta_prime,
            weight_epochs=cfg.weight_epochs, layer_sizes=layer_sizes,
        )
        history = trainer.train(eval_every=cell.eval_every, verbose=True)
        bytes_sent_out = trainer.bytes_sent
        comm_rounds_out = trainer.comm_rounds
    elif cell.arm.startswith("v2_m"):
        trainer = FederatedTrainerV2(
            split, cfg,
            secagg_threshold=cell.secagg_threshold,
            measure_zeta=True,
        )
        history = trainer.train(eval_every=cell.eval_every, verbose=True)
        failed_secagg_rounds = trainer.failed_secagg_rounds
        bytes_sent_out = trainer.bytes_sent
        comm_rounds_out = trainer.comm_rounds
    elif cell.arm == T7_RARESIG_ARM:
        # D1c trend addendum: undefended FedGRAA-v2 itself, under the
        # mechanism-specific attack (JC-T7-13/15/18, §2.6 step 5).
        trainer = FederatedTrainerV2(
            split, cfg,
            secagg_threshold=cell.secagg_threshold,
            measure_zeta=True,
            raresig_attacker_fraction=cell.attacker_fraction,
        )
        history = trainer.train(eval_every=cell.eval_every, verbose=True)
        failed_secagg_rounds = trainer.failed_secagg_rounds
        bytes_sent_out = trainer.bytes_sent
        comm_rounds_out = trainer.comm_rounds
        attacker_weight_history = trainer.attacker_weight_history
    elif cell.arm in t7_cross_product_arms():
        # T7 defense-reproduction cross-product: label-flip poisons
        # split.clients[i].y_train upstream (applies to every trainer with
        # zero modification, JC-T7-16); param-space attacks (modelpoison,
        # adaptive) are wired via the uniform (P_current, P_global) ->
        # P_poisoned attack_fn contract, passed only to the two new T7
        # trainers (AdaAggRL only ever pairs with labelflip, JC-T7-16).
        import importlib

        import numpy as np

        from fedgraa.attacks import label_flip_attack

        defense, attack = cell.arm.split("_", 1)
        n_clients = len(split.clients)
        n_attackers = int(round(cell.attacker_fraction * n_clients))
        attacker_ids = set(
            np.random.default_rng(cfg.seed).choice(
                n_clients, size=n_attackers, replace=False,
            ).tolist()
        ) if n_attackers > 0 else set()

        rng = np.random.default_rng(cfg.seed + 1)
        if attack in T7_LABEL_ATTACKS:
            n_classes = spec.n_classes
            flip_map = {c: spec.benign_id for c in range(n_classes) if c != spec.benign_id}
            for i in attacker_ids:
                split.clients[i].y_train = label_flip_attack(
                    split.clients[i].y_train, flip_map, fraction=1.0, rng=rng,
                )
            attack_fn = None
        else:
            attack_fn = _t7_param_attack_fn(attack, rng, spec.n_classes)

        module_name, class_name = T7_DEFENSE_CLASSES[defense]
        trainer_cls = getattr(importlib.import_module(module_name), class_name)
        if defense == "adaaggrl":
            trainer = trainer_cls(split, cfg)  # unmodified T5 class, JC-T7-16
        else:
            trainer = trainer_cls(
                split, cfg, attack_fn=attack_fn, attacker_fraction=cell.attacker_fraction,
            )
        history = trainer.train(eval_every=cell.eval_every, verbose=True)
        n_params = sum(
            int(p.size) for p in init_params(spec.model_config().layer_sizes, seed=0)
        )
        bytes_sent_out = 4 * 2 * n_params * len(split.clients) * cell.rounds
        comm_rounds_out = cell.rounds
        attacker_weight_history = getattr(trainer, "attack_success_rate_history", [])
    elif cell.arm in BASELINE_ARM_CLASSES:
        # T3/T5/TC external-baseline reproductions (fedgraa/baselines_t3_*.py,
        # baselines_t5_*.py, baselines_classic.py, baselines.py): none of
        # these do SecAgg or a two-phase protocol -- every one is plain
        # single-phase, one-download + one-upload-per-client-per-round
        # communication, identical in shape to v1_projected's own billing
        # above. None of these Trainer classes instrument bytes internally
        # (unlike FederatedTrainer/FederatedTrainerV2), so the same formula
        # is applied analytically here rather than silently leaving the
        # frontier's x-axis undefined for any of them (19 arms as of §4:
        # 6 T3 + 8 T5 + 5 TC = 19, BASELINE_ARM_CLASSES).
        import importlib

        module_name, class_name = BASELINE_ARM_CLASSES[cell.arm]
        trainer_cls = getattr(importlib.import_module(module_name), class_name)
        trainer = trainer_cls(split, cfg)
        history = trainer.train(eval_every=cell.eval_every, verbose=True)
        n_params = sum(
            int(p.size) for p in init_params(spec.model_config().layer_sizes, seed=0)
        )
        bytes_sent_out = 4 * 2 * n_params * len(split.clients) * cell.rounds
        comm_rounds_out = cell.rounds
    else:
        raise ValueError(f"unrecognized arm {cell.arm!r} for cell {cell.path!r}")

    result = {
        # --- what was run (identity) --------------------------------------
        "dataset": cell.dataset,
        "arm": cell.arm,
        "gamma": cell.gamma,
        "seed": cell.seed,
        "grad_refresh_every": cell.grad_refresh_every,
        # --- the full configuration it was run under ----------------------
        # Self-describing on purpose: a result JSON must be interpretable
        # without its filename and without knowing which CLI invocation
        # produced it. Omitting these is what made T4 and T1 results
        # indistinguishable before docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md.
        "config": {
            "eta_prime": cell.eta_prime,
            "n_rare": cell.n_rare,
            "p_drop": cell.p_drop,
            "dp_epsilon": cell.dp_epsilon,
            "secagg_threshold": cell.secagg_threshold,
            "weight_epochs": cfg.weight_epochs,
            "local_epochs": cfg.local_epochs,
            "eta": cfg.eta,
            "batch_size": cfg.batch_size,
            "val_fraction": cfg.val_fraction,
            "n_rounds": cell.rounds,
            "eval_every": cell.eval_every,
            "attacker_fraction": cell.attacker_fraction,
        },
        "provenance": provenance(),
        # --- results ------------------------------------------------------
        "rounds": history.rounds,
        "all_acc": history.all_acc,
        "last_n_acc": history.last_n_acc,
        "zeta": [v if v == v else None for v in history.zeta],  # NaN -> null
        "final_all_acc": history.final_all_acc,
        "final_last_n_acc": history.final_last_n_acc,
        # Billed for every arm -- the frontier's x-axis needs a measured (or,
        # for T3/T5, analytically-computed per the disclosure above) value at
        # every point, not only at the v2 points.
        "bytes_sent": bytes_sent_out,
        "comm_rounds": comm_rounds_out,
        "failed_secagg_rounds": failed_secagg_rounds,
        # T7 only: v2_raresig's per-round mean attacker aggregation weight
        # (§2.6 step 8's gamma/poisoning-tradeoff paragraph needs this), or
        # the defense cross-product's per-round attack_success_rate_history
        # (1 - detection recall). Empty for every non-T7 arm.
        "attacker_weight_history": attacker_weight_history,
    }
    with open(cell.path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


# --------------------------------------------------------------------------- #
# CLI -- local sequential/parallel execution (the pre-Modal fallback)
# --------------------------------------------------------------------------- #
def execute(cells: Sequence[CellSpec], workers: int = 1) -> int:
    if not cells:
        return 0
    failures = 0
    if workers <= 1:
        for cell in cells:
            print(f">>> {cell.dataset} {cell.arm} g={cell.gamma} seed={cell.seed}")
            try:
                run_one_cell(cell)
            except Exception as exc:  # noqa: BLE001 -- record and continue
                failures += 1
                print(f"  [FAILED] {exc}")
        return failures

    # Force 'spawn' before constructing the pool -- Linux/Modal's default
    # 'fork' deadlocks once TensorFlow has been imported in the parent
    # (COMPUTE_INFRASTRUCTURE_2026-07-27.md §4.3, the one portability finding
    # from the Modal smoke test).
    ctx = multiprocessing.get_context("spawn")
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {pool.submit(run_one_cell, cell): cell for cell in cells}
        for future in as_completed(futures):
            cell = futures[future]
            try:
                future.result()
                print(f"  [OK] {cell.dataset} {cell.arm} g={cell.gamma} seed={cell.seed}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  [FAILED] {cell.dataset} {cell.arm} g={cell.gamma} "
                      f"seed={cell.seed}: {exc}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", nargs="+", default=["T1"],
                    choices=["T1", "T2", "T3", "T4", "T5", "T6"])
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--n-rare", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default=".cache")
    ap.add_argument("--p-drop", type=float, default=0.0)
    ap.add_argument("--dp-epsilon", type=float, default=None)
    ap.add_argument(
        "--eta-prime", type=float, default=DEFAULT_ETA_PRIME,
        help="Aggregation-weight learning rate eta' for BOTH arms. Default "
             f"{DEFAULT_ETA_PRIME} is the Stage B combined-fix value that "
             "FEDGRAA_V2_MASTER_PLAN.md F1 specifies as v2's comparison "
             "reference -- NOT FedConfig's own 0.1 default, which with "
             "projection is Stage B's projection-alone arm (0.0%% rare-class "
             "accuracy on CIC at gamma <= 0.5).",
    )
    ap.add_argument(
        "--secagg-threshold", type=int, default=None,
        help="Shamir reconstruction threshold. Default None = majority "
             "(floor(K/2)+1), the security-correct choice. Rounds with fewer "
             "survivors are aborted and counted (failed_secagg_rounds), not "
             "fatal.",
    )
    ap.add_argument(
        "--t4-m", type=int, default=T4_DEFAULT_M,
        help="Which v2 refresh interval T4's robustness sweep runs on. Set "
             "this to whichever m T1 identifies as best before running T4.",
    )
    ap.add_argument(
        "--nf-uq-max-per-class", type=int,
        default=NF_UQ_NIDS_V2_DEFAULT_MAX_PER_CLASS,
        help="T6 only: per-(silo,class) row cap for the NF-UQ-NIDS-v2 loader "
             "(B6). Bounds memory via deterministic subsampling instead of "
             "an ~88 GB uncapped load of the real ~76M-row release.",
    )
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_cells = enumerate_cells(
        args.tier, datasets=args.datasets, seeds=args.seeds, rounds=args.rounds,
        n_rare=args.n_rare, eval_every=args.eval_every, data_dir=args.data_dir,
        cache_dir=args.cache_dir, p_drop=args.p_drop, dp_epsilon=args.dp_epsilon,
        eta_prime=args.eta_prime, secagg_threshold=args.secagg_threshold,
        t4_m=args.t4_m, nf_uq_max_per_class=args.nf_uq_max_per_class,
        out_dir=args.out_dir,
    )
    to_run = missing_cells(all_cells)
    already_done = len(all_cells) - len(to_run)

    print("=" * 78)
    print(f"  FedGRAA-v2 frontier sweep -- tiers={args.tier}")
    print(f"  datasets={args.datasets} arms={ARM_NAMES} seeds={args.seeds}")
    print(f"  eta'={args.eta_prime}  secagg_threshold="
          f"{args.secagg_threshold or 'majority'}  rounds={args.rounds}")
    print(f"  {len(all_cells)} total cells | {already_done} already complete "
          f"| {len(to_run)} to run")
    print("=" * 78)

    if args.dry_run:
        for cell in to_run:
            print(f"  WOULD RUN: {cell.dataset} {cell.arm} g={cell.gamma} "
                  f"seed={cell.seed} -> {cell.path}")
        return 0
    if not to_run:
        print("Nothing to do -- every cell already has a complete result JSON.")
        return 0

    failures = execute(to_run, workers=args.workers)
    print("=" * 78)
    if failures:
        print(f"  {failures}/{len(to_run)} cells FAILED. Re-run to retry only those.")
        return 1
    print(f"  All {len(to_run)} cells completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
