"""Tests for the v2 frontier campaign driver's enumeration/resumability logic
(scripts/run_v2_frontier_sweep.py) -- structure only, no training runs (the
actual campaign is deferred to Modal; see
docs/research_c/REMAINING_COMPUTE_WORK.md)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_v2_frontier_sweep import (
    ARM_NAMES,
    CellSpec,
    enumerate_cells,
    missing_cells,
)


def test_t1_enumerates_gamma_1_only():
    cells = enumerate_cells(["T1"], datasets=["CIC-IDS-2017"], seeds=[0, 1])
    gammas = {c.gamma for c in cells}
    assert gammas == {1.0}
    assert len(cells) == len(ARM_NAMES) * 2  # 2 seeds


def test_t2_alone_enumerates_extra_gammas_only():
    cells = enumerate_cells(["T2"], datasets=["CIC-IDS-2017"], seeds=[0])
    gammas = {c.gamma for c in cells}
    assert gammas == {0.0, 0.5}


def test_t1_plus_t2_enumerates_all_three_gammas_without_duplication():
    cells = enumerate_cells(["T1", "T2"], datasets=["CIC-IDS-2017"], seeds=[0])
    gammas = sorted({c.gamma for c in cells})
    assert gammas == [0.0, 0.5, 1.0]
    assert len(cells) == len(ARM_NAMES) * 3  # 1 seed x 3 gammas


def test_arm_names_cover_v1_projected_and_all_four_m_values():
    assert ARM_NAMES == ["v1_projected", "v2_m1", "v2_m2", "v2_m3", "v2_m5"]


def test_grad_refresh_every_property():
    v1 = CellSpec(dataset="CIC-IDS-2017", arm="v1_projected", gamma=1.0, seed=0)
    assert v1.grad_refresh_every is None
    v2 = CellSpec(dataset="CIC-IDS-2017", arm="v2_m3", gamma=1.0, seed=0)
    assert v2.grad_refresh_every == 3


def test_missing_cells_skips_complete_results(tmp_path):
    out_dir = str(tmp_path)
    cell = CellSpec(dataset="CIC-IDS-2017", arm="v2_m1", gamma=1.0, seed=0, out_dir=out_dir)
    cells = [cell]
    assert missing_cells(cells) == cells  # nothing on disk yet

    with open(cell.path, "w", encoding="utf-8") as f:
        json.dump({"final_all_acc": 0.5, "final_last_n_acc": 0.5}, f)
    assert missing_cells(cells) == []


def test_missing_cells_reruns_incomplete_json(tmp_path):
    out_dir = str(tmp_path)
    cell = CellSpec(dataset="CIC-IDS-2017", arm="v2_m1", gamma=1.0, seed=0, out_dir=out_dir)
    with open(cell.path, "w", encoding="utf-8") as f:
        json.dump({"final_all_acc": 0.5}, f)  # missing final_last_n_acc
    assert missing_cells([cell]) == [cell]


def test_cell_path_is_stable_and_unique_per_cell():
    a = CellSpec(dataset="CIC-IDS-2017", arm="v2_m1", gamma=1.0, seed=0)
    b = CellSpec(dataset="CIC-IDS-2017", arm="v2_m1", gamma=1.0, seed=1)
    c = CellSpec(dataset="CIC-IDS-2017", arm="v2_m2", gamma=1.0, seed=0)
    assert a.path != b.path != c.path != a.path


# --------------------------------------------------------------------------- #
# T6 -- NF-UQ-NIDS-v2 wired into the campaign driver
# (docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md B6, "completely implement B6")
# --------------------------------------------------------------------------- #
from scripts.run_v2_frontier_sweep import (  # noqa: E402
    NF_UQ_NIDS_V2_DATASET,
    NF_UQ_NIDS_V2_DEFAULT_MAX_PER_CLASS,
)


def test_t6_enumerates_all_arms_at_gamma_1_on_the_nf_uq_dataset_only():
    """T6 is 'T1 repeated on the cross-silo benchmark': all five arms,
    gamma=1.0, over seeds -- but always on NF_UQ_NIDS_V2_DATASET, never
    mixed into the caller-supplied ``datasets`` list (that list only ever
    means CIC-IDS-2017/UNSW-NB15's synthetic-partition path)."""
    cells = enumerate_cells(
        ["T6"], datasets=["CIC-IDS-2017", "UNSW-NB15"], seeds=[0, 1],
    )
    assert len(cells) == len(ARM_NAMES) * 2  # 5 arms x 2 seeds
    assert {c.dataset for c in cells} == {NF_UQ_NIDS_V2_DATASET}
    assert {c.gamma for c in cells} == {1.0}
    assert {c.arm for c in cells} == set(ARM_NAMES)


def test_t6_cells_carry_the_max_per_class_cap():
    cells = enumerate_cells(["T6"], seeds=[0], nf_uq_max_per_class=500)
    assert all(c.nf_uq_max_per_class == 500 for c in cells)


def test_t6_default_max_per_class_matches_frequent_cap_convention():
    cells = enumerate_cells(["T6"], seeds=[0])
    assert all(
        c.nf_uq_max_per_class == NF_UQ_NIDS_V2_DEFAULT_MAX_PER_CLASS
        for c in cells
    )
    assert NF_UQ_NIDS_V2_DEFAULT_MAX_PER_CLASS == 10_000


def test_t6_paths_never_collide_with_t1_t2_t4():
    others = {
        c.path for c in enumerate_cells(["T1", "T2", "T4"])
    }
    t6 = {c.path for c in enumerate_cells(["T6"])}
    assert not (others & t6)
    assert len(t6) == len(enumerate_cells(["T6"]))  # all unique among themselves


def test_t1_t2_t4_t6_together_enumerate_without_any_collision():
    cells = enumerate_cells(["T1", "T2", "T4", "T6"])
    assert len({c.path for c in cells}) == len(cells)


# --------------------------------------------------------------------------- #
# --arm filter (STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md §6.1): an optional
# arms sequence on enumerate_cells, applied uniformly across tiers; empty =
# all arms (zero behavior change for existing callers).
# --------------------------------------------------------------------------- #
def test_arms_filter_empty_is_all_arms():
    all_cells = enumerate_cells(["T1"], datasets=["CIC-IDS-2017"], seeds=[0, 1])
    filtered = enumerate_cells(
        ["T1"], datasets=["CIC-IDS-2017"], seeds=[0, 1], arms=(),
    )
    assert filtered == all_cells


def test_arms_filter_restricts_arms_uniformly_across_tiers():
    cells = enumerate_cells(["T5"], datasets=["CIC-IDS-2017"], seeds=[0, 1],
                            arms=["fedpgo"])
    assert len(cells) == 2  # 2 seeds
    assert {c.arm for c in cells} == {"fedpgo"}
    assert {c.dataset for c in cells} == {"CIC-IDS-2017"}


def test_arms_filter_applies_to_fedgraa_arms_too():
    cells = enumerate_cells(["T1", "T2"], datasets=["CIC-IDS-2017"], seeds=[0],
                            arms=["v2_m1"])
    assert len(cells) == 3  # 3 gammas x 1 seed, v2_m1 only
    assert {c.arm for c in cells} == {"v2_m1"}
    assert sorted({c.gamma for c in cells}) == [0.0, 0.5, 1.0]


def test_arms_filter_unknown_arm_yields_no_cells():
    cells = enumerate_cells(["T5"], datasets=["CIC-IDS-2017"], seeds=[0, 1],
                            arms=["does_not_exist"])
    assert cells == []


def test_arms_filter_preserves_stable_order_and_uniqueness():
    cells = enumerate_cells(["T1", "T5"], datasets=["CIC-IDS-2017"], seeds=[0],
                            arms=["fedpgo", "v2_m1"])
    assert len({c.path for c in cells}) == len(cells)
    assert [c.arm for c in cells] == ["v2_m1", "fedpgo"]
