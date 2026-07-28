"""Regression coverage for ``scripts/modal_frontier_campaign.py``'s
``_ignore`` image-exclusion function.

Context (``docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md``, "second new defect"
box): a first version of this driver put T6's image -- which includes the
12.79 GB ``data/NF-UQ-NIDS-v2`` merged CSV -- on the SAME ``modal.App`` as
T1/T2/T4, and a plain ``--tier T1`` launch built and uploaded both images,
turning a ~2-minute launch into 15+ minutes. The fix moved T6 to its own
file/App and made this App's image exclude ``data/NF-UQ-NIDS-v2``
unconditionally via ``_ignore``. That fix was verified once, manually, by
timing a real Modal launch -- it had no unit test, so a future edit to
``EXCLUDE_PARTS`` or the string check inside ``_ignore`` could silently
reintroduce the same bug with no CI signal. These tests are that signal.

``modal`` is not a dependency of this project's own test venv (the Modal
CLI/SDK is only needed to actually launch a campaign, from a separate
environment) -- ``importorskip`` lets this file skip cleanly here while
still running wherever ``modal`` is installed alongside the suite.
"""

from __future__ import annotations

import pytest

modal = pytest.importorskip("modal")

from scripts.modal_frontier_campaign import _ignore  # noqa: E402


@pytest.mark.parametrize(
    "path",
    [
        "data/NF-UQ-NIDS-v2",
        "data\\NF-UQ-NIDS-v2",
        "data/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv",
        "data\\NF-UQ-NIDS-v2\\NF-UQ-NIDS-v2.csv",
        "/repo/data/NF-UQ-NIDS-v2",
        "/repo/data/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv",
    ],
)
def test_ignore_excludes_nf_uq_nids_v2_directory_and_its_contents(path):
    assert _ignore(path) is True, (
        f"{path!r} must be excluded -- this App never runs a T6 cell, so "
        "the 12.79 GB NF-UQ-NIDS-v2 file must never be uploaded with it "
        "(the bug this driver's docstring documents fixing)"
    )


@pytest.mark.parametrize(
    "path",
    [
        "data/CIC-IDS-2017",
        "data/CIC-IDS-2017/some_file.csv",
        "data/UNSW-NB15/some_file.csv",
        "fedgraa/data.py",
        "scripts/modal_frontier_campaign_t6.py",
        # A same-prefix sibling must NOT be treated as NF-UQ-NIDS-v2 itself.
        "data/NF-UQ-NIDS-v2-notes.txt",
    ],
)
def test_ignore_keeps_everything_t1_t2_t4_actually_need(path):
    assert _ignore(path) is False, (
        f"{path!r} must be included -- T1/T2/T4 need data/CIC-IDS-2017 and "
        "data/UNSW-NB15, and over-excluding here would break the campaign "
        "the opposite way (missing data instead of an oversized image)"
    )


@pytest.mark.parametrize("part", ["__pycache__", "results", ".git"])
def test_ignore_excludes_the_other_declared_parts(part):
    assert _ignore(f"{part}/whatever") is True
