"""Unit tests for the doc-link checker (AUDIT_REMEDIATION_PLAN_2026-07-25.md R10).

Negative-controlled per the Stage B correctness audit's own lesson (A5): an
audit tool that has never been made to fail is not trustworthy, because the
audit's *own* bug was in its failure path, which by construction never runs
while everything passes. These tests build tiny scratch trees under
``tmp_path`` and assert the checker both (a) stays quiet on a fully-resolving
tree and (b) reports every kind of broken link it is meant to catch.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.check_doc_links import find_broken_links  # noqa: E402


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def test_clean_tree_reports_nothing(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "a.md"), "See [b](b.md) and the [image](img/pic.png).")
    _write(os.path.join(root, "b.md"), "Back to [a](a.md).")
    _write(os.path.join(root, "img", "pic.png"), "not a real png, just needs to exist")

    assert find_broken_links(root) == []


def test_catches_a_broken_sibling_link(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "a.md"), "See [missing](does_not_exist.md).")

    broken = find_broken_links(root)
    assert len(broken) == 1
    assert broken[0] == ("a.md", "does_not_exist.md")


def test_catches_a_broken_relative_parent_link(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "sub", "a.md"), "See [gone](../also_missing.md).")

    broken = find_broken_links(root)
    assert broken == [("sub/a.md", "../also_missing.md")]


def test_ignores_absolute_urls_mailto_and_bare_anchors(tmp_path):
    root = str(tmp_path)
    _write(
        os.path.join(root, "a.md"),
        "[web](https://example.com/x) [mail](mailto:x@example.com) [anchor](#section) [https b](http://example.com).",
    )

    assert find_broken_links(root) == []


def test_resolves_percent_encoded_targets(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "a.md"), "See [pdf](file%20name.pdf).")
    _write(os.path.join(root, "file name.pdf"), "stand-in for a binary file")

    assert find_broken_links(root) == []


def test_strips_heading_anchor_before_resolving(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "a.md"), "See [b section](b.md#some-heading).")
    _write(os.path.join(root, "b.md"), "# Some heading")

    assert find_broken_links(root) == []


def test_catches_broken_link_with_dangling_anchor(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "a.md"), "See [gone](missing.md#some-heading).")

    broken = find_broken_links(root)
    assert broken == [("a.md", "missing.md#some-heading")]


def test_skips_ignored_directories(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, ".venv313", "lib", "a.md"), "See [gone](missing.md).")
    _write(os.path.join(root, "__pycache__", "a.md"), "See [gone](missing.md).")

    assert find_broken_links(root) == []


def test_multiple_broken_links_all_reported(tmp_path):
    root = str(tmp_path)
    _write(
        os.path.join(root, "a.md"),
        "First [x](missing1.md), then [y](missing2.md), then [ok](b.md).",
    )
    _write(os.path.join(root, "b.md"), "fine")

    broken = find_broken_links(root)
    assert sorted(broken) == sorted(
        [("a.md", "missing1.md"), ("a.md", "missing2.md")]
    )
