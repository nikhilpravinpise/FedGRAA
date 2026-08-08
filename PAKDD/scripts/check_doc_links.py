"""Documentation link-integrity check (AUDIT_REMEDIATION_PLAN_2026-07-25.md R6/R10).

Walks every ``.md`` file in the repository and verifies that every relative
markdown link/image target resolves to a real file on disk. Neither Stage
completeness gate checked this before -- both only validate ``results/...``
paths -- which is how 34 links silently broke across a doc reorganization
without either gate noticing (F6 in the verification audit).

Also checks raw HTML ``<img src="...">`` tags, not just markdown
``![]()``/``[]()`` syntax: READMEs that center or size-constrain an image
via ``<p align="center"><img src="..." width="...%"></p>`` (used for the
Stage A/B result figures embedded in the root and ``PAKDD/README.md``) would
otherwise have their image paths silently unchecked by this script -- the
same class of blind spot R6/R10 exists to close, just for HTML instead of
markdown syntax.

Skips absolute URLs (``http(s):``, ``mailto:``), bare anchors (``#...``),
and the fragment portion of a link (``file.md#section`` is resolved against
``file.md``; the anchor itself is not verified against the target's headers).

Exposes :func:`find_broken_links` so both completeness gates (and tests) can
call it as a library function, plus a standalone CLI::

    python scripts/check_doc_links.py
"""

from __future__ import annotations

import os
import re
import sys
import urllib.parse

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_IMG_SRC_RE = re.compile(r'<img\s[^>]*\bsrc=["\']([^"\']+)["\']', re.IGNORECASE)
SKIP_ROOT_DIRS = {".venv313", "__pycache__", ".git", "node_modules", ".pytest_cache"}


def find_broken_links(root_dir: str) -> list[tuple[str, str]]:
    """Return a list of ``(markdown_file_relpath, broken_target)`` pairs.

    ``root_dir`` is walked recursively for ``*.md`` files, skipping
    :data:`SKIP_ROOT_DIRS`. Each relative link/image target is resolved
    against its containing file's directory; targets that do not exist on
    disk are reported. An empty return value means every relative link in
    the tree resolves.
    """
    root_dir = os.path.abspath(root_dir)
    broken: list[tuple[str, str]] = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_ROOT_DIRS]
        for filename in filenames:
            if not filename.endswith(".md"):
                continue
            md_path = os.path.join(dirpath, filename)
            try:
                text = open(md_path, encoding="utf-8", errors="replace").read()
            except OSError:
                continue

            raw_targets = LINK_RE.findall(text) + HTML_IMG_SRC_RE.findall(text)
            for raw_target in raw_targets:
                target = raw_target.split()[0].strip()  # drop optional "title"
                if not target or re.match(r"^(https?:|mailto:|#)", target):
                    continue
                path_part = target.split("#", 1)[0]
                if not path_part:
                    continue
                decoded = urllib.parse.unquote(path_part)
                resolved = os.path.normpath(os.path.join(dirpath, decoded))
                if not os.path.exists(resolved):
                    rel_md = os.path.relpath(md_path, root_dir).replace(os.sep, "/")
                    broken.append((rel_md, target))

    return broken


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    broken = find_broken_links(repo_root)

    print("=" * 80)
    print("  Documentation link-integrity check")
    print("=" * 80)

    if broken:
        print(f"\n  {len(broken)} broken relative link(s):\n")
        for md_file, target in broken:
            print(f"  [FAIL] {md_file}\n         -> {target}")
        print("\n" + "=" * 80)
        print(f"  DOC LINKS BROKEN — {len(broken)} link(s) do not resolve.")
        print("=" * 80)
        return 1

    print("\n  All relative markdown links resolve.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
