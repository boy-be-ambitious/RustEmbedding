"""
stats.py — Count files and lines under a directory, grouped by extension.

Usage:
    python stats.py [directory]

If no directory is given, defaults to hmosworld-master/ relative to this script.
"""

import os
import sys
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

# Only these extensions are counted for lines and shown in the per-type table.
# Files with any other extension are counted toward Total Files but their lines
# are NOT added to Total Lines (avoids binary files inflating the count).
TRACKED_EXTENSIONS = {".ets", ".ts", ".js", ".json", ".c", ".cpp", ".h", ".hpp"}

# Directories to skip entirely.
SKIP_DIRS = {".git", "node_modules", ".hvigor", "oh_modules"}


# ── Core ──────────────────────────────────────────────────────────────────────

def count(root: str) -> tuple:
    """
    Walk *root* and return:
        total_files  int
        total_lines  int   (only tracked extensions)
        by_ext       dict[ext -> [file_count, line_count]]
    """
    total_files = 0
    total_lines = 0
    by_ext = defaultdict(lambda: [0, 0])

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place so os.walk won't descend into them.
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            _, ext = os.path.splitext(fname)
            ext = ext.lower()

            total_files += 1
            by_ext[ext][0] += 1

            # Count lines only for tracked (text) extensions.
            if ext in TRACKED_EXTENSIONS:
                lines = _count_lines(fpath)
                total_lines += lines
                by_ext[ext][1] += lines

    return total_files, total_lines, dict(by_ext)


def _count_lines(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


# ── Output ────────────────────────────────────────────────────────────────────

def report(root: str) -> None:
    total_files, total_lines, by_ext = count(root)

    print()
    print("===== Code Statistics =====")
    print(f"Directory  : {os.path.abspath(root)}")
    print(f"Total Files: {total_files}")
    print(f"Total Lines: {total_lines}")
    print()
    print("---- By File Type ----")

    # Show only tracked extensions that actually appear, sorted alphabetically.
    visible = {ext: v for ext, v in by_ext.items() if ext in TRACKED_EXTENSIONS}
    if not visible:
        print("  (no tracked extensions found)")
    else:
        max_ext = max(len(e) for e in visible)
        for ext in sorted(visible):
            fc, lc = visible[ext]
            print(f"  {ext:<{max_ext}}  files: {fc:>6}  lines: {lc:>10}")

    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hmosworld-master"
    )
    if not os.path.isdir(target):
        print(f"Error: '{target}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    report(target)
