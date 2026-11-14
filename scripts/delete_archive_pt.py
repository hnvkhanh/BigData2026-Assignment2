"""Delete all .pt files and cache directories under results/archive_old_data.

Targets: *.pt files, and directories named "cache" or "__pycache__".

Usage:
    python scripts/delete_archive_pt.py            # dry run (lists targets)
    python scripts/delete_archive_pt.py --yes      # actually delete
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results" / "archive_old_data"
CACHE_NAMES = {"cache", "__pycache__"}


def size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> None:
    cache_dirs = sorted(d for d in ROOT.rglob("*") if d.is_dir() and d.name in CACHE_NAMES)
    # drop nested cache dirs already covered by a parent cache dir
    cache_dirs = [d for d in cache_dirs if not any(p in cache_dirs for p in d.parents)]
    pt_files = sorted(
        f for f in ROOT.rglob("*.pt")
        if not any(p in cache_dirs for p in f.parents)
    )
    targets = cache_dirs + pt_files

    for t in targets:
        print(f"{size(t) / 1e6:10.1f} MB  {t.relative_to(ROOT)}")
    print(f"\n{len(cache_dirs)} cache dir(s), {len(pt_files)} .pt file(s), "
          f"{sum(size(t) for t in targets) / 1e6:.1f} MB total")

    if "--yes" not in sys.argv:
        print("Dry run. Re-run with --yes to delete.")
        return
    for t in targets:
        shutil.rmtree(t) if t.is_dir() else t.unlink()
    print("Deleted.")


if __name__ == "__main__":
    main()
