"""Download GH Archive hour files, optionally keeping only what preprocess.py would read.

    python scripts/download_gharchive.py --out data/gharchive --cohort-only

The whole archive since 2025-12-01 (the default start) is over a TB. --cohort-only runs preprocess.py's own
extract_hour on every file as it arrives and stores just the kept lines under the same file
name, so preprocess.py produces the same tables from these files at a tiny fraction of the
disk space. Hours GH Archive does not have (HTTP 404) are listed in <out>/missing_hours.txt.
Finished hours are skipped, so the command can be re-run to resume or to add newer hours.
"""
import argparse
import gzip
import importlib.util
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
URL = "https://data.gharchive.org/{}.json.gz"
CHUNK = 1 << 20
RETRIES = 40  # with the 30 s backoff cap below, rides out an ~18 min network outage


def load_preprocess():
    spec = importlib.util.spec_from_file_location("preprocess", ROOT / "assignment2" / "scripts" / "preprocess.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PP = None


def init_worker(cohort_only):
    global PP
    if cohort_only:
        PP = load_preprocess()
        PP._init_worker(PP.load_cohort(ROOT / "assignment2" / "scripts" / "cohort.txt"))


def fetch(name, path):
    """Download one file to path, checking its size. Returns False if GH Archive has no such hour."""
    req = urllib.request.Request(URL.format(name), headers={"User-Agent": "course-assignment-downloader"})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                want, got = int(r.headers.get("Content-Length", -1)), 0
                while chunk := r.read(CHUNK):
                    f.write(chunk)
                    got += len(chunk)
            if want in (-1, got):
                return True
            raise OSError(f"truncated download: {got} of {want} bytes")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            err = e
        except OSError as e:
            err = e
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"{name}: {err}")


def process(name, out_dir, cohort_only):
    """Returns (name, stored bytes, source bytes), stored bytes None if the hour does not exist."""
    final = out_dir / f"{name}.json.gz"
    tmp = out_dir / f"{name}.part{os.getpid()}"
    try:
        if not fetch(name, tmp):
            return name, None, 0
        source = tmp.stat().st_size
        if cohort_only:
            lines = PP.extract_hour(tmp)
            filtered = out_dir / f"{name}.kept{os.getpid()}"
            with gzip.open(filtered, "wb") as f:
                for line in lines:
                    f.write(line + b"\n")
            os.replace(filtered, final)
        else:
            os.replace(tmp, final)
        return name, final.stat().st_size, source
    finally:
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", default="2025-12-01", help="first hour, UTC")
    ap.add_argument("--end", default=None, help="stop before this hour, UTC (default: now)")
    ap.add_argument("--cohort-only", action="store_true", help="store only the lines preprocess.py keeps")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.cohort_only:
        # Load preprocess.py here first: if it cannot be imported (e.g. numpy/pandas missing),
        # every worker would die in its initializer and each hour would fail with an opaque
        # "process pool terminated abruptly" error instead of this one clear message.
        try:
            load_preprocess()
        except ImportError as e:
            sys.exit(f"Cannot load assignment2/scripts/preprocess.py: {e}\n"
                     f"Install the requirements first: {sys.executable} -m pip install -r requirements.txt")

    # Temp files left by a killed earlier run (the per-hour cleanup never ran for them).
    for stale in [*args.out.glob("*.part*"), *args.out.glob("*.kept*")]:
        stale.unlink(missing_ok=True)

    h = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = (datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end
           else datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0))
    names = []
    while h < end:
        names.append(f"{h:%Y-%m-%d}-{h.hour}")
        h += timedelta(hours=1)
    todo = [n for n in names if not (args.out / f"{n}.json.gz").exists()]
    print(f"{len(names)} hours, {len(todo)} to fetch", flush=True)

    missing, failed, stored, source, t0 = [], [], 0, 0, time.time()
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(args.cohort_only,)) as pool, \
            tqdm(total=len(todo), unit="hour", dynamic_ncols=True, smoothing=0.05) as bar:
        jobs = {pool.submit(process, n, args.out, args.cohort_only): n for n in todo}
        for job in as_completed(jobs):
            bar.update()
            try:
                name, size, src = job.result()
            except Exception as e:  # keep going; a re-run retries this hour
                failed.append(jobs[job])
                tqdm.write(f"FAILED {jobs[job]}: {e}")
                continue
            source += src
            if size is None:
                missing.append(name)
            else:
                stored += size
            bar.set_postfix_str(f"read {source / 1e9:.1f} GB, stored {stored / 1e6:.0f} MB, "
                                f"{len(missing)} missing, {len(failed)} failed", refresh=False)
    (args.out / "missing_hours.txt").write_text(
        "".join(n + "\n" for n in names if not (args.out / f"{n}.json.gz").exists()))
    print(f"done in {timedelta(seconds=round(time.time() - t0))}: {len(missing)} hours missing on "
          f"GH Archive, {len(failed)} failed (re-run to retry)")
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
