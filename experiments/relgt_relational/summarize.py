"""Average backtest metrics over seeds for <architecture>_<variant>_s<seed> run folders.

Usage: python -m experiments.relgt_relational.summarize [OUTPUTS_DIR] [SUMMARY_CSV]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

from experiments.relgt.data import LABELS

OUTPUTS = Path(__file__).resolve().parent / "outputs"
PATTERN = re.compile(r"^(?P<prefix>sel_)?(?P<arch>relgt|gnn)_(?P<groups>\w+?)_s(?P<seed>\d+)$")


def main() -> None:
    outputs = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUTS
    rows = []
    for path in sorted(outputs.glob("*/backtest_metrics.json")):
        match = PATTERN.match(path.parent.name)
        if not match:
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "epochs": "best calib" if metrics.get("select_epoch") else "last",
                "architecture": match["arch"],
                "groups": match["groups"],
                "seed": int(match["seed"]),
                "score": metrics["score"],
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                **{label: metrics["per_action_f1"][label] for label in LABELS},
            }
        )
    frame = pd.DataFrame(rows)
    keys = ["epochs", "architecture", "groups"]
    summary = frame.groupby(keys).agg(
        seeds=("seed", "count"), **{c: (c, "mean") for c in ["score", "macro_f1", "micro_f1", *LABELS]}
    )
    pd.set_option("display.width", 220)
    print(frame.sort_values([*keys, "seed"]).round(4).to_string(index=False))
    print("\nMean over seeds:")
    print(summary.round(4).to_string())
    if len(sys.argv) > 2:
        summary.round(4).to_csv(sys.argv[2])


if __name__ == "__main__":
    main()
