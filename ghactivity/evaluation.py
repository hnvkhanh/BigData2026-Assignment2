"""The official scorer (assignment2/scripts/evaluate.py), thresholds and the submission format."""
import sys

import numpy as np
import pandas as pd

from .config import EVAL_DIR

sys.path.insert(0, str(EVAL_DIR))
from evaluate import EVENT_COLS, OUTPUT_COLUMNS, scores  # noqa: E402,F401


def best_threshold(y, p):
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    f1 = 2 * tp / (k + ys.sum())
    i = int(np.argmax(f1))
    return float(ps[i]), float(f1[i])


def to_output(gids, pred):
    out = pd.DataFrame({"UserID": gids})
    for k in EVENT_COLS:
        out[k] = pred[k].astype(int)
    return out[OUTPUT_COLUMNS]
