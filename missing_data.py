"""Interactive explorer for GH Archive's missing events (Dash + Plotly).

    python missing_data.py          # then open http://127.0.0.1:8051 (startup ~15 s)

Every hour file exists, but many hours are *thinned*: they hold only part of the events. The tabs
answer one question each:
  Artifact or trend?  is the drop a natural change in behaviour or events missing from the archive?
                one row per test below, with what each explanation predicts and what the data shows
  Connection    level breaks (change points) per activity and the archive size: same days? plus
                correlation of day-to-day changes and cohort volume vs archive size
  Shape         weekday / hour-of-day profiles vs random-thinning noise; ratios of linked activities
  Who loses events  churn (users vanish) vs thinning (every user loses events), by activity tier
  Orphans       reviews of PRs whose opening was never captured
  Overview      where capture drops below a threshold of normal (table) and the raw hour-file size
  Daily volume  events per day as % of normal, raw counts, or share of hours with zero events
  Heatmap       date x hour (UTC): are whole days or particular hours lost?
  Weekly labels positive rate / events per user-week / single-event share for eligible users
  Thinning sim  drop events of chosen "normal" weeks at random: does it reproduce chosen recent weeks?
  Persistence   P(active next week | active / inactive this week)
  Test users    share of the 4,465 test users with activity in a window, before vs at the end of the data
  Raw archive   event types in the raw hour files (all actors), parsed on demand
The sidebar (activities, date range, reference period = "normal capture", comparison period, threshold,
change-point settings) drives all tabs.
Activity definitions, cohort and the test-user selection rule come from experiments/baseline/run.py.
"""
import argparse
import gzip
import importlib.util
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("baseline_run", ROOT / "experiments" / "baseline" / "run.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

ACTS = list(B.EVENT_COLS)
NONPUSH = [k for k in ACTS if k != "push"]
WEEK, DAY = B.WEEK, B.DAY
ARCHIVE = ROOT / "data" / "gharchive"
REF_DEFAULT = ("2026-01-05", "2026-05-01")                 # "normal" capture
SPRING_DEFAULT = ("2026-01-17", "2026-05-02")              # week starts used as the unthinned source
RECENT_DEFAULT = ("2026-10-03", "2026-10-31")              # week starts used as the thinned target
CMP_DEFAULT = ("2026-10-17", "2026-11-14")                 # period suspected of missing data
COLORS = dict(zip(ACTS, ["#2a6fdb", "#e07b00", "#1f9e74", "#c2357a", "#7a52c7", "#6b6b6b"]))
ARCHIVE_TYPES = {"PullRequestEvent": "open_pr", "IssuesEvent": "open_issue",
                 "IssueCommentEvent": "issue_comment", "PullRequestReviewEvent": "pr_review",
                 "PullRequestReviewCommentEvent": "pr_review_comment", "PushEvent": "push"}
LAYOUT = dict(template="plotly_white", margin=dict(l=60, r=20, t=50, b=40), hovermode="x unified")


# ---------------------------------------------------------------- data (computed once at startup)

class Data:
    def __init__(self):
        print("loading tables ...")
        tables = B.read_tables()
        self.cohort, self.ev, self.own, data_end = B.load(tables)
        self.n = len(self.cohort)
        self.last_day = max(self.ev[k].t.max() for k in ACTS).floor("D")
        days = pd.date_range(B.DATA_START, self.last_day, freq="D")
        hours = pd.date_range(B.DATA_START, self.last_day + DAY - pd.Timedelta(hours=1), freq="h")
        self.daily = pd.DataFrame({k: self.ev[k].t.dt.floor("D").value_counts().reindex(days, fill_value=0)
                                   for k in ACTS}, index=days)
        self.hourly = pd.DataFrame({k: self.ev[k].t.dt.floor("h").value_counts().reindex(hours, fill_value=0)
                                    for k in ACTS}, index=hours)
        self.zero = (self.hourly == 0).groupby(self.hourly.index.floor("D")).mean()

        print("weekly labels ...")
        lags = {k: B.TEST_START - v for k, v in data_end.items()}
        weeks, s = [], B.TEST_START - WEEK
        while s >= B.DATA_START + 120 * DAY:   # eligibility needs ~90 days of history
            if s + WEEK <= self.last_day:
                weeks.append(s)
            s -= WEEK
        self.weeks = sorted(weeks)
        self.elig, self.wcount, rows = {}, {}, []
        for s in self.weeks:
            el = B.build_week(s, self.ev, self.own, lags, data_end, self.n, with_labels=False).eligible.values
            self.elig[s] = el
            r = {"week": s, "eligible": int(el.sum())}
            for k in ACTS:
                c = np.bincount(self.week_uids(k, s), minlength=self.n)
                self.wcount[(s, k)] = c
                ce = c[el]
                r[f"rate_{k}"] = (ce > 0).mean()
                r[f"events_{k}"] = ce.mean()
                r[f"single_{k}"] = (ce == 1).sum() / max((ce > 0).sum(), 1)
            rows.append(r)
        self.wk = pd.DataFrame(rows)

        tin = pd.read_csv(B.TEST_INPUT)
        self.test_cidx = np.flatnonzero(self.cohort.github_user_id.isin(tin.iloc[:, 0]).values)

        rows = []
        for f in os.scandir(ARCHIVE):
            if f.name.endswith(".json.gz"):
                d, h = f.name[:-8].rsplit("-", 1)
                rows.append((pd.Timestamp(d) + pd.Timedelta(hours=int(h)), f.stat().st_size / 1e6))
        self.sizes = pd.DataFrame(rows, columns=["hour", "mb"]).set_index("hour").mb.sort_index()
        mh = ARCHIVE / "missing_hours.txt"
        self.missing_hours = mh.read_text().split() if mh.exists() else []
        self.size_day = self.sizes.groupby(self.sizes.index.floor("D")).median().reindex(days)

        # Orphans: follow-up rows (all actors) whose PR was never seen being opened. A PR row exists
        # without created_at when GH Archive only showed it through a review / comment / reference.
        prs = tables["prs"]
        opened = prs.created_at.notna().to_numpy()   # row position == pr_id
        orph = {}
        for name, key, t in [("reviews", "reviews", "submitted_at"), ("review comments", "rcomments", "created_at")]:
            d = tables[key]
            d = d[d.pr_id.notna() & d[t].notna() & (d[t] >= B.DATA_START)]
            orph[name] = pd.Series(~opened[d.pr_id.astype(np.int64).values], index=d[t].values)
        seen = prs[prs.first_seen_at >= B.DATA_START]
        orph["PR rows (by first seen)"] = pd.Series(seen.created_at.isna().values, index=seen.first_seen_at.values)
        self.orphans = pd.DataFrame({k: v.astype(float).sort_index().resample("D").mean()
                                     for k, v in orph.items()}).reindex(days)

    def week_uids(self, k, s):
        e = self.ev[k]
        m = (e.t.values >= np.datetime64(s)) & (e.t.values < np.datetime64(s + WEEK))
        return e.uid.values[m]


def regimes(series, lo, min_days=4):
    """Runs of days where the 5-day median of `series` (fraction of normal) stays below lo."""
    s = series.rolling(5, center=True, min_periods=1).median()
    out, start = [], None
    for d, v in (s < lo).items():
        if v and start is None:
            start = d
        if not v and start is not None:
            if (d - start).days >= min_days:
                out.append((start, d, series[start:d - DAY].mean()))
            start = None
    if start is not None:
        out.append((start, series.index[-1] + DAY, series[start:].mean()))
    return out


# ---------------------------------------------------------------- artifact-vs-trend tests
#
# A natural decline (people working less, users leaving the cohort) and a collection loss leave
# different fingerprints:
#   * timing   natural trends drift and differ per activity; a collection loss steps abruptly, on the
#              same days for every activity, and moves the size of the whole archive with it;
#   * shape    a natural change alters *when* and *how* people work; random event loss keeps the
#              weekday / hour-of-day shape and only scales the level;
#   * coupling linked activities (review comments per review, PRs per push) keep stable ratios in real
#              behaviour; per-type loss rates break them;
#   * users    churn removes some users entirely and leaves the rest at full rate; thinning removes the
#              same share of every user's events, so even heavy users' counts collapse;
#   * orphans  when events are lost, follow-ups (reviews) point at PRs whose opening was never seen.

PAIRS = [("pr_review_comment", "pr_review"), ("pr_review", "open_pr"), ("issue_comment", "open_issue"),
         ("open_pr", "push"), ("issue_comment", "open_pr")]


def deseason(df, ref):
    """Daily values ÷ the reference median of the same weekday (1 = normal level, weekly cycle removed)."""
    r = df[ts(ref[0]):ts(ref[1])]
    prof = r.groupby(r.index.dayofweek).median().replace(0, np.nan)
    return pd.DataFrame(df.values / prof.reindex(df.index.dayofweek).values, index=df.index, columns=df.columns)


def segments(y, k, min_len=7):
    """Binary segmentation of a piecewise-constant mean: the k breaks that most reduce squared error."""
    segs, breaks = [(0, len(y))], []

    def best(a, b):
        n = b - a
        if n < 2 * min_len:
            return None
        cs = np.cumsum(y[a:b])
        i = np.arange(min_len, n - min_len + 1)
        left, tot = cs[i - 1], cs[-1]
        gain = left ** 2 / i + (tot - left) ** 2 / (n - i) - tot ** 2 / n
        j = int(np.argmax(gain))
        return gain[j], a + int(i[j])

    for _ in range(k):
        cands = [(best(a, b), (a, b)) for a, b in segs]
        cands = [c for c in cands if c[0] is not None]
        if not cands:
            break
        (_, i), seg = max(cands, key=lambda c: c[0][0])
        segs.remove(seg)
        segs += [(seg[0], i), (i, seg[1])]
        breaks.append(i)
    return sorted(breaks)


def coincidence(breaks, n_days, tol, sims=2000):
    """Share of breaks with a break of another series within ±tol days, vs randomly shifted break sets."""
    names = [k for k, v in breaks.items() if len(v)]
    if len(names) < 2:
        return np.nan, np.nan, np.nan

    def frac(b):
        hit = tot = 0
        for k in names:
            other = np.concatenate([b[o] for o in names if o != k])
            for d in b[k]:
                tot += 1
                hit += bool((np.abs(other - d) <= tol).any())
        return hit / tot

    obs = frac({k: np.asarray(breaks[k]) for k in names})
    rng = np.random.default_rng(0)
    null = np.array([frac({k: (np.asarray(breaks[k]) + rng.integers(n_days)) % n_days for k in names})
                     for _ in range(sims)])
    return obs, null.mean(), (null >= obs).mean()


def comovement(acts, rng, ref, k, tol):
    x = pd.concat([D.daily[acts], D.size_day.rename("archive MB")], axis=1)[rng[0]:rng[1]].ffill()
    r = deseason(x, ref)
    lr = np.log(r.clip(lower=0.01).fillna(0.01))
    breaks, steps = {}, {}
    for c in lr.columns:
        b = segments(lr[c].values, k)
        breaks[c] = b
        edges = [0] + b + [len(lr)]
        means = [lr[c].values[a:e].mean() for a, e in zip(edges[:-1], edges[1:])]
        steps[c] = (edges, means)
    obs, null, p = coincidence(breaks, len(lr), tol)
    d = lr.diff()
    corr_rng = d.corr()
    corr_ref = np.log(deseason(x, ref).clip(lower=0.01))[ts(ref[0]):ts(ref[1])].diff().corr()
    return dict(ratio=r, logratio=lr, breaks=breaks, steps=steps, coin=(obs, null, p),
                corr_rng=corr_rng, corr_ref=corr_ref)


def event_times(k, a, b):
    t = D.ev[k].t
    return t[(t >= a) & (t < b)]


def tv(p, q):
    return 0.5 * np.abs(p / p.sum() - q / q.sum()).sum()


def profile_test(k, kind, base, test, draws=500):
    """TV distance between the base and test profile, and its 95th percentile if test were just thinned base."""
    nb = 7 if kind == "weekday" else 24
    get = (lambda t: t.dt.dayofweek) if kind == "weekday" else (lambda t: t.dt.hour)
    pb = np.bincount(get(event_times(k, *base)), minlength=nb).astype(float)
    pt = np.bincount(get(event_times(k, *test)), minlength=nb).astype(float)
    if pb.sum() < 50 or pt.sum() < 20:
        return None
    sims = np.random.default_rng(0).multinomial(int(pt.sum()), pb / pb.sum(), size=draws)
    null = np.array([tv(s, pb) for s in sims])
    lo, hi = np.percentile(sims / pt.sum(), [2.5, 97.5], axis=0)
    return dict(base=pb, test=pt, tv=tv(pt, pb), tv95=np.percentile(null, 95), band=(lo, hi))


def halves(ref):
    a, b = ts(ref[0]), ts(ref[1])
    m = a + (b - a) / 2
    return (a, m), (m, b)


def shape_table(acts, ref, cmp):
    base, test = (ts(ref[0]), ts(ref[1])), (ts(cmp[0]), ts(cmp[1]))
    h1, h2 = halves(ref)
    rows = []
    for k in acts:
        for kind in ("weekday", "hour"):
            t = profile_test(k, kind, base, test)
            c = profile_test(k, kind, h1, h2)
            if t is None:
                continue
            rows.append({"activity": k, "profile": kind, "events ref": int(t["base"].sum()),
                         "events cmp": int(t["test"].sum()), "TV cmp": round(t["tv"], 3),
                         "thinning noise (95%)": round(t["tv95"], 3),
                         "natural drift (ref halves)": round(c["tv"], 3) if c else 0,
                         "allowed": round(t["tv95"] + (c["tv"] if c else 0), 3)})
            rows[-1]["shape"] = "changed" if t["tv"] > rows[-1]["allowed"] else "kept"
    return pd.DataFrame(rows)


def coupling(acts, ref):
    wk = D.daily.resample("7D", origin=B.TEST_START.normalize()).sum()
    out = {}
    for a, b in PAIRS:
        if a in acts and b in acts:
            r = wk[a] / wk[b].replace(0, np.nan)
            out[f"{a} / {b}"] = r / r[ts(ref[0]):ts(ref[1])].mean()
    return pd.DataFrame(out, index=wk.index)


def user_counts(acts, a, b):
    c = np.zeros(D.n)
    for k in acts:
        e = D.ev[k]
        m = (e.t.values >= np.datetime64(a)) & (e.t.values < np.datetime64(b))
        c += np.bincount(e.uid.values[m], minlength=D.n)
    return c


def tiers(acts, base, test, n_bins=5):
    """Users active in `base`, binned by their base rate, and what is left of them in `test`."""
    cb, ct = user_counts(acts, *base), user_counts(acts, *test)
    db, dt = (base[1] - base[0]).days, (test[1] - test[0]).days
    u = cb > 0
    rate, got = cb[u] / db, ct[u]
    p = got.sum() / (rate.sum() * dt)                       # overall share kept
    exp = p * rate * dt
    tier = pd.qcut(pd.Series(rate).rank(method="first"), n_bins, labels=False)
    rows = []
    for q in range(n_bins):
        m = (tier == q).values
        seen = got[m] > 0
        rows.append({"tier": q + 1, "users": int(m.sum()), "events/week in base": round(np.median(rate[m]) * 7, 2),
                     "seen %": seen.mean() * 100, "seen % if uniform": (1 - np.exp(-exp[m])).mean() * 100,
                     "events per seen user, % of base": got[m][seen].sum() / max(rate[m][seen].sum() * dt, 1e-9) * 100})
    return pd.DataFrame(rows), p


@lru_cache(maxsize=32)
def scan_archive(stride, hour):
    rows = []
    for d in pd.date_range(B.DATA_START, D.last_day, freq=f"{stride}D"):
        f = ARCHIVE / f"{d:%Y-%m-%d}-{hour}.json.gz"
        if not f.exists():
            continue
        cnt = dict.fromkeys(list(ARCHIVE_TYPES.values()) + ["other"], 0)
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                cnt[ARCHIVE_TYPES.get(json.loads(line).get("type"), "other")] += 1
        rows.append({"day": d, **cnt})
    return pd.DataFrame(rows).set_index("day")


# ---------------------------------------------------------------- layout

def control(label, comp):
    return html.Div([html.Label(label, style={"fontWeight": 600, "fontSize": 13}), comp],
                    style={"marginBottom": 14})


def picker(id_, start, end):
    return dcc.DatePickerRange(id=id_, start_date=start, end_date=end, min_date_allowed=B.DATA_START.date(),
                               max_date_allowed=D.last_day.date(), display_format="YYYY-MM-DD")


def note(text):
    return html.P(text, style={"color": "#555", "fontSize": 14, "maxWidth": 900})


def build_app():
    app = Dash(__name__, title="Missing data explorer")
    sidebar = html.Div([
        html.H3("Missing data", style={"marginTop": 0}),
        control("Activities", dcc.Checklist(id="acts", options=ACTS, value=ACTS,
                                            labelStyle={"display": "block"})),
        control("Date range", picker("range", B.DATA_START.date(), D.last_day.date())),
        control("Reference period (normal capture)", picker("ref", *REF_DEFAULT)),
        control("Comparison period (suspected loss)", picker("cmp", *CMP_DEFAULT)),
        control("Change points per series", dcc.Slider(id="cp-k", min=1, max=12, step=1, value=6,
                                                       marks={1: "1", 6: "6", 12: "12"})),
        control("Breaks count as shared within ± days", dcc.Slider(id="cp-tol", min=0, max=7, step=1, value=2,
                                                                   marks={0: "0", 2: "2", 7: "7"})),
        control("Low-capture threshold (% of normal)",
                dcc.Slider(id="thr", min=10, max=90, step=5, value=50,
                           marks={v: str(v) for v in (10, 30, 50, 70, 90)})),
        control("Smoothing (days)", dcc.Slider(id="smooth", min=1, max=28, step=1, value=7,
                                               marks={1: "1", 7: "7", 14: "14", 28: "28"})),
        control("Shade low-capture periods of",
                dcc.Dropdown(id="shade", options=["none", "all non-push"] + ACTS, value="all non-push",
                             clearable=False)),
    ], style={"width": 290, "padding": 16, "borderRight": "1px solid #ddd", "flexShrink": 0})

    tabs = dcc.Tabs([
        dcc.Tab(label="Artifact or trend?", children=[
            note("Is the drop a natural change in behaviour, or events missing from the archive? Each row is one "
                 "test, with what each explanation predicts. Uses the sidebar's reference and comparison periods, "
                 "date range and change-point settings; the next four tabs show each test in detail."),
            dcc.Loading(dash_table.DataTable(
                id="vd-table", style_cell={"fontSize": 13, "whiteSpace": "normal", "height": "auto",
                                           "textAlign": "left", "maxWidth": 320},
                style_data_conditional=[
                    {"if": {"filter_query": '{reads as} contains "loss"', "column_id": "reads as"},
                     "backgroundColor": "#fde2e2"},
                    {"if": {"filter_query": '{reads as} contains "natural"', "column_id": "reads as"},
                     "backgroundColor": "#e2f3e5"}])),
        ]),
        dcc.Tab(label="Connection", children=[
            html.Div(id="cp-text", style={"margin": "12px 0"}),
            dcc.Loading(dcc.Graph(id="cp-fig", style={"height": 640})),
            note("Top: each series ÷ its normal level for that weekday (log), 7-day mean, with the fitted step "
                 "levels dashed. Bottom: where each series' level breaks. A collection loss shows as steps on the "
                 "same days in every activity and in the archive's file size; natural trends drift and differ."),
            html.Div([dcc.Graph(id="cp-corr-ref", style={"flex": 1}), dcc.Graph(id="cp-corr-rng", style={"flex": 1})],
                     style={"display": "flex"}),
            note("Correlation of day-to-day changes (weekday cycle removed). In normal weeks activities move "
                 "fairly independently; if shared jumps make them move together across the date range, a common "
                 "cause (collection) is driving them."),
            dcc.Graph(id="cp-archive", style={"height": 460}),
            note("Cohort events per day against the archive's hour-file size (all actors). 9,000 users changing "
                 "their habits cannot shrink the whole archive; if the two move together, the archive is losing events."),
        ]),
        dcc.Tab(label="Shape", children=[
            dcc.Dropdown(id="sh-act", options=ACTS, value="issue_comment", clearable=False,
                         style={"width": 300, "marginTop": 10}),
            dcc.Graph(id="sh-fig", style={"height": 420}),
            note("Share of events by weekday and by hour (UTC): reference vs comparison period. Grey band = 95% "
                 "range if the comparison period were the reference with events dropped at random. Inside the band "
                 "= same shape, only fewer events. Note: daylight saving moves human hours by 1 h in UTC."),
            dash_table.DataTable(id="sh-table", sort_action="native",
                                 style_cell={"fontFamily": "monospace", "fontSize": 12},
                                 style_data_conditional=[{"if": {"filter_query": '{shape} = "changed"',
                                                                 "column_id": "shape"}, "backgroundColor": "#fde2e2"}]),
            note("TV = total variation distance between the two profiles (0 = identical). Control: the same test "
                 "between the two halves of the reference period, i.e. how much shape changes naturally."),
            dcc.Graph(id="sh-coupling", style={"height": 440}),
            note("Weekly ratio between linked activities, ÷ its reference mean (1 = normal). Real behaviour keeps "
                 "these roughly stable (a review comment needs a review, a PR needs a push); jumps by several times "
                 "mean the archive loses some event types more than others."),
        ]),
        dcc.Tab(label="Who loses events", children=[
            dcc.Dropdown(id="us-act", options=["all non-push"] + ACTS, value="all non-push", clearable=False,
                         style={"width": 300, "marginTop": 10}),
            html.Div(id="us-text", style={"margin": "10px 0"}),
            dcc.Graph(id="us-fig", style={"height": 440}),
            dash_table.DataTable(id="us-table", style_cell={"fontFamily": "monospace", "fontSize": 12}),
            note("Users active in the reference period, in 5 tiers by their reference rate. Churn (natural) removes "
                 "users and leaves the rest at their usual rate: fewer users seen than predicted, but events per "
                 "seen user ≈ 100%. Thinning removes events from everyone: users seen ≈ predicted, and even the "
                 "heaviest users drop to the overall keep rate. Control = the same between the reference halves: "
                 "it shows the natural pattern, including regression to the mean (light tiers were picked for having "
                 "had an event, so they look like they grow; heavy tiers look like they shrink)."),
        ]),
        dcc.Tab(label="Orphans", children=[
            html.Div(id="or-text", style={"margin": "10px 0"}),
            dcc.Graph(id="or-fig", style={"height": 460}),
            note("Share of PR reviews / review comments (all actors) whose PR was never seen being opened, and "
                 "share of PR rows known only through references (no created_at). Behaviour changes do not create "
                 "follow-ups to unseen events; lost opening events do."),
        ]),
        dcc.Tab(label="Overview", children=[
            html.Div(id="ov-text", style={"margin": "12px 0"}),
            dash_table.DataTable(id="ov-table", style_cell={"fontFamily": "monospace", "fontSize": 13},
                                 sort_action="native", page_size=30),
            dcc.Graph(id="ov-size"),
            note("A shrinking hour file means GH Archive itself captured fewer events, not that the download failed."),
        ]),
        dcc.Tab(label="Daily volume", children=[
            dcc.RadioItems(id="dv-metric", inline=True, value="pct",
                           options=[{"label": " % of normal ", "value": "pct"},
                                    {"label": " raw count ", "value": "raw"},
                                    {"label": " % of hours with zero events ", "value": "zero"}]),
            dcc.Checklist(id="dv-log", options=[{"label": " log y", "value": "log"}], value=["log"], inline=True),
            dcc.Graph(id="dv-fig", style={"height": 520}),
            note("Far below 100% = missing events; above it (September) = backfilled events. If whole hours were "
                 "lost, the zero-hour share would jump to ~100%; a modest rise while counts fall 5-10x means events "
                 "are dropped inside every hour (thinning)."),
        ]),
        dcc.Tab(label="Heatmap", children=[
            dcc.Dropdown(id="hm-act", options=["all non-push"] + ACTS, value="all non-push", clearable=False,
                         style={"width": 300, "marginTop": 10}),
            dcc.Graph(id="hm-fig", style={"height": 520}),
            note("Each hour of the day is divided by its own median in the reference period. Vertical bands = days "
                 "thinned as a whole; horizontal stripes would mean particular hours are lost."),
        ]),
        dcc.Tab(label="Weekly labels", children=[
            dcc.RadioItems(id="wl-metric", inline=True, value="rate",
                           options=[{"label": " positive rate % ", "value": "rate"},
                                    {"label": " events per eligible user-week ", "value": "events"},
                                    {"label": " % of active users with exactly 1 event ", "value": "single"},
                                    {"label": " rate, % of reference ", "value": "rate_rel"},
                                    {"label": " events, % of reference ", "value": "events_rel"}]),
            dcc.Graph(id="wl-fig", style={"height": 520}),
            note("A user stays positive while one event survives, so label rates fall less than event counts "
                 "(compare the two '% of reference' views). The test week's labels come from the same thinned "
                 "archive: that is the target, not something to repair."),
        ]),
        dcc.Tab(label="Thinning sim", children=[
            html.Div([
                control("Source weeks (start dates)", picker("ts-src", *SPRING_DEFAULT)),
                control("Target weeks (start dates)", picker("ts-tgt", *RECENT_DEFAULT)),
                control("Draws", dcc.Input(id="ts-seeds", type="number", value=5, min=1, max=50)),
                control("Keep probability", dcc.RadioItems(
                    id="ts-pmode", value="auto", options=[
                        {"label": " auto: target events/user-week ÷ source", "value": "auto"},
                        {"label": " fixed:", "value": "fixed"}])),
                dcc.Slider(id="ts-p", min=0.02, max=1, step=0.02, value=0.2,
                           marks={v: str(v) for v in (0.1, 0.25, 0.5, 0.75, 1)}),
            ], style={"display": "grid", "gridTemplateColumns": "repeat(3, auto)", "gap": 16, "marginTop": 12}),
            dcc.Loading(dcc.Graph(id="ts-fig", style={"height": 480})),
            dash_table.DataTable(id="ts-table", style_cell={"fontFamily": "monospace", "fontSize": 13}),
            note("Each source-week event is kept with probability p. If the middle bar matches the target bar, "
                 "the loss behaves like random per-event dropping and the recent labels are just 'thinned normal'."),
        ]),
        dcc.Tab(label="Persistence", children=[
            dcc.Graph(id="ps-fig", style={"height": 480}),
            note("With thinning, 'active last week' is a noisier signal and count features change scale — one "
                 "reason counts relative to the cohort's weekly mean help the model."),
        ]),
        dcc.Tab(label="Test users", children=[
            control("Window length (days)", dcc.Slider(id="tu-days", min=7, max=90, step=7, value=28,
                                                       marks={7: "7", 28: "28", 56: "56", 90: "90"})),
            control("Compare with window ending", dcc.DatePickerSingle(id="tu-ref", date="2026-03-29",
                                                                      display_format="YYYY-MM-DD")),
            dcc.Graph(id="tu-fig", style={"height": 440}),
            note("How much of the test users' recent history survived. The 'end of data' window ends at each "
                 "table's last event."),
        ]),
        dcc.Tab(label="Raw archive", children=[
            html.Div([
                control("Every N days", dcc.Input(id="ra-stride", type="number", value=14, min=1, max=60)),
                control("Hour (UTC)", dcc.Input(id="ra-hour", type="number", value=14, min=0, max=23)),
                html.Button("Scan", id="ra-go", n_clicks=0, style={"height": 36}),
            ], style={"display": "flex", "gap": 24, "alignItems": "end", "marginTop": 12}),
            dcc.Loading(dcc.Graph(id="ra-fig", style={"height": 480})),
            note("Parses raw hour files (all actors, not only the cohort). If these fall too, the loss is upstream "
                 "at GH Archive. Every 14 days takes ~10-30 s; results are cached."),
        ]),
    ])
    app.layout = html.Div([sidebar, html.Div(tabs, style={"flex": 1, "padding": "8px 16px", "minWidth": 0})],
                          style={"display": "flex", "fontFamily": "system-ui, sans-serif"})
    register(app)
    return app


# ---------------------------------------------------------------- callbacks

def ts(x):
    return pd.Timestamp(x)


def ref_median(df, ref):
    return df[ts(ref[0]):ts(ref[1])].median().replace(0, np.nan)


def shade_spans(shade, ref, thr):
    if shade == "none":
        return []
    cols = NONPUSH if shade == "all non-push" else [shade]
    tot = D.daily[cols].sum(axis=1)
    return regimes(tot / ref_median(tot.to_frame(), ref).iloc[0], thr / 100)


def add_shading(fig, spans, **kw):
    for a, b, _ in spans:
        fig.add_vrect(x0=a, x1=b, fillcolor="rgba(200,0,0,0.08)", line_width=0, **kw)


def register(app):
    G = [Input("acts", "value"), Input("range", "start_date"), Input("range", "end_date"),
         Input("ref", "start_date"), Input("ref", "end_date"), Input("thr", "value"),
         Input("smooth", "value"), Input("shade", "value")]

    def unpack(acts, r0, r1, f0, f1, thr, smooth, shade):
        return acts or [], (ts(r0), ts(r1)), (f0, f1), thr, smooth, shade

    C = [Input("cmp", "start_date"), Input("cmp", "end_date"), Input("cp-k", "value"), Input("cp-tol", "value")]

    def periods(ref, c0, c1):
        return (ts(ref[0]), ts(ref[1])), (ts(c0), ts(c1))

    @app.callback(Output("vd-table", "data"), Output("vd-table", "columns"), *G, *C)
    def verdict(*a):
        *g, c0, c1, k, tol = a
        acts, rng, ref, thr, smooth, shade = unpack(*g)
        base, test = periods(ref, c0, c1)
        np_acts = [x for x in acts if x in NONPUSH] or NONPUSH
        rows = []

        def row(test_, natural, loss, observed, reads):
            rows.append({"test": test_, "natural trend predicts": natural, "collection loss predicts": loss,
                         "observed": observed, "reads as": reads})

        cm = comovement(acts, rng, ref, k, tol)
        obs, null, p = cm["coin"]
        row("Timing: do level breaks line up across series?", "breaks at unrelated dates",
            "breaks on the same days in every activity and the archive",
            f"{obs * 100:.0f}% of breaks within ±{tol} d of another series' break; {null * 100:.0f}% expected "
            f"by chance (p = {p:.3f})", "collection loss (shared cause)" if p < 0.05 else "natural / no shared timing")

        jumps = []
        for c, (edges, means) in cm["steps"].items():
            if c == "archive MB":
                continue
            for e, m0, m1 in zip(edges[1:-1], means[:-1], means[1:]):
                jumps.append((abs(m1 - m0), cm["logratio"].index[e], c, np.exp(m1 - m0)))
        if jumps:
            j = max(jumps)
            row("Abruptness: largest one-day step in level", "gradual drift (weeks to months)",
                "level changes several-fold overnight",
                f"{j[2]} ×{j[3]:.2f} on {j[1].date()}", "collection loss" if max(j[3], 1 / j[3]) >= 2.5 else "natural")

        x = pd.concat([D.daily[np_acts].sum(axis=1).rename("c"), D.size_day.rename("a")], axis=1)
        lx = np.log(deseason(x, ref).clip(lower=1e-3))
        r_rng = lx[rng[0]:rng[1]].corr().iloc[0, 1]
        r_ref = lx[base[0]:base[1]].corr().iloc[0, 1]
        row("Archive: does the cohort track the size of the whole archive?",
            "no: the archive does not care what 9,000 users do", "yes: both shrink together",
            f"corr(log daily cohort events, log file size), weekday cycle removed = {r_rng:.2f} over the date range ({r_ref:.2f} within "
            f"the reference period)", "collection loss" if r_rng > 0.6 else "natural / unclear")

        st = shape_table(acts, ref, (c0, c1))
        if len(st):
            bad = st[st["shape"] == "changed"]
            row("Shape: weekday and hour-of-day profiles", "people change when they work: the profile changes",
                "random loss keeps the profile, only the level falls; lost whole days distort the weekday profile",
                f"{len(bad)} of {len(st)} profiles change more than natural drift + thinning noise"
                + (": " + ", ".join(f"{r.activity} {r.profile}" for r in bad.itertuples()) if len(bad) else ""),
                "consistent with random loss" if len(bad) <= len(st) / 4
                else "shape changed (behaviour change or uneven / whole-day loss)")

        cp = coupling(acts, ref)[test[0]:test[1]]
        if cp.shape[1]:
            f = np.exp(np.log(cp.mean().clip(lower=1e-3)))
            worst = f.index[int(np.argmax(np.abs(np.log(f))))]
            row("Coupling: ratios between linked activities", "stable (a review comment needs a review)",
                "jump: some event types lost more than others",
                "; ".join(f"{n} ×{v:.2f}" for n, v in f.items()),
                "collection loss (uneven by type)" if max(abs(np.log(f))) > np.log(2) else "stable")

        t1, p1 = tiers(np_acts, base, test)
        t0, p0 = tiers(np_acts, *halves(ref))
        h1 = t1.iloc[-1]["events per seen user, % of base"] / (p1 * 100)
        h0 = t0.iloc[-1]["events per seen user, % of base"] / (p0 * 100)
        row("Users: lost users (churn) or lost events from everyone?",
            f"some users vanish, the rest keep their rate: heaviest users ≈ 1/keep rate ≈ ×{1 / p1:.1f} the keep rate",
            "everyone thinned: heaviest users at ≈ the keep rate, like in the control",
            f"keep rate {p1 * 100:.0f}% (control {p0 * 100:.0f}%); heaviest tier per seen user at ×{h1:.2f} the "
            f"keep rate (control ×{h0:.2f})",
            "collection loss (thinning)" if h1 < 2 * h0 and p1 < 0.5 else "natural (churn) or mixed")

        o = D.orphans
        ob, ot = o[base[0]:base[1]].mean() * 100, o[test[0]:test[1]].mean() * 100
        row("Orphans: reviews of PRs never seen opened", "unchanged", "rises: opening events lost",
            "; ".join(f"{c} {ob[c]:.0f}% → {ot[c]:.0f}%" for c in o.columns),
            "collection loss" if ot["reviews"] - ob["reviews"] > 5 else "unchanged")
        cols = ["test", "natural trend predicts", "collection loss predicts", "observed", "reads as"]
        return rows, [{"name": c, "id": c} for c in cols]

    @app.callback(Output("cp-text", "children"), Output("cp-fig", "figure"), Output("cp-corr-ref", "figure"),
                  Output("cp-corr-rng", "figure"), Output("cp-archive", "figure"), *G, *C)
    def connection(*a):
        *g, c0, c1, k, tol = a
        acts, rng, ref, thr, smooth, shade = unpack(*g)
        cm = comovement(acts, rng, ref, k, tol)
        r, lr = cm["ratio"], cm["logratio"]
        color = {**COLORS, "archive MB": "#000000"}
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
        for i, c in enumerate(r.columns):
            sm = r[c].rolling(smooth, center=True, min_periods=1).mean() * 100
            fig.add_trace(go.Scatter(x=sm.index, y=sm, name=c, legendgroup=c,
                                     line=dict(color=color[c], width=3 if c == "archive MB" else 1.5)), 1, 1)
            edges, means = cm["steps"][c]
            xs, ys = [], []
            for e0, e1, m in zip(edges[:-1], edges[1:], means):
                xs += [lr.index[e0], lr.index[e1 - 1], None]
                ys += [np.exp(m) * 100, np.exp(m) * 100, None]
            fig.add_trace(go.Scatter(x=xs, y=ys, legendgroup=c, showlegend=False, hoverinfo="skip",
                                     line=dict(color=color[c], dash="dash", width=1)), 1, 1)
            bx = [lr.index[b] for b in cm["breaks"][c]]
            fig.add_trace(go.Scatter(x=bx, y=[c] * len(bx), mode="markers", legendgroup=c, showlegend=False,
                                     marker=dict(color=color[c], size=11, symbol="line-ns-open",
                                                 line=dict(width=3)),
                                     hovertemplate="%{x|%Y-%m-%d}<extra>" + c + "</extra>"), 2, 1)
        fig.add_hline(y=100, line_dash="dot", row=1, col=1)
        fig.update_yaxes(type="log", title_text="% of normal (weekday-adjusted)", row=1, col=1)
        add_shading(fig, shade_spans(shade, ref, thr), row=1, col=1)
        fig.update_layout(**LAYOUT)
        obs, null, p = cm["coin"]
        text = html.Span([html.B(f"{obs * 100:.0f}% "), f"of breaks fall within ±{tol} days of a break in another "
                          f"series; with the same breaks placed at random dates this would be {null * 100:.0f}% "
                          f"(p = {p:.3f}, 2,000 random shifts). ",
                          html.B("Shared timing → common cause." if p < 0.05 else "No clear shared timing.")])

        def heat(cm_, title):
            f = go.Figure(go.Heatmap(z=cm_.values, x=cm_.columns, y=cm_.index, zmin=-1, zmax=1,
                                     colorscale="RdBu", reversescale=True, text=np.round(cm_.values, 2),
                                     texttemplate="%{text}"))
            f.update_layout(title=title, height=430, margin=dict(l=120, r=10, t=50, b=100),
                            template="plotly_white")
            return f

        x = pd.concat([D.daily[[c for c in acts if c in NONPUSH] or NONPUSH].sum(axis=1).rename("events"),
                       D.size_day.rename("mb")], axis=1)[rng[0]:rng[1]]
        sc = go.Figure(go.Scatter(x=x.mb, y=x.events, mode="markers", text=x.index.strftime("%Y-%m-%d"),
                                  marker=dict(color=(x.index - x.index[0]).days, colorscale="Viridis", size=6,
                                              colorbar=dict(title="day")),
                                  hovertemplate="%{text}<br>%{x:.1f} MB, %{y} events<extra></extra>"))
        lx = np.log(deseason(x, ref).clip(lower=1e-3))
        sc.update_layout(title=f"Non-push cohort events vs median hour-file size (corr of logs, weekday-adjusted: "
                               f"{lx.corr().iloc[0, 1]:.2f})", xaxis=dict(type="log", title="MB"),
                         yaxis=dict(type="log", title="cohort events / day"), template="plotly_white")
        return (text, fig, heat(cm["corr_ref"], "Reference period: day-to-day changes"),
                heat(cm["corr_rng"], "Date range: day-to-day changes"), sc)

    @app.callback(Output("sh-fig", "figure"), Output("sh-table", "data"), Output("sh-table", "columns"),
                  Output("sh-coupling", "figure"), *G, *C, Input("sh-act", "value"))
    def shape(*a):
        *g, c0, c1, k, tol, act = a
        acts, rng, ref, thr, smooth, shade = unpack(*g)
        base, test = periods(ref, c0, c1)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("weekday (0 = Mon)", "hour of day (UTC)"))
        for j, kind in enumerate(("weekday", "hour"), start=1):
            t = profile_test(act, kind, base, test)
            if t is None:
                continue
            x = np.arange(len(t["base"]))
            lo, hi = t["band"]
            fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[hi, lo[::-1]] * 100, fill="toself",
                                     fillcolor="rgba(120,120,120,0.2)", line=dict(width=0), hoverinfo="skip",
                                     name="95% if only thinned", showlegend=j == 1), 1, j)
            fig.add_trace(go.Scatter(x=x, y=t["base"] / t["base"].sum() * 100, name="reference",
                                     line=dict(color="#333"), showlegend=j == 1), 1, j)
            fig.add_trace(go.Scatter(x=x, y=t["test"] / t["test"].sum() * 100, name="comparison",
                                     line=dict(color=COLORS[act]), mode="lines+markers", showlegend=j == 1), 1, j)
        fig.update_yaxes(title_text="% of events")
        fig.update_layout(title=f"{act}: TV distance shown in the table below", **{**LAYOUT, "hovermode": "closest"})
        st = shape_table(acts, ref, (c0, c1))
        cp = coupling(acts, ref)[rng[0]:rng[1]]
        cf = go.Figure([go.Scatter(x=cp.index, y=cp[c], name=c, mode="lines+markers") for c in cp.columns])
        cf.add_hline(y=1, line_dash="dot")
        cf.add_vrect(x0=test[0], x1=test[1], fillcolor="rgba(0,0,200,0.07)", line_width=0)
        add_shading(cf, shade_spans(shade, ref, thr))
        cf.update_layout(title="Coupling between linked activities (1 = reference ratio; blue = comparison period)",
                         yaxis=dict(type="log", title="ratio ÷ reference"), **LAYOUT)
        return fig, st.to_dict("records"), [{"name": c, "id": c} for c in st.columns], cf

    @app.callback(Output("us-text", "children"), Output("us-fig", "figure"), Output("us-table", "data"),
                  Output("us-table", "columns"), *G, *C, Input("us-act", "value"))
    def users(*a):
        *g, c0, c1, k, tol, act = a
        acts, rng, ref, thr, smooth, shade = unpack(*g)
        base, test = periods(ref, c0, c1)
        sel = NONPUSH if act == "all non-push" else [act]
        t1, p1 = tiers(sel, base, test)
        t0, p0 = tiers(sel, *halves(ref))
        fig = make_subplots(rows=1, cols=2, subplot_titles=("users still seen: observed − predicted if uniform (pts)",
                                                            "events per still-seen user, % of their base rate"))
        for t, name, col, p in ((t1, "comparison vs reference", "#c2357a", p1),
                                (t0, "control: 2nd vs 1st half of reference", "#333", p0)):
            lab = [f"T{r.tier} (~{r['events/week in base']}/wk)" for _, r in t.iterrows()]
            fig.add_trace(go.Bar(x=lab, y=t["seen %"] - t["seen % if uniform"], name=name, marker_color=col), 1, 1)
            fig.add_trace(go.Scatter(x=lab, y=t["events per seen user, % of base"], name=name, showlegend=False,
                                     mode="lines+markers", line=dict(color=col)), 1, 2)
            fig.add_trace(go.Scatter(x=lab, y=[p * 100] * len(lab), showlegend=False, hoverinfo="skip",
                                     line=dict(color=col, dash="dot")), 1, 2)
        fig.update_layout(**{**LAYOUT, "hovermode": "closest"}, barmode="group")
        text = (f"Overall keep rate: {p1 * 100:.0f}% of the reference rate in the comparison period "
                f"(control {p0 * 100:.0f}%). Dotted line = the keep rate: under pure thinning every tier sits on it; "
                f"under churn the still-seen users stay near 100%.")
        t = t1.round(2)
        return text, fig, t.to_dict("records"), [{"name": c, "id": c} for c in t.columns]

    @app.callback(Output("or-text", "children"), Output("or-fig", "figure"), *G, *C)
    def orphans(*a):
        *g, c0, c1, k, tol = a
        acts, rng, ref, thr, smooth, shade = unpack(*g)
        base, test = periods(ref, c0, c1)
        o = D.orphans[rng[0]:rng[1]].rolling(smooth, center=True, min_periods=1).mean() * 100
        fig = go.Figure([go.Scatter(x=o.index, y=o[c], name=c) for c in o.columns])
        add_shading(fig, shade_spans(shade, ref, thr))
        fig.update_layout(yaxis_title="% with PR opening not captured", **LAYOUT)
        ob, ot = D.orphans[base[0]:base[1]].mean() * 100, D.orphans[test[0]:test[1]].mean() * 100
        text = "Reference → comparison: " + " · ".join(f"{c} {ob[c]:.1f}% → {ot[c]:.1f}%" for c in o.columns)
        return text, fig

    @app.callback(Output("ov-text", "children"), Output("ov-table", "data"), Output("ov-table", "columns"),
                  Output("ov-size", "figure"), *G)
    def overview(*a):
        acts, rng, ref, thr, smooth, shade = unpack(*a)
        pct = D.daily / ref_median(D.daily, ref)
        rows = []
        for k in acts:
            for s, e, m in regimes(pct[k], thr / 100):
                rows.append({"activity": k, "from": str(s.date()), "to": str((e - DAY).date()),
                             "days": (e - s).days, "mean % of normal": round(m * 100, 1)})
        last4 = D.last_day - 27 * DAY
        text = [html.B(f"Data {B.DATA_START.date()} → {D.last_day.date()} · {D.n:,} cohort users · "
                       f"{len(D.sizes):,} hour files, {len(D.missing_hours)} missing"), html.Br(),
                "Last 4 weeks, % of normal: " + " · ".join(f"{k} {pct[k][last4:].mean() * 100:.0f}%" for k in acts)]
        sz = D.sizes.groupby(D.sizes.index.floor("D")).median()[rng[0]:rng[1]]
        fig = go.Figure(go.Scatter(x=sz.index, y=sz.rolling(smooth, center=True, min_periods=1).mean(),
                                   name="MB"))
        add_shading(fig, shade_spans(shade, ref, thr))
        fig.update_layout(title="Raw hour-file size (median per day, MB)", height=340, **LAYOUT)
        cols = [{"name": c, "id": c} for c in ["activity", "from", "to", "days", "mean % of normal"]]
        return text, rows, cols, fig

    @app.callback(Output("dv-fig", "figure"), *G, Input("dv-metric", "value"), Input("dv-log", "value"))
    def daily(*a):
        *g, metric, log = a
        acts, rng, ref, thr, smooth, shade = unpack(*g)
        if metric == "pct":
            df, yt = D.daily / ref_median(D.daily, ref) * 100, "% of normal"
        elif metric == "raw":
            df, yt = D.daily, "events per day"
        else:
            df, yt = D.zero * 100, "% of hours with zero events"
        df = df[rng[0]:rng[1]].rolling(smooth, center=True, min_periods=1).mean()
        fig = go.Figure([go.Scatter(x=df.index, y=df[k], name=k, line=dict(color=COLORS[k])) for k in acts])
        if metric == "pct":
            fig.add_hline(y=100, line_dash="dot")
            fig.add_hline(y=thr, line_dash="dash", line_color="red")
        add_shading(fig, shade_spans(shade, ref, thr))
        fig.update_layout(yaxis_title=yt, yaxis_type="log" if log and metric != "zero" else "linear", **LAYOUT)
        return fig

    @app.callback(Output("hm-fig", "figure"), *G, Input("hm-act", "value"))
    def heatmap(*a):
        *g, act = a
        acts, rng, ref, thr, smooth, shade = unpack(*g)
        h = D.hourly[NONPUSH].sum(axis=1) if act == "all non-push" else D.hourly[act]
        hm = h.groupby([h.index.floor("D"), h.index.hour]).sum().unstack(fill_value=0)
        hm = hm.div(ref_median(hm, ref), axis=1) * 100
        hm = hm[rng[0]:rng[1]]
        fig = go.Figure(go.Heatmap(z=hm.T.values, x=hm.index, y=hm.columns, colorscale="Viridis", zmin=0,
                                   zmax=150, colorbar=dict(title="% normal"),
                                   hovertemplate="%{x|%Y-%m-%d} %{y}:00 UTC<br>%{z:.0f}%<extra></extra>"))
        fig.update_layout(yaxis_title="hour (UTC)", **{**LAYOUT, "hovermode": "closest"})
        return fig

    @app.callback(Output("wl-fig", "figure"), *G, Input("wl-metric", "value"))
    def weekly(*a):
        *g, metric = a
        acts, rng, ref, thr, smooth, shade = unpack(*g)
        wk = D.wk.set_index("week")
        base, rel = metric.replace("_rel", ""), metric.endswith("_rel")
        fig = go.Figure()
        for k in acts:
            y = wk[f"{base}_{k}"]
            if rel:
                y = y / y[ts(ref[0]):ts(ref[1])].mean()
            y = y[rng[0]:rng[1]] * (100 if base != "events" or rel else 1)
            fig.add_trace(go.Scatter(x=y.index, y=y, name=k, mode="lines+markers", line=dict(color=COLORS[k])))
        e = wk.eligible[rng[0]:rng[1]]
        fig.add_trace(go.Bar(x=e.index, y=e, name="eligible users", yaxis="y2", opacity=0.15,
                             marker_color="#888"))
        add_shading(fig, shade_spans(shade, ref, thr))
        fig.update_layout(yaxis=dict(title={"rate": "%", "events": "events", "single": "%"}.get(metric, "% of ref"),
                                     type="log" if metric == "events" else "linear"),
                          yaxis2=dict(overlaying="y", side="right", showgrid=False, title="eligible"), **LAYOUT)
        return fig

    @app.callback(Output("ts-fig", "figure"), Output("ts-table", "data"), Output("ts-table", "columns"),
                  Input("acts", "value"), Input("ts-src", "start_date"), Input("ts-src", "end_date"),
                  Input("ts-tgt", "start_date"), Input("ts-tgt", "end_date"), Input("ts-seeds", "value"),
                  Input("ts-pmode", "value"), Input("ts-p", "value"))
    def thinning(acts, s0, s1, t0, t1, seeds, pmode, pfix):
        src = [w for w in D.weeks if ts(s0) <= w.normalize() <= ts(s1)]
        tgt = [w for w in D.weeks if ts(t0) <= w.normalize() <= ts(t1)]
        rng = np.random.default_rng(0)
        rows = []
        for k in [k for k in acts if k in NONPUSH or pmode == "fixed"]:
            ev_pu = lambda ws: np.mean([D.wcount[(w, k)][D.elig[w]].mean() for w in ws])  # noqa: E731
            rate = lambda ws: np.mean([(D.wcount[(w, k)][D.elig[w]] > 0).mean() for w in ws])  # noqa: E731
            if not src or not tgt:
                continue
            p = min(ev_pu(tgt) / ev_pu(src), 1) if pmode == "auto" else pfix
            sims = []
            for _ in range(int(seeds or 1)):
                r = []
                for w in src:
                    u = D.week_uids(k, w)
                    u = u[rng.random(len(u)) < p]
                    r.append((np.bincount(u, minlength=D.n)[D.elig[w]] > 0).mean())
                sims.append(np.mean(r))
            rows.append({"activity": k, "p kept": round(p, 3), "source %": round(rate(src) * 100, 2),
                         "source thinned %": round(np.mean(sims) * 100, 2),
                         "± sd": round(np.std(sims) * 100, 2), "target %": round(rate(tgt) * 100, 2)})
        df = pd.DataFrame(rows)
        fig = go.Figure()
        if len(df):
            for c in ["source %", "source thinned %", "target %"]:
                fig.add_trace(go.Bar(x=df.activity, y=df[c], name=c, text=df[c], textposition="outside"))
        fig.update_layout(barmode="group", yaxis_title="positive rate %",
                          title=f"{len(src)} source weeks → {len(tgt)} target weeks", **LAYOUT)
        return fig, df.to_dict("records"), [{"name": c, "id": c} for c in df.columns]

    @app.callback(Output("ps-fig", "figure"), *G)
    def persist(*a):
        acts, rng, ref, thr, smooth, shade = unpack(*a)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("P(active next week | active)",
                                                            "P(active next week | inactive)"))
        for k in acts:
            x, stay, ret = [], [], []
            for w0, w1 in zip(D.weeks[:-1], D.weeks[1:]):
                if not rng[0] <= w1 <= rng[1]:
                    continue
                both = D.elig[w0] & D.elig[w1]
                a0, a1 = D.wcount[(w0, k)][both] > 0, D.wcount[(w1, k)][both] > 0
                x.append(w1)
                stay.append(a1[a0].mean() * 100 if a0.any() else np.nan)
                ret.append(a1[~a0].mean() * 100 if (~a0).any() else np.nan)
            fig.add_trace(go.Scatter(x=x, y=stay, name=k, legendgroup=k, line=dict(color=COLORS[k])), 1, 1)
            fig.add_trace(go.Scatter(x=x, y=ret, name=k, legendgroup=k, showlegend=False,
                                     line=dict(color=COLORS[k])), 1, 2)
        spans = shade_spans(shade, ref, thr)
        add_shading(fig, spans, row=1, col=1)
        add_shading(fig, spans, row=1, col=2)
        fig.update_yaxes(title_text="%")
        fig.update_layout(**LAYOUT)
        return fig

    @app.callback(Output("tu-fig", "figure"), Input("acts", "value"), Input("tu-days", "value"),
                  Input("tu-ref", "date"))
    def test_users(acts, days, ref_end):
        a_vals, b_vals = [], []
        for k in acts:
            e = D.ev[k]
            for end, out in ((ts(ref_end), a_vals), (e.t.max(), b_vals)):
                m = (e.t.values >= np.datetime64(end - days * DAY)) & (e.t.values < np.datetime64(end))
                out.append((np.bincount(e.uid.values[m], minlength=D.n)[D.test_cidx] > 0).mean() * 100)
        fig = go.Figure([go.Bar(x=acts, y=a_vals, name=f"{days} d to {ts(ref_end).date()}"),
                         go.Bar(x=acts, y=b_vals, name=f"last {days} d of each table")])
        fig.update_layout(barmode="group", yaxis_title="% of test users with ≥1 event",
                          title=f"{len(D.test_cidx):,} test users", **LAYOUT)
        return fig

    @app.callback(Output("ra-fig", "figure"), Input("ra-go", "n_clicks"), State("ra-stride", "value"),
                  State("ref", "start_date"), State("ref", "end_date"), State("ra-hour", "value"))
    def raw(n, stride, f0, f1, hour):
        fig = go.Figure()
        if not n:
            fig.update_layout(title="Press Scan", **LAYOUT)
            return fig
        raw = scan_archive(int(stride or 14), int(hour or 0))
        rel = raw / ref_median(raw, (f0, f1)) * 100
        for c in raw.columns:
            fig.add_trace(go.Scatter(x=rel.index, y=rel[c], name=c, mode="lines+markers",
                                     line=dict(color=COLORS.get(c, "#999")),
                                     customdata=raw[c], hovertemplate="%{y:.0f}% (%{customdata} events)"))
        fig.add_hline(y=100, line_dash="dot")
        fig.update_layout(yaxis_type="log", yaxis_title="% of reference median",
                          title=f"Raw events at {hour}:00 UTC, every {stride} days", **LAYOUT)
        return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8051)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    D = Data()
    build_app().run(host=args.host, port=args.port, debug=args.debug)
