"""Analyze holdout failure cases for LightGBM and saved experiment runs.

Install once:  pip install dash plotly lightgbm scikit-learn
Run LightGBM:  python exam_fail_case.py
Analyze a LightGBM backtest:
               python exam_fail_case.py --run-dir experiments/lightgbm_baseline/outputs/RUN
Run a saved experiment:
               python exam_fail_case.py --run-dir experiments/relgt/outputs/RUN
               python exam_fail_case.py --run-dir experiments/gru_relational_ple/outputs/RUN

The dashboard includes the original overview/features/time/users/case tabs plus:
- FP Explorer: filter false positives by probability, compare them with TNs,
  inspect feature-space patterns, identify users with repeated FPs, and inspect
  model contributions when the selected experiment supports them.
- Threshold Lab: interactively inspect the FP/FN trade-off at alternative
  thresholds. This is a holdout diagnostic only; do not use the holdout to select
  the final threshold reported in an experiment.
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from experiments.lightgbm_baseline import run as baseline


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "lightgbm_baseline" / "outputs"

OUTCOMES = ("TP", "TN", "FP", "FN")
REFERENCE_OUTCOME = {"FN": "TP", "FP": "TN"}
FAILURE_TYPE = {"FN": "false_negative", "FP": "false_positive"}
COLORS = {"TP": "#2ca02c", "TN": "#9ecae1", "FP": "#ff7f0e", "FN": "#d62728"}
MAX_POINTS = 15000
PREDICTION_BRANCHES = ("fused", "relgt", "lightgbm", "s2p2")


# --------------------------------------------------------------------------- #
# Case tables
# --------------------------------------------------------------------------- #
def build_case_table(
    holdout: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    github_user_ids: pd.Series,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Return one row per user-week-label, with a unique integer case index."""
    frames: list[pd.DataFrame] = []
    for label in baseline.LABELS:
        actual = holdout[label].to_numpy(dtype=np.int8)
        probability = np.asarray(probabilities[label], dtype=float)
        threshold = float(thresholds[label])
        predicted = (probability >= threshold).astype(np.int8)

        rows = holdout[["user_id", "cutoff", *feature_columns]].copy()
        rows.insert(1, "github_user_id", rows["user_id"].map(github_user_ids))
        rows["label"] = label
        rows["actual"] = actual
        rows["predicted"] = predicted
        rows["probability"] = probability
        rows["threshold"] = threshold
        rows["margin"] = probability - threshold
        rows["outcome"] = np.select(
            [
                (actual == 1) & (predicted == 1),
                (actual == 0) & (predicted == 0),
                (actual == 0) & (predicted == 1),
            ],
            ["TP", "TN", "FP"],
            default="FN",
        )
        rows["correct"] = actual == predicted
        frames.append(rows)

    # IMPORTANT: reset_index prevents duplicated case IDs across labels.
    cases = pd.concat(frames, ignore_index=True)
    cases.index = pd.RangeIndex(len(cases), name="case_id")
    return cases


def build_failure_cases(cases: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return misclassified rows, retaining the unique case_id for traceability."""
    failures = cases.loc[~cases["correct"]].copy().reset_index()
    failures["failure_type"] = failures["outcome"].map(FAILURE_TYPE)
    columns = [
        "case_id", "user_id", "github_user_id", "cutoff", "label", "actual",
        "predicted", "probability", "threshold", "margin", "failure_type",
        *feature_columns,
    ]
    return failures[columns]


def build_fp_cases(cases: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return only FPs, ordered from highest to lowest predicted probability."""
    fp = cases.loc[cases["outcome"] == "FP"].copy().reset_index()
    columns = [
        "case_id", "user_id", "github_user_id", "cutoff", "label", "actual",
        "predicted", "probability", "threshold", "margin", *feature_columns,
    ]
    return fp[columns].sort_values("probability", ascending=False)


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def confusion_summary(cases: pd.DataFrame) -> pd.DataFrame:
    counts = (
        cases.groupby(["label", "outcome"]).size().unstack(fill_value=0)
        .reindex(columns=list(OUTCOMES), fill_value=0)
    )
    total = counts.sum(axis=1).replace(0, np.nan)
    out = counts.copy()
    out["total"] = total
    out["accuracy"] = (counts["TP"] + counts["TN"]) / total
    out["precision"] = counts["TP"] / (counts["TP"] + counts["FP"]).replace(0, np.nan)
    out["recall"] = counts["TP"] / (counts["TP"] + counts["FN"]).replace(0, np.nan)
    out["fpr"] = counts["FP"] / (counts["FP"] + counts["TN"]).replace(0, np.nan)
    return out.round(3)


def compare_features(
    cases: pd.DataFrame, label: str, outcome: str, feature_columns: list[str], top: int
) -> pd.DataFrame | None:
    """Standardized mean difference: error group vs correct cases of same true class."""
    sub = cases.loc[cases["label"] == label]
    bad = sub.loc[sub["outcome"] == outcome, feature_columns]
    ok = sub.loc[sub["outcome"] == REFERENCE_OUTCOME[outcome], feature_columns]
    if bad.empty or ok.empty:
        return None
    pooled = pd.concat([bad, ok]).apply(pd.to_numeric, errors="coerce")
    bad_num = bad.apply(pd.to_numeric, errors="coerce")
    ok_num = ok.apply(pd.to_numeric, errors="coerce")
    pooled_std = pooled.std(ddof=0).replace(0, np.nan)
    smd = ((bad_num.mean() - ok_num.mean()) / pooled_std).replace([np.inf, -np.inf], np.nan).dropna()
    if smd.empty:
        return None
    order = smd.abs().sort_values(ascending=False).head(int(top)).index
    return pd.DataFrame({"std_diff": smd[order]}).rename_axis("feature")


def metric_counts(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=np.int8)
    predicted = np.asarray(predicted, dtype=np.int8)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall
        else np.nan
    )
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "fpr": fpr, "f1": f1,
        "predicted_positive": int(predicted.sum()),
        "predicted_positive_rate": float(predicted.mean()) if len(predicted) else np.nan,
    }


def threshold_sweep(sub: pd.DataFrame) -> pd.DataFrame:
    actual = sub["actual"].to_numpy(dtype=np.int8)
    probability = sub["probability"].to_numpy(dtype=float)
    rows = []
    for threshold in np.linspace(0, 1, 101):
        predicted = (probability >= threshold).astype(np.int8)
        rows.append({"threshold": threshold, **metric_counts(actual, predicted)})
    return pd.DataFrame(rows)


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=message, xaxis={"visible": False}, yaxis={"visible": False}, height=300
    )
    return fig


def metric_card(label: str, value: str, subtitle: str | None = None) -> html.Div:
    from dash import html

    children = [html.Div(label, style={"fontSize": "12px", "color": "#666"}),
                html.Div(value, style={"fontSize": "25px", "fontWeight": "bold"})]
    if subtitle:
        children.append(html.Div(subtitle, style={"fontSize": "11px", "color": "#777"}))
    return html.Div(
        children,
        style={"border": "1px solid #ddd", "borderRadius": "7px", "padding": "10px 14px",
                "minWidth": "145px", "flex": "1"},
    )


# --------------------------------------------------------------------------- #
# Dash app
# --------------------------------------------------------------------------- #
def make_app(
    cases: pd.DataFrame,
    models: dict,
    feature_columns: list[str],
    experiment_name: str = "LightGBM",
) -> Dash:
    from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html

    labels = list(baseline.LABELS)
    summary = confusion_summary(cases).reset_index()
    numeric_features = [
        c for c in feature_columns
        if pd.api.types.is_numeric_dtype(cases[c])
    ]
    if not numeric_features:
        raise ValueError("No numeric feature columns are available for the dashboard.")

    default_x = numeric_features[0]
    default_y = numeric_features[1] if len(numeric_features) > 1 else numeric_features[0]

    def contributions(row: pd.Series) -> pd.Series | None:
        """Per-feature model contributions toward the positive class (if available)."""
        model = models.get(row["label"])
        x = row[feature_columns].to_frame().T.astype(float)
        for predictor in (model, getattr(model, "booster_", None)):
            if predictor is None:
                continue
            try:
                values = np.asarray(predictor.predict(x, pred_contrib=True)).reshape(-1)
                if len(values) >= len(feature_columns):
                    return pd.Series(values[:len(feature_columns)], index=feature_columns)
            except Exception:
                continue
        return None

    def render_case_detail(case_id: int):
        if case_id not in cases.index:
            return f"case_id {case_id} not found.", empty_figure("Case not found"), []

        row = cases.loc[case_id]
        header = (
            f"case {case_id} | {row['label']} | user {row['user_id']} "
            f"(GitHub {row['github_user_id']}) | {pd.Timestamp(row['cutoff']).date()} | "
            f"actual={row['actual']} predicted={row['predicted']} → {row['outcome']} | "
            f"p={row['probability']:.3f}, threshold={row['threshold']:.3f}, "
            f"margin={row['margin']:+.3f}"
        )

        # For FP, this reference group is correctly classified actual-negative cases (TNs).
        same_class = cases.loc[
            (cases["label"] == row["label"])
            & (cases["actual"] == row["actual"])
            & cases["correct"]
        ]
        median = same_class[feature_columns].apply(pd.to_numeric, errors="coerce").median()
        values = row[feature_columns].apply(pd.to_numeric, errors="coerce")
        contrib = contributions(row)

        if contrib is not None:
            order = contrib.abs().sort_values(ascending=False).head(15).index
            plot = pd.DataFrame({"feature": order, "score": contrib[order].to_numpy()})
            title = "Top model contributions (+ pushes toward label=1)"
            score_name = "contribution"
        else:
            std = same_class[feature_columns].apply(pd.to_numeric, errors="coerce").std(ddof=0)
            z = ((values - median) / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
            order = z.abs().sort_values(ascending=False).head(15).index
            plot = pd.DataFrame({"feature": order, "score": z[order].to_numpy()})
            title = "Most unusual features vs correctly classified cases of the same true class"
            score_name = "z_score"

        if plot.empty:
            fig = empty_figure("No usable feature values for this case")
        else:
            fig = px.bar(
                plot, x="score", y="feature", orientation="h", title=title,
                color=np.where(plot["score"] >= 0, "positive", "negative"),
                color_discrete_map={"positive": "#d62728", "negative": "#1f77b4"},
            )
            fig.update_layout(yaxis={"autorange": "reversed"}, showlegend=False, height=470)

        table = pd.DataFrame({
            "feature": order,
            "value": values[order].round(4).to_numpy(),
            "correct_case_median": median[order].round(4).to_numpy(),
            score_name: (contrib[order] if contrib is not None else z[order]).round(4).to_numpy(),
        })
        return header, fig, table.to_dict("records")

    app = Dash(
        __name__,
        title=f"{experiment_name} failure analysis",
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div(
        style={"fontFamily": "sans-serif", "margin": "16px 24px"},
        children=[
            html.H2(f"{experiment_name} holdout: error analysis"),
            html.Div(
                [html.Label("Label  "), dcc.Dropdown(
                    labels, labels[0], id="label", clearable=False, style={"width": "320px"}
                )],
                style={"marginBottom": "12px"},
            ),
            dcc.Tabs(
                id="tabs", value="fp", children=[
                    dcc.Tab(label="FP Explorer", value="fp", children=[
                        html.H3("False-positive exploration"),
                        html.Div(
                            "A false positive is an actual 0 that the model predicted as 1. "
                            "The reference group in this tab is TN (actual 0, predicted 0).",
                            style={"color": "#555", "marginBottom": "12px"},
                        ),
                        html.Div([
                            html.Div([
                                html.Label("FP probability range"),
                                dcc.RangeSlider(
                                    0, 1, 0.01, value=[0, 1], id="fp-prob-range",
                                    marks={0: "0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1: "1"},
                                    allowCross=False, tooltip={"always_visible": True},
                                ),
                            ], style={"width": "48%", "minWidth": "330px"}),
                            html.Div([
                                html.Label("Scatter X feature"),
                                dcc.Dropdown(
                                    numeric_features, default_x, id="fp-x-feature", clearable=False
                                ),
                            ], style={"width": "24%", "minWidth": "220px"}),
                            html.Div([
                                html.Label("Scatter Y feature"),
                                dcc.Dropdown(
                                    numeric_features, default_y, id="fp-y-feature", clearable=False
                                ),
                            ], style={"width": "24%", "minWidth": "220px"}),
                        ], style={"display": "flex", "flexWrap": "wrap", "gap": "20px",
                                  "alignItems": "center", "margin": "10px 0 20px"}),
                        html.Div(id="fp-kpis", style={"display": "flex", "flexWrap": "wrap",
                                                       "gap": "10px", "marginBottom": "16px"}),
                        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "12px"}, children=[
                            dcc.Graph(id="fp-prob-fig", style={"width": "48%", "minWidth": "420px"}),
                            dcc.Graph(id="fp-scatter", style={"width": "50%", "minWidth": "420px"}),
                        ]),
                        html.H4("Users with the most false positives"),
                        dcc.Graph(id="fp-users-fig"),
                        html.H4("False-positive cases (filter and sort the table)"),
                        dash_table.DataTable(
                            id="fp-table", columns=[], data=[],
                            row_selectable="single", selected_rows=[], active_cell=None,
                            sort_action="native", filter_action="native", page_action="native",
                            page_size=12, style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "padding": "5px 8px",
                                        "minWidth": "85px", "maxWidth": "220px",
                                        "whiteSpace": "normal"},
                            style_header={"fontWeight": "bold", "backgroundColor": "#f4f4f4"},
                        ),
                        html.Div(id="fp-case-header", style={"fontWeight": "bold", "margin": "12px 0"}),
                        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "12px"}, children=[
                            dcc.Graph(id="fp-case-contrib", style={"width": "55%", "minWidth": "420px"}),
                            dash_table.DataTable(
                                id="fp-case-table", data=[], columns=[],
                                style_cell={"textAlign": "right", "padding": "4px 8px"},
                                style_header={"fontWeight": "bold"},
                            ),
                        ]),
                    ]),
                    dcc.Tab(label="Threshold Lab", value="threshold", children=[
                        html.H3("Threshold what-if analysis"),
                        html.Div(
                            "This recalculates predictions on the holdout set at a different threshold. "
                            "Use it to understand the FP/FN trade-off, not to tune the final threshold "
                            "on the holdout data.",
                            style={"color": "#8a4b08", "marginBottom": "12px"},
                        ),
                        html.Div([
                            html.Label("Hypothetical decision threshold"),
                            dcc.Slider(
                                0, 1, 0.01, value=float(cases.loc[cases["label"] == labels[0], "threshold"].iloc[0]),
                                id="threshold-slider", marks={0: "0", 0.5: "0.5", 1: "1"},
                                tooltip={"always_visible": True},
                            ),
                        ], style={"maxWidth": "750px", "margin": "20px 0"}),
                        html.Div(id="threshold-kpis", style={"display": "flex", "flexWrap": "wrap",
                                                                "gap": "10px", "margin": "14px 0"}),
                        dash_table.DataTable(
                            id="threshold-table", columns=[], data=[],
                            style_cell={"textAlign": "right", "padding": "5px 9px"},
                            style_header={"fontWeight": "bold"},
                        ),
                        dcc.Graph(id="threshold-curve"),
                    ]),
                    dcc.Tab(label="Overview", value="overview", children=[
                        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                            dcc.Graph(id="cm", style={"width": "40%", "minWidth": "340px"}),
                            dcc.Graph(id="hist", style={"width": "58%", "minWidth": "400px"}),
                        ]),
                        html.H4("Metrics (all labels)"),
                        dash_table.DataTable(
                            summary.to_dict("records"),
                            [{"name": c, "id": c} for c in summary.columns],
                            style_cell={"textAlign": "right", "padding": "4px 10px"},
                            style_header={"fontWeight": "bold"},
                        ),
                    ]),
                    dcc.Tab(label="Features", value="features", children=[
                        html.Div(style={"display": "flex", "gap": "32px", "margin": "12px 0"}, children=[
                            html.Div([
                                html.Label("Error type"),
                                dcc.RadioItems(
                                    [{"label": " FN vs TP", "value": "FN"},
                                     {"label": " FP vs TN", "value": "FP"}],
                                    "FP", id="feat-outcome", inline=True,
                                ),
                            ]),
                            html.Div([
                                html.Label("Top features"),
                                dcc.Slider(5, 40, 5, value=15, id="feat-top", marks=None,
                                           tooltip={"always_visible": True}),
                            ], style={"width": "320px"}),
                        ]),
                        html.Div("Click a bar to see that feature's distribution.",
                                 style={"color": "#666"}),
                        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                            dcc.Graph(id="feat-bar", style={"width": "48%", "minWidth": "400px"}),
                            dcc.Graph(id="feat-dist", style={"width": "50%", "minWidth": "400px"}),
                        ]),
                    ]),
                    dcc.Tab(label="Over time", value="time", children=[dcc.Graph(id="time-fig")]),
                    dcc.Tab(label="Users", value="users", children=[
                        html.Div([
                            html.Label("Users shown"),
                            dcc.Slider(5, 50, 5, value=20, id="users-n", marks=None,
                                       tooltip={"always_visible": True}),
                        ], style={"width": "320px", "margin": "12px 0"}),
                        dcc.Graph(id="users-fig"),
                    ]),
                    dcc.Tab(label="Case explorer", value="cases", children=[
                        html.Div(style={"display": "flex", "gap": "32px", "margin": "12px 0",
                                        "alignItems": "center"}, children=[
                            html.Div([
                                html.Label("Show outcomes"),
                                dcc.Checklist(
                                    [{"label": f" {o}", "value": o} for o in OUTCOMES],
                                    ["FP", "FN"], id="case-outcomes", inline=True,
                                ),
                            ]),
                            html.Div([
                                html.Label("Or enter case_id  "),
                                dcc.Input(id="case-id", type="number", debounce=True, min=0,
                                          step=1, placeholder="case_id"),
                            ]),
                        ]),
                        dcc.Graph(id="case-scatter"),
                        html.Div(id="case-header", style={"fontWeight": "bold", "margin": "8px 0"}),
                        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                            dcc.Graph(id="case-contrib", style={"width": "55%", "minWidth": "400px"}),
                            dash_table.DataTable(
                                id="case-table", columns=[], data=[],
                                style_cell={"textAlign": "right", "padding": "4px 10px"},
                                style_header={"fontWeight": "bold"},
                            ),
                        ]),
                    ]),
                ],
            ),
        ],
    )

    # -- FP Explorer -------------------------------------------------------- #
    @app.callback(
        Output("fp-kpis", "children"), Output("fp-prob-fig", "figure"),
        Output("fp-scatter", "figure"), Output("fp-users-fig", "figure"),
        Output("fp-table", "data"), Output("fp-table", "columns"),
        Input("label", "value"), Input("fp-prob-range", "value"),
        Input("fp-x-feature", "value"), Input("fp-y-feature", "value"),
    )
    def fp_explorer(label, probability_range, x_feature, y_feature):
        sub = cases.loc[cases["label"] == label]
        negative = sub.loc[sub["actual"] == 0]
        fp_all = negative.loc[negative["outcome"] == "FP"]
        tn = negative.loc[negative["outcome"] == "TN"]
        low, high = probability_range or [0, 1]
        fp = fp_all.loc[fp_all["probability"].between(low, high, inclusive="both")]

        threshold = float(sub["threshold"].iloc[0])
        mean_probability = fp["probability"].mean() if not fp.empty else np.nan
        median_margin = fp["margin"].median() if not fp.empty else np.nan
        cards = [
            metric_card("All FPs", f"{len(fp_all):,}", f"for label {label}"),
            metric_card("Filtered FPs", f"{len(fp):,}", f"p in [{low:.2f}, {high:.2f}]"),
            metric_card("Mean FP probability", f"{mean_probability:.3f}" if np.isfinite(mean_probability) else "—"),
            metric_card("Median threshold margin", f"{median_margin:+.3f}" if np.isfinite(median_margin) else "—",
                        f"threshold = {threshold:.3f}"),
        ]

        prob_data = pd.concat([
            fp_all.assign(plot_group="FP"), tn.assign(plot_group="TN")
        ], ignore_index=True)
        if prob_data.empty:
            prob_fig = empty_figure("No actual-negative cases")
        else:
            prob_fig = px.histogram(
                prob_data, x="probability", color="plot_group", nbins=45,
                barmode="overlay", opacity=0.62, histnorm="probability density",
                color_discrete_map={"FP": COLORS["FP"], "TN": COLORS["TN"]},
                title="Probability distributions among actual-negative cases (FP vs TN)",
            )
            prob_fig.add_vline(x=threshold, line_dash="dash", annotation_text="decision threshold")
            prob_fig.update_layout(yaxis_title="Probability density")

        # Plot all filtered FPs and a bounded random TN reference sample.
        tn_budget = max(MAX_POINTS - len(fp), 0)
        tn_sample = tn.sample(min(len(tn), tn_budget), random_state=7) if tn_budget else tn.iloc[0:0]
        cloud = pd.concat([fp, tn_sample]).copy()
        if cloud.empty:
            scatter_fig = empty_figure("No FP/TN points in this probability selection")
        else:
            cloud = cloud.reset_index().rename(columns={"index": "case_id"})
            scatter_fig = px.scatter(
                cloud, x=x_feature, y=y_feature, color="outcome", custom_data=["case_id"],
                hover_data={"user_id": True, "github_user_id": True, "cutoff": True,
                            "probability": ":.3f", "margin": ":+.3f", "case_id": True},
                color_discrete_map=COLORS, opacity=0.7,
                title=f"Feature space: filtered FPs vs sampled TNs ({len(cloud):,} points)",
            )
            scatter_fig.update_layout(height=470)

        if fp.empty:
            user_fig = empty_figure("No false positives in the selected probability range")
        else:
            users = fp.assign(user=fp["github_user_id"].fillna(fp["user_id"]).astype(str))
            by_user = users.groupby("user").agg(
                fp_count=("probability", "size"),
                mean_probability=("probability", "mean"),
                weeks=("cutoff", "nunique"),
            ).sort_values(["fp_count", "mean_probability"], ascending=False).head(20).reset_index()
            user_fig = px.bar(
                by_user.sort_values("fp_count"), x="fp_count", y="user", orientation="h",
                hover_data={"mean_probability": ":.3f", "weeks": True},
                title="Users with the most FPs in the selected probability range",
            )
            user_fig.update_layout(height=max(350, 24 * len(by_user) + 100),
                                   yaxis_title="GitHub user ID", xaxis_title="FP cases")

        table = fp.reset_index().rename(columns={"index": "case_id"}).copy()
        table["cutoff"] = table["cutoff"].astype(str)
        visible_cols = ["case_id", "github_user_id", "user_id", "cutoff", "probability", "threshold", "margin"]
        table_data = table[visible_cols].sort_values("probability", ascending=False).to_dict("records")
        table_columns = [
            {"name": c, "id": c, **({"type": "numeric", "format": {"specifier": ".3f"}}
                                      if c in {"probability", "threshold", "margin"} else {})}
            for c in visible_cols
        ]
        return cards, prob_fig, scatter_fig, user_fig, table_data, table_columns

    @app.callback(
        Output("fp-case-header", "children"), Output("fp-case-contrib", "figure"),
        Output("fp-case-table", "data"), Output("fp-case-table", "columns"),
        Input("fp-scatter", "clickData"), Input("fp-table", "active_cell"),
        State("fp-table", "derived_viewport_data"),
    )
    def fp_case_detail(scatter_click, active_cell, visible_rows):
        case_id = None
        if ctx.triggered_id == "fp-scatter" and scatter_click:
            case_id = int(scatter_click["points"][0]["customdata"][0])
        elif active_cell and visible_rows:
            row_index = active_cell.get("row")
            if row_index is not None and row_index < len(visible_rows):
                case_id = int(visible_rows[row_index]["case_id"])
        if case_id is None:
            return "Click a FP point or select a row in the table to inspect its explanation.", empty_figure("Select an FP case"), [], []
        header, fig, data = render_case_detail(case_id)
        cols = [{"name": c, "id": c} for c in (data[0].keys() if data else [])]
        return header, fig, data, cols

    # -- Threshold Lab ------------------------------------------------------ #
    @app.callback(Output("threshold-slider", "value"), Input("label", "value"))
    def reset_threshold_for_label(label):
        return float(cases.loc[cases["label"] == label, "threshold"].iloc[0])

    @app.callback(
        Output("threshold-kpis", "children"), Output("threshold-table", "data"),
        Output("threshold-table", "columns"), Output("threshold-curve", "figure"),
        Input("label", "value"), Input("threshold-slider", "value"),
    )
    def threshold_lab(label, selected_threshold):
        sub = cases.loc[cases["label"] == label]
        actual = sub["actual"].to_numpy(dtype=np.int8)
        probability = sub["probability"].to_numpy(dtype=float)
        original_threshold = float(sub["threshold"].iloc[0])
        selected_threshold = float(selected_threshold)
        baseline_metrics = metric_counts(actual, (probability >= original_threshold).astype(np.int8))
        current_metrics = metric_counts(actual, (probability >= selected_threshold).astype(np.int8))
        baseline_metrics["threshold"] = original_threshold
        current_metrics["threshold"] = selected_threshold

        fp_delta = int(current_metrics["FP"] - baseline_metrics["FP"])
        fn_delta = int(current_metrics["FN"] - baseline_metrics["FN"])
        fp_change = f"{fp_delta:+,} vs original" if fp_delta else "unchanged vs original"
        fn_change = f"{fn_delta:+,} vs original" if fn_delta else "unchanged vs original"
        cards = [
            metric_card("FP at selected threshold", f"{current_metrics['FP']:,}", fp_change),
            metric_card("FN at selected threshold", f"{current_metrics['FN']:,}", fn_change),
            metric_card("Precision", f"{current_metrics['precision']:.3f}" if np.isfinite(current_metrics["precision"]) else "—"),
            metric_card("Recall", f"{current_metrics['recall']:.3f}" if np.isfinite(current_metrics["recall"]) else "—"),
            metric_card("False-positive rate", f"{current_metrics['fpr']:.3f}" if np.isfinite(current_metrics["fpr"]) else "—"),
        ]

        metric_rows = [
            {"metric": name, "calibrated_threshold": baseline_metrics[key],
             "selected_threshold": current_metrics[key]}
            for name, key in [
                ("Threshold", "threshold"), ("TP", "TP"), ("TN", "TN"),
                ("FP", "FP"), ("FN", "FN"), ("Precision", "precision"),
                ("Recall", "recall"), ("FPR", "fpr"), ("F1", "f1"),
                ("Predicted positives", "predicted_positive"),
            ]
        ]
        for row in metric_rows:
            if row["metric"] == "Threshold":
                row["calibrated_threshold"] = original_threshold
                row["selected_threshold"] = selected_threshold
            for key in ("calibrated_threshold", "selected_threshold"):
                value = row[key]
                if isinstance(value, (float, np.floating)) and np.isfinite(value):
                    row[key] = round(float(value), 4)

        sweep = threshold_sweep(sub)
        curve = px.line(
            sweep, x="threshold", y=["precision", "recall", "fpr"],
            title="Precision, recall and false-positive rate across thresholds",
            labels={"value": "Metric value", "variable": "Metric"},
        )
        curve.add_vline(x=original_threshold, line_dash="dash", line_color="gray",
                        annotation_text="calibration-selected threshold")
        curve.add_vline(x=selected_threshold, line_dash="dot", line_color="#ff7f0e",
                        annotation_text="what-if threshold")
        curve.update_yaxes(range=[0, 1])
        curve.update_layout(height=480, hovermode="x unified")
        columns = [{"name": c, "id": c} for c in ["metric", "calibrated_threshold", "selected_threshold"]]
        return cards, metric_rows, columns, curve

    # -- Overview ----------------------------------------------------------- #
    @app.callback(Output("cm", "figure"), Output("hist", "figure"), Input("label", "value"))
    def overview(label):
        sub = cases.loc[cases["label"] == label]
        n = sub["outcome"].value_counts().reindex(OUTCOMES, fill_value=0)
        z = [[n["TP"], n["FN"]], [n["FP"], n["TN"]]]
        cm = go.Figure(go.Heatmap(
            z=z, x=["Predicted 1", "Predicted 0"], y=["Actual 1", "Actual 0"],
            text=[[f"TP\n{z[0][0]:,}", f"FN\n{z[0][1]:,}"],
                  [f"FP\n{z[1][0]:,}", f"TN\n{z[1][1]:,}"]],
            texttemplate="%{text}", colorscale="Blues", showscale=False,
        ))
        cm.update_layout(title=f"Confusion matrix: {label}", yaxis={"autorange": "reversed"})
        hist = px.histogram(
            sub, x="probability", color="outcome", nbins=50, barmode="overlay",
            opacity=0.65, color_discrete_map=COLORS, log_y=True,
            category_orders={"outcome": list(OUTCOMES)},
            title="Predicted probability by outcome (log count)",
        )
        hist.add_vline(x=float(sub["threshold"].iloc[0]), line_dash="dash",
                       annotation_text="threshold")
        return cm, hist

    # -- Feature comparison ------------------------------------------------- #
    @app.callback(Output("feat-bar", "figure"), Input("label", "value"),
                  Input("feat-outcome", "value"), Input("feat-top", "value"))
    def feature_bar(label, outcome, top):
        table = compare_features(cases, label, outcome, numeric_features, top)
        if table is None:
            return empty_figure(f"No {outcome} (or reference) cases for {label}")
        fig = px.bar(
            table.reset_index(), x="std_diff", y="feature", orientation="h",
            title=f"{outcome} vs {REFERENCE_OUTCOME[outcome]}: standardized mean difference",
            color_discrete_sequence=[COLORS[outcome]],
        )
        fig.update_layout(yaxis={"autorange": "reversed"}, height=max(350, 24 * len(table) + 120))
        return fig

    @app.callback(Output("feat-dist", "figure"), Input("feat-bar", "clickData"),
                  Input("label", "value"), Input("feat-outcome", "value"))
    def feature_dist(click, label, outcome):
        if ctx.triggered_id in ("label", "feat-outcome") or not click:
            top = compare_features(cases, label, outcome, numeric_features, 1)
            if top is None:
                return empty_figure("No data")
            feature = top.index[0]
        else:
            feature = click["points"][0]["y"]
        ref = REFERENCE_OUTCOME[outcome]
        sub = cases.loc[(cases["label"] == label) & cases["outcome"].isin([outcome, ref])]
        fig = px.histogram(
            sub, x=feature, color="outcome", barmode="overlay", opacity=0.6, nbins=50,
            histnorm="probability density", marginal="box", color_discrete_map=COLORS,
            title=f"{feature}: {outcome} vs {ref}",
        )
        return fig

    # -- Over time ---------------------------------------------------------- #
    @app.callback(Output("time-fig", "figure"), Input("label", "value"))
    def over_time(label):
        sub = cases.loc[cases["label"] == label]
        counts = pd.crosstab(sub["cutoff"], sub["outcome"]).reindex(columns=list(OUTCOMES), fill_value=0)
        total = counts.sum(axis=1).replace(0, np.nan)
        fp_rate = counts["FP"] / (counts["FP"] + counts["TN"]).replace(0, np.nan)
        fn_rate = counts["FN"] / (counts["FN"] + counts["TP"]).replace(0, np.nan)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for outcome in ("FP", "FN"):
            fig.add_bar(x=counts.index, y=counts[outcome], name=outcome, marker_color=COLORS[outcome])
        fig.add_scatter(x=counts.index, y=fp_rate, name="FPR", mode="lines+markers",
                        line={"color": "#ff7f0e"}, secondary_y=True)
        fig.add_scatter(x=counts.index, y=fn_rate, name="FNR", mode="lines+markers",
                        line={"color": "#d62728", "dash": "dot"}, secondary_y=True)
        fig.update_layout(barmode="stack", title=f"FP/FN counts and class-specific error rates: {label}")
        fig.update_yaxes(title_text="error cases", secondary_y=False)
        fig.update_yaxes(title_text="rate", range=[0, 1], secondary_y=True)
        return fig

    # -- Users -------------------------------------------------------------- #
    @app.callback(Output("users-fig", "figure"), Input("label", "value"), Input("users-n", "value"))
    def users(label, n):
        sub = cases.loc[(cases["label"] == label) & (~cases["correct"])]
        if sub.empty:
            return empty_figure("No errors for this label")
        sub = sub.assign(user=sub["github_user_id"].fillna(sub["user_id"]).astype(str))
        counts = pd.crosstab(sub["user"], sub["outcome"]).reindex(columns=["FN", "FP"], fill_value=0)
        counts = counts.assign(total=counts.sum(axis=1)).sort_values("total", ascending=False).head(int(n))
        fig = px.bar(counts.drop(columns="total"), barmode="stack", color_discrete_map=COLORS,
                     title=f"Users with the most errors: {label}")
        fig.update_xaxes(type="category", title="user (GitHub user ID)")
        fig.update_yaxes(title="error cases")
        return fig

    # -- General case explorer --------------------------------------------- #
    @app.callback(Output("case-scatter", "figure"), Input("label", "value"),
                  Input("case-outcomes", "value"))
    def case_scatter(label, outcomes):
        sub = cases.loc[(cases["label"] == label) & cases["outcome"].isin(outcomes or [])]
        if sub.empty:
            return empty_figure("Select at least one outcome")
        errors = sub.loc[~sub["correct"]]
        correct = sub.loc[sub["correct"]]
        budget = max(MAX_POINTS - len(errors), 0)
        if len(correct) > budget:
            correct = correct.sample(budget, random_state=0)
        shown = pd.concat([errors, correct]).reset_index()
        fig = px.scatter(
            shown, x="cutoff", y="probability", color="outcome", custom_data=["case_id"],
            hover_data=["user_id", "github_user_id", "actual", "margin"],
            color_discrete_map=COLORS, opacity=0.7,
            title=f"Click a point to inspect it ({len(shown):,} of {len(sub):,} cases shown)",
        )
        fig.add_hline(y=float(sub["threshold"].iloc[0]), line_dash="dash", annotation_text="threshold")
        fig.update_layout(height=450)
        return fig

    @app.callback(
        Output("case-header", "children"), Output("case-contrib", "figure"),
        Output("case-table", "data"), Output("case-table", "columns"),
        Input("case-scatter", "clickData"), Input("case-id", "value"),
    )
    def case_detail(click, typed):
        if ctx.triggered_id == "case-id" and typed is not None:
            case_id = int(typed)
        elif click:
            case_id = int(click["points"][0]["customdata"][0])
        else:
            return "Click a point above (or enter a case_id) to inspect it.", empty_figure("Select a case"), [], []
        header, fig, data = render_case_detail(case_id)
        columns = [{"name": c, "id": c} for c in (data[0].keys() if data else [])]
        return header, fig, data, columns

    return app


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_lightgbm(
    db_dir: Path,
    output_dir: Path,
    start_cutoff: pd.Timestamp,
    model_params: dict[str, int | float],
    launch_dashboard: bool = True,
    port: int = 8050,
    export_all: bool = False,
) -> None:
    data, _, _ = baseline.build_training_data(db_dir, start_cutoff)
    cutoffs = [pd.Timestamp(value) for value in sorted(data["cutoff"].unique())]
    train_end, calibration_end = baseline.chronological_splits(cutoffs)
    feature_columns = [
        column for column in data.columns
        if column not in {"user_id", "cutoff", *baseline.LABELS}
    ]

    train = data.loc[data["cutoff"] < train_end]
    calibration = data.loc[(data["cutoff"] >= train_end) & (data["cutoff"] < calibration_end)]
    holdout = data.loc[data["cutoff"] >= calibration_end]
    print(
        f"Chronological split: train < {train_end.date()}, "
        f"calibrate < {calibration_end.date()}, holdout >= {calibration_end.date()}"
    )

    models = baseline.fit_models(train, feature_columns, model_params)
    thresholds = baseline.DEFAULT_THRESHOLDS.copy()
    holdout_probabilities = baseline.predict_probabilities(models, holdout, feature_columns)
    metrics = baseline.evaluate_predictions(holdout, holdout_probabilities, thresholds)
    metrics.update({
        "temporal_model": "lightgbm",
        "threshold_mode": "fixed_default",
        "model_params": model_params,
        "split": {
            "train_weeks": int(train["cutoff"].nunique()),
            "calibration_weeks": int(calibration["cutoff"].nunique()),
            "holdout_weeks": int(holdout["cutoff"].nunique()),
            "holdout_start": str(calibration_end),
        },
    })

    users_table = pd.read_parquet(
        db_dir / "Users.parquet", columns=["user_id", "github_user_id", "is_cohort"]
    )
    cohort = users_table.loc[users_table["is_cohort"]]
    github_user_ids = cohort.set_index("user_id")["github_user_id"]

    cases = build_case_table(holdout, holdout_probabilities, thresholds, github_user_ids, feature_columns)
    failure_cases = build_failure_cases(cases, feature_columns)
    fp_cases = build_fp_cases(cases, feature_columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = output_dir / "failure_cases.csv"
    fp_path = output_dir / "false_positive_cases.csv"
    metrics_path = output_dir / "backtest_metrics.json"
    failure_cases.to_csv(failure_path, index=False)
    fp_cases.to_csv(fp_path, index=False)
    if export_all:
        cases.reset_index().to_csv(output_dir / "all_cases.csv", index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved {len(failure_cases):,} failure rows to {failure_path}")
    print(f"Saved {len(fp_cases):,} false-positive rows to {fp_path}")
    print(f"Saved holdout metrics to {metrics_path}")

    if launch_dashboard:
        app = make_app(cases, models, feature_columns)
        url = f"http://127.0.0.1:{port}"
        print(f"\nDashboard running at {url}  (Ctrl+C to stop)")
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        app.run(host="127.0.0.1", port=port, debug=False)


def _saved_probability_branches(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    branches: dict[str, dict[str, str]] = {}
    for label in baseline.LABELS:
        if label in frame.columns and not label.startswith("actual_"):
            branches.setdefault("model", {})[label] = label
        suffix = f"_{label}"
        for column in frame.columns:
            if column.startswith("actual_"):
                continue
            if column.endswith(suffix):
                branch = column[: -len(suffix)]
                branches.setdefault(branch, {})[label] = column
    complete = {
        branch: columns
        for branch, columns in branches.items()
        if set(columns) == set(baseline.LABELS)
    }
    if not complete:
        raise ValueError(
            "No complete six-label probability branch was found in "
            "holdout_probabilities.csv."
        )
    return complete


def _saved_branch_thresholds(
    metrics: dict, selected_branch: str
) -> dict[str, float]:
    branch_metrics = metrics.get("branch_holdout", {}).get(selected_branch, {})
    metric_container = (
        branch_metrics if branch_metrics else metrics.get("holdout_metrics", metrics)
    )
    thresholds = metric_container.get("thresholds", metrics.get("thresholds"))
    if not isinstance(thresholds, dict) or set(thresholds) != set(baseline.LABELS):
        raise ValueError(
            f"No complete calibration threshold set is available for branch "
            f"{selected_branch!r}."
        )
    result = {label: float(thresholds[label]) for label in baseline.LABELS}
    for label, threshold in result.items():
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"Invalid threshold for {label}: {threshold}.")
    return result


def _match_cached_labels(
    predictions: pd.DataFrame,
    actual: pd.DataFrame,
    keys: list[str],
    run_dir: Path,
) -> pd.DataFrame:
    if predictions.duplicated(keys).any():
        raise ValueError("Saved holdout predictions have duplicate user-week rows.")
    if actual.duplicated(keys).any():
        raise ValueError(f"Cached examples contain duplicate user-week rows in {run_dir}.")
    holdout = predictions.merge(
        actual,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = holdout["_merge"] != "both"
    if unmatched.any():
        raise ValueError(
            f"{int(unmatched.sum())} saved prediction row(s) could not be matched "
            "to labels in the run's graph-example cache."
        )
    return holdout.drop(columns="_merge")


def load_saved_experiment_cases(
    run_dir: Path,
    db_dir: Path,
    model_branch: str | None,
) -> tuple[pd.DataFrame, dict[str, float], list[str], str]:
    """Load saved holdout predictions and labels from an experiment run."""
    probability_path = run_dir / "holdout_probabilities.csv"
    metrics_path = run_dir / "backtest_metrics.json"
    cache_path = run_dir / "cache" / "weekly_graph_examples.npz"
    missing = [
        path.name for path in (probability_path, metrics_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{run_dir} is missing required saved-run file(s): {', '.join(missing)}"
        )

    predictions = pd.read_csv(probability_path)
    required_prediction_columns = {"user_id", "cutoff"}
    if not required_prediction_columns.issubset(predictions.columns):
        raise ValueError(
            "holdout_probabilities.csv must contain user_id and cutoff columns."
        )
    predictions["cutoff"] = pd.to_datetime(predictions["cutoff"]).dt.normalize()
    branches = _saved_probability_branches(predictions)
    if model_branch is None:
        selected_branch = next(
            (
                preferred for preferred in PREDICTION_BRANCHES
                if preferred in branches
            ),
            next(iter(branches)),
        )
    else:
        selected_branch = model_branch
    if selected_branch not in branches:
        raise ValueError(
            f"Probability branch {selected_branch!r} is unavailable. "
            f"Available branches: {', '.join(sorted(branches))}."
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    normalized_thresholds = _saved_branch_thresholds(metrics, selected_branch)
    for branch, columns in branches.items():
        values = predictions[list(columns.values())].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(
                f"Branch {branch!r} contains non-finite or out-of-range probabilities."
            )

    actual_columns = [f"actual_{label}" for label in baseline.LABELS]
    if set(actual_columns).issubset(predictions.columns):
        actual = predictions[["user_id", "cutoff", *actual_columns]].copy()
    elif cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            cache_user_ids = cache["user_ids"]
            cache_cutoff_days = cache["cutoff_days"]
            cache_labels = cache["labels"]
        if cache_labels.shape != (len(cache_user_ids), len(baseline.LABELS)):
            raise ValueError(f"Cached labels have an invalid shape in {cache_path}.")
        actual = pd.DataFrame(
            cache_labels,
            columns=actual_columns,
        )
        actual.insert(0, "cutoff", pd.to_datetime(cache_cutoff_days, unit="D"))
        actual.insert(0, "user_id", cache_user_ids)
    else:
        raise FileNotFoundError(
            f"{probability_path} does not include actual_<label> columns and "
            f"the required label cache is missing: {cache_path}"
        )
    if actual.duplicated(["user_id", "cutoff"]).any():
        raise ValueError(
            f"Actual labels contain duplicate user-week rows in {probability_path}."
        )

    prediction_rows = predictions.drop(columns=actual_columns, errors="ignore")
    holdout = _match_cached_labels(
        prediction_rows, actual, ["user_id", "cutoff"], run_dir
    )
    saved_probability_columns = {
        column for columns in branches.values() for column in columns.values()
    }
    normalized_probability_columns: set[str] = set()
    for branch, columns in branches.items():
        for label, probability_column in columns.items():
            normalized_column = f"prediction_{branch}_{label}"
            holdout[normalized_column] = holdout[probability_column].astype(float)
            columns[label] = normalized_column
            normalized_probability_columns.add(normalized_column)
    for label in baseline.LABELS:
        holdout[label] = holdout[f"actual_{label}"].astype(np.int8)

    score_features: list[str] = []
    probability_map: dict[str, np.ndarray] = {}
    for label in baseline.LABELS:
        probability_map[label] = holdout[branches[selected_branch][label]].to_numpy(
            dtype=float
        )
    for branch, columns in branches.items():
        for label in baseline.LABELS:
            feature_name = f"score_{branch}_{label}"
            holdout[feature_name] = holdout[columns[label]].astype(float)
            score_features.append(feature_name)

    excluded_columns = {
        "user_id",
        "cutoff",
        *baseline.LABELS,
        *actual_columns,
        *saved_probability_columns,
        *normalized_probability_columns,
    }
    experiment_features = [
        column
        for column in holdout.columns
        if column not in excluded_columns
        and column not in score_features
        and pd.api.types.is_numeric_dtype(holdout[column])
    ]
    feature_columns = [*experiment_features, *score_features]

    users = pd.read_parquet(
        db_dir / "Users.parquet",
        columns=["user_id", "github_user_id", "is_cohort"],
    )
    cohort = users.loc[users["is_cohort"]]
    github_user_ids = cohort.set_index("user_id")["github_user_id"]
    cases = build_case_table(
        holdout,
        probability_map,
        normalized_thresholds,
        github_user_ids,
        feature_columns,
    )
    experiment_name = str(
        metrics.get("temporal_model")
        or metrics.get("experiment")
        or run_dir.parent.name
    )
    return (
        cases,
        normalized_thresholds,
        feature_columns,
        f"{experiment_name} / {selected_branch}",
    )


def run_saved_experiment(
    run_dir: Path,
    db_dir: Path,
    output_dir: Path,
    model_branch: str | None = None,
    launch_dashboard: bool = True,
    port: int = 8050,
    export_all: bool = False,
) -> None:
    cases, thresholds, feature_columns, experiment_name = load_saved_experiment_cases(
        run_dir, db_dir, model_branch
    )
    failures = build_failure_cases(cases, feature_columns)
    false_positives = build_fp_cases(cases, feature_columns)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures.to_csv(output_dir / "failure_cases.csv", index=False)
    false_positives.to_csv(output_dir / "false_positive_cases.csv", index=False)
    if export_all:
        cases.reset_index().to_csv(output_dir / "all_cases.csv", index=False)
    metrics = {
        "source_run": str(run_dir.resolve()),
        "experiment": experiment_name,
        "thresholds": thresholds,
        "holdout_cases": int(len(cases)),
        "failure_cases": int(len(failures)),
        "false_positive_cases": int(len(false_positives)),
        "confusion_summary": confusion_summary(cases).reset_index().to_dict(
            orient="records"
        ),
    }
    (output_dir / "failure_analysis_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, default=str))
    print(f"Saved failure cases to {output_dir / 'failure_cases.csv'}")
    print(f"Saved false-positive cases to {output_dir / 'false_positive_cases.csv'}")

    if launch_dashboard:
        app = make_app(cases, {}, feature_columns, experiment_name)
        url = f"http://127.0.0.1:{port}"
        print(f"\nDashboard running at {url}  (Ctrl+C to stop)")
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        app.run(host="127.0.0.1", port=port, debug=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, default=baseline.DEFAULT_DB_DIR,
                        help="directory containing the assignment Parquet tables")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "an existing experiment backtest folder containing "
            "holdout_probabilities.csv and backtest_metrics.json; RelGT/GRU "
            "runs also use their graph-example cache"
        ),
    )
    parser.add_argument(
        "--model-branch",
        default=None,
        help=(
            "probability branch to analyze (for example fused, relgt, lightgbm, "
            "s2p2, or model); defaults to fused when available"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="analysis output folder (default: timestamped or under --run-dir)")
    parser.add_argument("--start-cutoff", type=pd.Timestamp,
                        default=pd.Timestamp(baseline.DEFAULT_START_CUTOFF),
                        help="first weekly prediction cutoff used to construct training samples")
    parser.add_argument("--port", type=int, default=8050, help="dashboard port")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="only train and export files, do not start the dashboard")
    parser.add_argument("--export-all", action="store_true",
                        help="also save every holdout case (correct and wrong) to all_cases.csv")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--min-child-samples", type=int, default=40)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    args = parser.parse_args()
    if args.model_branch is not None and args.run_dir is None:
        parser.error("--model-branch requires --run-dir.")
    return args


def main() -> None:
    args = parse_args()
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        output_dir = args.output_dir or run_dir / "failure_analysis"
        run_saved_experiment(
            run_dir,
            args.db_dir,
            output_dir,
            model_branch=args.model_branch,
            launch_dashboard=not args.no_dashboard,
            port=args.port,
            export_all=args.export_all,
        )
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR / f"failure_analysis_{timestamp}"
    model_params: dict[str, int | float] = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "subsample_freq": 1,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
    }
    run_lightgbm(
        args.db_dir, output_dir, args.start_cutoff, model_params,
        launch_dashboard=not args.no_dashboard, port=args.port, export_all=args.export_all,
    )


if __name__ == "__main__":
    main()
