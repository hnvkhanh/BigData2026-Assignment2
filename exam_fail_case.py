"""Run the LightGBM holdout and explore correct / misclassified cases in a Dash dashboard.

Install once:  pip install dash plotly
Run:           python lightgbm_failure_dashboard.py   (opens http://127.0.0.1:8050)
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
from dash import Dash, Input, Output, ctx, dash_table, dcc, html
from plotly.subplots import make_subplots

from experiments.lightgbm_baseline import run as baseline


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "lightgbm_baseline" / "outputs"

OUTCOMES = ("TP", "TN", "FP", "FN")
REFERENCE_OUTCOME = {"FN": "TP", "FP": "TN"}  # compare errors with correct cases of the same true class
FAILURE_TYPE = {"FN": "false_negative", "FP": "false_positive"}
COLORS = {"TP": "#2ca02c", "TN": "#9ecae1", "FP": "#ff7f0e", "FN": "#d62728"}
MAX_POINTS = 15000  # scatter cap per label (all errors are always kept)


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
    """Return one row per user-week-label (correct AND incorrect) with its features."""
    frames: list[pd.DataFrame] = []
    for label in baseline.LABELS:
        actual = holdout[label].to_numpy(dtype=np.int8)
        probability = probabilities[label]
        threshold = thresholds[label]
        predicted = (probability >= threshold).astype(np.int8)

        rows = holdout[["user_id", "cutoff", *feature_columns]].copy()
        rows.insert(1, "github_user_id", rows["user_id"].map(github_user_ids))
        rows["label"] = label
        rows["actual"] = actual
        rows["predicted"] = predicted
        rows["probability"] = probability
        rows["threshold"] = threshold
        rows["outcome"] = np.select(
            [(actual == 1) & (predicted == 1), (actual == 0) & (predicted == 0),
             (actual == 0) & (predicted == 1)],
            ["TP", "TN", "FP"],
            default="FN",
        )
        rows["correct"] = actual == predicted
        frames.append(rows)

    cases = pd.concat(frames, ignore_index=True)
    cases.index.name = "case_id"
    return cases


def build_failure_cases(cases: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return only the misclassified rows (same layout as the original export)."""
    failures = cases.loc[~cases["correct"]].copy()
    failures["failure_type"] = failures["outcome"].map(FAILURE_TYPE)
    columns = [
        "user_id", "github_user_id", "cutoff", "label", "actual", "predicted",
        "probability", "threshold", "failure_type", *feature_columns,
    ]
    return failures[columns].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def confusion_summary(cases: pd.DataFrame) -> pd.DataFrame:
    counts = (
        cases.groupby(["label", "outcome"]).size().unstack(fill_value=0)
        .reindex(columns=list(OUTCOMES), fill_value=0)
    )
    total = counts.sum(axis=1)
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
    """Standardised mean difference: error group vs correct cases with the same true class."""
    sub = cases.loc[cases["label"] == label]
    bad = sub.loc[sub["outcome"] == outcome, feature_columns]
    ok = sub.loc[sub["outcome"] == REFERENCE_OUTCOME[outcome], feature_columns]
    if bad.empty or ok.empty:
        return None
    pooled_std = pd.concat([bad, ok]).std(ddof=0).replace(0, np.nan)
    smd = ((bad.mean() - ok.mean()) / pooled_std).dropna()
    order = smd.abs().sort_values(ascending=False).head(top).index
    return pd.DataFrame({"std_diff": smd[order]}).rename_axis("feature")


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=message, xaxis={"visible": False}, yaxis={"visible": False}, height=300
    )
    return fig


# --------------------------------------------------------------------------- #
# Dash app
# --------------------------------------------------------------------------- #
def make_app(cases: pd.DataFrame, models: dict, feature_columns: list[str]) -> Dash:
    labels = list(baseline.LABELS)
    summary = confusion_summary(cases).reset_index()

    def contributions(row: pd.Series) -> pd.Series | None:
        """Per-feature LightGBM contributions (SHAP-style) toward the positive class."""
        model = models.get(row["label"])
        x = row[feature_columns].to_frame().T.astype(float)
        for predictor in (model, getattr(model, "booster_", None)):
            if predictor is None:
                continue
            try:
                values = np.asarray(predictor.predict(x, pred_contrib=True)).reshape(-1)
                return pd.Series(values[: len(feature_columns)], index=feature_columns)
            except Exception:
                continue
        return None

    app = Dash(__name__, title="LightGBM failure analysis", suppress_callback_exceptions=True)

    app.layout = html.Div(
        style={"fontFamily": "sans-serif", "margin": "16px 24px"},
        children=[
            html.H2("LightGBM holdout: correct vs failed cases"),
            html.Div(
                [
                    html.Label("Label  "),
                    dcc.Dropdown(labels, labels[0], id="label", clearable=False,
                                 style={"width": "320px"}),
                ],
                style={"marginBottom": "12px"},
            ),
            dcc.Tabs(
                id="tabs",
                value="overview",
                children=[
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
                                    "FN", id="feat-outcome", inline=True),
                            ]),
                            html.Div([
                                html.Label("Top features"),
                                dcc.Slider(5, 40, 5, value=15, id="feat-top",
                                           marks=None, tooltip={"always_visible": True}),
                            ], style={"width": "320px"}),
                        ]),
                        html.Div("Click a bar to see that feature's distribution.",
                                 style={"color": "#666"}),
                        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                            dcc.Graph(id="feat-bar", style={"width": "48%", "minWidth": "400px"}),
                            dcc.Graph(id="feat-dist", style={"width": "50%", "minWidth": "400px"}),
                        ]),
                    ]),
                    dcc.Tab(label="Over time", value="time", children=[
                        dcc.Graph(id="time-fig"),
                    ]),
                    dcc.Tab(label="Users", value="users", children=[
                        html.Div([
                            html.Label("Users shown"),
                            dcc.Slider(5, 50, 5, value=20, id="users-n",
                                       marks=None, tooltip={"always_visible": True}),
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
                                    ["FP", "FN"], id="case-outcomes", inline=True),
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
                            html.Div(style={"width": "42%", "minWidth": "320px"}, children=[
                                dash_table.DataTable(
                                    id="case-table",
                                    columns=[{"name": c, "id": c}
                                             for c in ("feature", "value", "correct_median")],
                                    style_cell={"textAlign": "right", "padding": "4px 10px"},
                                    style_header={"fontWeight": "bold"},
                                ),
                            ]),
                        ]),
                    ]),
                ],
            ),
        ],
    )

    # -- Overview ----------------------------------------------------------- #
    @app.callback(Output("cm", "figure"), Output("hist", "figure"), Input("label", "value"))
    def overview(label):
        sub = cases.loc[cases["label"] == label]
        n = sub["outcome"].value_counts().reindex(OUTCOMES, fill_value=0)
        z = [[n["TP"], n["FN"]], [n["FP"], n["TN"]]]
        cm = go.Figure(go.Heatmap(
            z=z, x=["Predicted 1", "Predicted 0"], y=["Actual 1", "Actual 0"],
            text=[[f"TP\n{z[0][0]:,}", f"FN\n{z[0][1]:,}"], [f"FP\n{z[1][0]:,}", f"TN\n{z[1][1]:,}"]],
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

    # -- Features ----------------------------------------------------------- #
    @app.callback(Output("feat-bar", "figure"), Input("label", "value"),
                  Input("feat-outcome", "value"), Input("feat-top", "value"))
    def feature_bar(label, outcome, top):
        table = compare_features(cases, label, outcome, feature_columns, top)
        if table is None:
            return empty_figure(f"No {outcome} (or reference) cases for {label}")
        fig = px.bar(
            table.reset_index(), x="std_diff", y="feature", orientation="h",
            title=f"{outcome} vs {REFERENCE_OUTCOME[outcome]}: standardised mean difference",
            color_discrete_sequence=[COLORS[outcome]],
        )
        fig.update_layout(yaxis={"autorange": "reversed"}, height=max(350, 24 * len(table) + 120))
        return fig

    @app.callback(Output("feat-dist", "figure"), Input("feat-bar", "clickData"),
                  Input("label", "value"), Input("feat-outcome", "value"))
    def feature_dist(click, label, outcome):
        if ctx.triggered_id in ("label", "feat-outcome") or not click:
            top = compare_features(cases, label, outcome, feature_columns, 1)
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
        counts = (pd.crosstab(sub["cutoff"], sub["outcome"])
                  .reindex(columns=list(OUTCOMES), fill_value=0))
        rate = (counts["FP"] + counts["FN"]) / counts.sum(axis=1)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for outcome in ("FN", "FP"):
            fig.add_bar(x=counts.index, y=counts[outcome], name=outcome,
                        marker_color=COLORS[outcome])
        fig.add_scatter(x=counts.index, y=rate, name="error rate", mode="lines+markers",
                        line={"color": "black"}, secondary_y=True)
        fig.update_layout(barmode="stack", title=f"Errors per weekly cutoff: {label}")
        fig.update_yaxes(title_text="errors", secondary_y=False)
        fig.update_yaxes(title_text="error rate", rangemode="tozero", secondary_y=True)
        return fig

    # -- Users -------------------------------------------------------------- #
    @app.callback(Output("users-fig", "figure"), Input("label", "value"), Input("users-n", "value"))
    def users(label, n):
        sub = cases.loc[(cases["label"] == label) & (~cases["correct"])]
        if sub.empty:
            return empty_figure("No errors for this label")
        sub = sub.assign(user=sub["github_user_id"].fillna(sub["user_id"]).astype(str))
        counts = pd.crosstab(sub["user"], sub["outcome"]).reindex(columns=["FN", "FP"], fill_value=0)
        counts = counts.assign(total=counts.sum(axis=1)).sort_values("total", ascending=False).head(n)
        fig = px.bar(counts.drop(columns="total"), barmode="stack",
                     color_discrete_map=COLORS,
                     title=f"Users with the most errors: {label}")
        fig.update_xaxes(type="category", title="user (github_user_id)")
        fig.update_yaxes(title="errors")
        return fig

    # -- Case explorer ------------------------------------------------------ #
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
            shown, x="cutoff", y="probability", color="outcome",
            color_discrete_map=COLORS, custom_data=["case_id"],
            hover_data=["user_id", "github_user_id", "actual"], opacity=0.7,
            title=f"Click a point to inspect it ({len(shown):,} of {len(sub):,} cases shown)",
        )
        fig.add_hline(y=float(sub["threshold"].iloc[0]), line_dash="dash",
                      annotation_text="threshold")
        fig.update_layout(height=450)
        return fig

    @app.callback(
        Output("case-header", "children"),
        Output("case-contrib", "figure"),
        Output("case-table", "data"),
        Input("case-scatter", "clickData"),
        Input("case-id", "value"),
    )
    def case_detail(click, typed):
        if ctx.triggered_id == "case-id" and typed is not None:
            case_id = int(typed)
        elif click:
            case_id = int(click["points"][0]["customdata"][0])
        else:
            return "Click a point above (or enter a case_id) to inspect it.", empty_figure(""), []
        if case_id not in cases.index:
            return f"case_id {case_id} not found.", empty_figure(""), []

        row = cases.loc[case_id]
        header = (
            f"case {case_id} | {row['label']} | user {row['user_id']} (github {row['github_user_id']}) "
            f"| {pd.Timestamp(row['cutoff']).date()} | actual={row['actual']} predicted={row['predicted']} "
            f"→ {row['outcome']} | p={row['probability']:.3f} vs threshold {row['threshold']:.3f}"
        )
        same_class = cases.loc[
            (cases["label"] == row["label"]) & (cases["actual"] == row["actual"]) & cases["correct"]
        ]
        median = same_class[feature_columns].median()
        values = row[feature_columns].astype(float)

        contrib = contributions(row)
        if contrib is not None:
            order = contrib.abs().sort_values(ascending=False).head(12).index
            plot = pd.DataFrame({"feature": order, "score": contrib[order].to_numpy()})
            title = "Top contributions (+ toward positive class, − toward negative)"
        else:
            std = same_class[feature_columns].std(ddof=0).replace(0, np.nan)
            z = ((values - median) / std).dropna()
            order = z.abs().sort_values(ascending=False).head(12).index
            plot = pd.DataFrame({"feature": order, "score": z[order].to_numpy()})
            title = "Most unusual features (z-score vs correct cases of same class)"
        fig = px.bar(plot, x="score", y="feature", orientation="h", title=title,
                     color=np.where(plot["score"] >= 0, "positive", "negative"),
                     color_discrete_map={"positive": "#d62728", "negative": "#1f77b4"})
        fig.update_layout(yaxis={"autorange": "reversed"}, showlegend=False)

        table = pd.DataFrame({
            "feature": order,
            "value": values[order].round(3).to_numpy(),
            "correct_median": median[order].round(3).to_numpy(),
        })
        return header, fig, table.to_dict("records")

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
        column
        for column in data.columns
        if column not in {"user_id", "cutoff", *baseline.LABELS}
    ]

    train = data.loc[data["cutoff"] < train_end]
    calibration = data.loc[
        (data["cutoff"] >= train_end) & (data["cutoff"] < calibration_end)
    ]
    holdout = data.loc[data["cutoff"] >= calibration_end]
    print(
        f"Chronological split: train < {train_end.date()}, "
        f"calibrate < {calibration_end.date()}, "
        f"holdout >= {calibration_end.date()}"
    )

    models = baseline.fit_models(train, feature_columns, model_params)
    calibration_probabilities = baseline.predict_probabilities(
        models, calibration, feature_columns
    )
    thresholds = {
        label: baseline.choose_threshold(
            calibration[label].to_numpy(dtype=np.int8),
            calibration_probabilities[label],
        )
        for label in baseline.LABELS
    }
    holdout_probabilities = baseline.predict_probabilities(
        models, holdout, feature_columns
    )
    metrics = baseline.evaluate_predictions(holdout, holdout_probabilities, thresholds)
    metrics.update(
        {
            "temporal_model": "lightgbm",
            "model_params": model_params,
            "split": {
                "train_weeks": int(train["cutoff"].nunique()),
                "calibration_weeks": int(calibration["cutoff"].nunique()),
                "holdout_weeks": int(holdout["cutoff"].nunique()),
                "holdout_start": str(calibration_end),
            },
        }
    )

    users = pd.read_parquet(
        db_dir / "Users.parquet",
        columns=["user_id", "github_user_id", "is_cohort"],
    )
    cohort = users.loc[users["is_cohort"]]
    github_user_ids = cohort.set_index("user_id")["github_user_id"]

    cases = build_case_table(
        holdout, holdout_probabilities, thresholds, github_user_ids, feature_columns
    )
    failure_cases = build_failure_cases(cases, feature_columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = output_dir / "failure_cases.csv"
    metrics_path = output_dir / "backtest_metrics.json"
    failure_cases.to_csv(failure_path, index=False)
    if export_all:
        cases.to_csv(output_dir / "all_cases.csv")
    metrics_path.write_text(
        json.dumps(metrics, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved {len(failure_cases):,} failure rows to {failure_path}")
    print(f"Saved holdout metrics to {metrics_path}")

    if launch_dashboard:
        app = make_app(cases, models, feature_columns)
        url = f"http://127.0.0.1:{port}"
        print(f"\nDashboard running at {url}  (Ctrl+C to stop)")
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        app.run(host="127.0.0.1", port=port, debug=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=baseline.DEFAULT_DB_DIR,
        help="directory containing the assignment Parquet tables",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output folder (default: a timestamped folder under LightGBM outputs)",
    )
    parser.add_argument(
        "--start-cutoff",
        type=pd.Timestamp,
        default=pd.Timestamp(baseline.DEFAULT_START_CUTOFF),
        help="first weekly prediction cutoff used to construct training samples",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        args.db_dir,
        output_dir,
        args.start_cutoff,
        model_params,
        launch_dashboard=not args.no_dashboard,
        port=args.port,
        export_all=args.export_all,
    )


if __name__ == "__main__":
    main()