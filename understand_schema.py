"""Explore the Assignment 2 Parquet tables and their relationships.

Run from the project root:
    python understand_schema.py

Optionally point to another copy of the database:
    python understand_schema.py --db-dir /path/to/assignment2/data/db

Choose where to save the report:
    python understand_schema.py --output schema_summary.txt
"""

import argparse
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pandas as pd


TABLES = (
    "Users",
    "Repositories",
    "PullRequests",
    "Issues",
    "IssueComments",
    "PRReviews",
    "PRReviewComments",
    "Pushes",
)

ROW_TIME_COLUMNS = {
    "Users": "first_seen_at",
    "Repositories": "first_seen_at",
    "PullRequests": "first_seen_at",
    "Issues": "first_seen_at",
    "IssueComments": "created_at",
    "PRReviews": "first_seen_at",
    "PRReviewComments": "created_at",
    "Pushes": "created_at",
}


def print_report(tables: dict[str, pd.DataFrame]) -> None:
    print("TABLE OVERVIEW")
    for name, frame in tables.items():
        time_column = ROW_TIME_COLUMNS[name]
        times = pd.to_datetime(frame[time_column], errors="coerce")
        date_range = f"; {time_column}: {times.min()} .. {times.max()}"
        print(f"- {name:18} {len(frame):>10,} rows, {len(frame.columns):>2} columns{date_range}")
        print("  " + ", ".join(frame.columns))

    users = tables["Users"]
    cohort_users = users.loc[users["is_cohort"]]
    print("\nUSERS")
    print(f"Cohort users with complete activity: {len(cohort_users):,}")
    print(f"Other referenced users (incomplete activity): {len(users) - len(cohort_users):,}")
    print("Join key inside this data release: user_id")
    print("Stable GitHub identity across releases: github_user_id")

    # Foreign keys are row numbers in this release, so resolve them by joining
    # on user_id/repo_id rather than github_user_id/repo_id.
    pushes = tables["Pushes"]
    repositories = tables["Repositories"]
    joined_pushes = (
        pushes.merge(
            users[["user_id", "github_user_id", "login", "is_cohort"]],
            on="user_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            repositories[["repo_id", "name"]],
            on="repo_id",
            how="left",
            validate="many_to_one",
        )
    )
    print("\nEXAMPLE JOIN: PUSH -> USER AND REPOSITORY")
    print(
        joined_pushes.loc[
            joined_pushes["is_cohort"],
            ["github_user_id", "login", "name", "branch", "created_at"],
        ]
        .head(5)
        .to_string(index=False)
    )

    if not pushes.empty:
        latest_push = pd.to_datetime(pushes["created_at"]).max()
        start = latest_push - pd.Timedelta(days=30)
        recent_cohort_pushes = joined_pushes.loc[
            joined_pushes["is_cohort"]
            & joined_pushes["created_at"].ge(start)
            & joined_pushes["created_at"].le(latest_push)
        ]
        top_users = (
            recent_cohort_pushes.groupby(["github_user_id", "login"])
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        print(f"\nTOP COHORT PUSHERS IN THE LAST 30 DAYS OF PUSH DATA (since {start})")
        print(top_users.rename("pushes").to_string() if not top_users.empty else "No pushes")

    issues = tables["Issues"]
    is_pr_conversation = issues["is_pull_request"].fillna(False)
    print("\nISSUES TABLE")
    print(f"Ordinary issues: {int((~is_pr_conversation).sum()):,}")
    print(f"PR conversations: {int(is_pr_conversation.sum()):,}")
    print("For an issue-opening label, exclude rows where is_pull_request is true.")

    print("\nTIME / LEAKAGE REMINDER")
    print(
        "For a historical snapshot at time t, filter rows by the documented row-time: "
        "first_seen_at for PullRequests, Issues, and PRReviews; created_at for "
        "IssueComments, PRReviewComments, and Pushes. Do not treat old created_at/"
        "submitted_at values as proof that a row was already available."
    )
    print(
        "Fields such as merged_at, closed_at, and dismissed_at are known only later; "
        "use them only when their value is earlier than t."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_db_dir = Path(__file__).resolve().parent / "assignment2" / "data" / "db"
    parser.add_argument("--db-dir", type=Path, default=default_db_dir)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("schema_summary.txt"),
        help="report file path (default: schema_summary.txt next to this script)",
    )
    args = parser.parse_args()

    missing = [name for name in TABLES if not (args.db_dir / f"{name}.parquet").is_file()]
    if missing:
        parser.error(
            f"Missing parquet file(s) in {args.db_dir}: "
            + ", ".join(f"{name}.parquet" for name in missing)
        )

    tables = {
        name: pd.read_parquet(args.db_dir / f"{name}.parquet")
        for name in TABLES
    }

    report_buffer = StringIO()
    with redirect_stdout(report_buffer):
        print_report(tables)
    report = report_buffer.getvalue()
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"\nSaved report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
