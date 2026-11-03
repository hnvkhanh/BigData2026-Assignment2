"""Cross-action relational features for one user at one weekly cutoff.

Each feature encodes one hypothesis about how a user's next actions depend on
their relationship to repositories, PRs and issues (owner, author, assignee,
past reviewer) and on what other people recently did there. Every row used is
both acted and available strictly before the cutoff; lifecycle fields known
only later (``closed_at``) are used only when they are earlier than the cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.lightgbm_baseline.run import LABELS

# Branches treated as a repository's mainline, not a feature branch awaiting a PR.
MAINLINE_BRANCHES = frozenset(
    {"main", "master", "dev", "develop", "development", "gh-pages", "staging", "trunk"}
)
RECENT_DAYS = 7
CONTEXT_DAYS = 30
OPEN_ITEM_DAYS = 90
HABIT_WEEKS = 8
# "@login" not preceded by a word character, "@", "/" or "." (skips e-mails and URLs).
MENTION_PATTERN = r"(?<![\w@/.])@([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))"

# Feature group -> feature names. Groups can be switched off for ablations.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "owner": (
        "own_push_share_30d",
        "own_repos_pushed_30d",
        "own_repo_new_issues_by_others_7d",
        "own_repo_open_issues_by_others",
        "own_repo_issue_comments_by_others_7d",
        "own_repo_new_prs_by_others_7d",
        "own_repo_open_prs_by_others",
    ),
    "sequence": (
        "feature_branch_pushes_7d",
        "unpaired_branches_7d",
        "new_issues_on_pushed_repos_7d",
    ),
    "assigned": (
        "assigned_open_issues",
        "assigned_new_issues_7d",
    ),
    "reviewer": (
        "reviewed_repos_90d",
        "new_prs_on_reviewed_repos_7d",
        "open_prs_on_reviewed_repos",
        "open_prs_reviewed_by_user",
    ),
    "author": (
        "open_authored_prs",
        "open_authored_issues",
        "reviews_received_7d",
        "review_comments_received_7d",
        "issue_comments_received_7d",
    ),
    "repo_activity": (
        "active_repos_30d",
        "neighbor_new_prs_7d",
        "neighbor_new_issues_7d",
        "neighbor_issue_comments_7d",
        "neighbor_reviews_7d",
        "neighbor_review_comments_7d",
        "neighbor_pushes_7d",
    ),
    # Groups below were found by experiments/relational_lightgbm/explore.py.
    "mentions": (
        "mentions_received_7d",
        "mentions_received_30d",
    ),
    "threads": (
        "thread_new_comments_7d",
    ),
    "review_cycle": (
        "own_prs_merged_7d",
        "merged_others_prs_7d",
        "approvals_received_30d",
        "open_prs_with_recent_push",
    ),
    "habit": (
        *(f"{label}_active_weeks_8w" for label in LABELS),
        "any_active_weeks_8w",
    ),
}
# The groups of the first version (owner/author/reviewer/assignee/sequence hypotheses).
HYPOTHESIS_GROUPS = ("owner", "sequence", "assigned", "reviewer", "author", "repo_activity")

# Hypothesis -> (feature, label it should raise). Used for the lift report.
HYPOTHESES: tuple[tuple[str, str, str], ...] = (
    ("push to repos the user owns", "own_push_share_30d", "push"),
    ("push after an issue on a pushed repo", "new_issues_on_pushed_repos_7d", "push"),
    ("open PR after feature-branch pushes", "unpaired_branches_7d", "open_pr"),
    ("open PR after feature-branch pushes", "feature_branch_pushes_7d", "open_pr"),
    ("push after an issue on own repo", "own_repo_new_issues_by_others_7d", "push"),
    ("open PR after an issue on own repo", "own_repo_new_issues_by_others_7d", "open_pr"),
    ("past reviewer comments on new PRs", "new_prs_on_reviewed_repos_7d", "pr_review_comment"),
    ("past reviewer reviews new PRs", "new_prs_on_reviewed_repos_7d", "pr_review"),
    ("past reviewer reviews again on same repo", "reviewed_repos_90d", "pr_review"),
    ("PR author comments on own PR", "open_authored_prs", "issue_comment"),
    ("PR author answers review comments", "review_comments_received_7d", "pr_review_comment"),
    ("owner comments on issues of own repo", "own_repo_new_issues_by_others_7d", "issue_comment"),
    ("owner comments on issues of own repo", "own_repo_issue_comments_by_others_7d", "issue_comment"),
    ("assignee pushes", "assigned_open_issues", "push"),
    ("assignee opens PR", "assigned_open_issues", "open_pr"),
    ("mentioned users comment", "mentions_received_7d", "issue_comment"),
    ("mentioned users review", "mentions_received_7d", "pr_review"),
    ("thread participants comment again", "thread_new_comments_7d", "issue_comment"),
    ("authors of merged PRs open another", "own_prs_merged_7d", "open_pr"),
    ("mergers review", "merged_others_prs_7d", "pr_review"),
    ("approved authors keep reviewing and commenting", "approvals_received_30d", "pr_review_comment"),
    ("open PR still being pushed -> review comments", "open_prs_with_recent_push", "pr_review_comment"),
    *((f"weekly habit: {label}", f"{label}_active_weeks_8w", label) for label in LABELS),
)


@dataclass(frozen=True)
class RelationalTables:
    """Full (all users, not only cohort) tables needed by the features."""

    repo_owner: pd.Series  # repo_id -> owner user_id (only repos owned by a known user)
    pushes: pd.DataFrame  # user_id, repo_id, branch, at
    prs: pd.DataFrame  # pr_id, author_id, repo_id, head_ref, opened_at, available_at, closed_at, merged_at, merged_by_id
    issues: pd.DataFrame  # issue_id, author_id, assignee_id, repo_id, is_pr, opened_at, available_at, closed_at
    issue_comments: pd.DataFrame  # issue_id, repo_id, author_id, at
    reviews: pd.DataFrame  # pr_id, repo_id, reviewer_id, state, at, available_at
    review_comments: pd.DataFrame  # pr_id, repo_id, author_id, at
    mentions: pd.DataFrame  # user_id (mentioned), author_id, at, available_at, node_type, entity_id


def extract_mentions(bodies: pd.DataFrame, login_to_user: pd.Series) -> pd.DataFrame:
    """One row per "@login" of a known user in ``bodies`` (body, author_id, at, available_at).

    Self-mentions are dropped; a user mentioned twice in one text counts once.
    Any other columns of ``bodies`` (such as the PR/issue the text belongs to)
    are kept.
    """
    found = bodies.assign(login=bodies["body"].str.findall(MENTION_PATTERN))
    found = found.explode("login").dropna(subset=["login"])
    found["user_id"] = found["login"].str.lower().map(login_to_user)
    found = found.dropna(subset=["user_id"]).astype({"user_id": "int64"})
    found = found.loc[found["author_id"].isna() | (found["user_id"] != found["author_id"])]
    found = found.drop_duplicates(subset=["user_id", "author_id", "at", "available_at", "body"])
    extra = [column for column in bodies.columns if column not in {"body", "author_id", "at", "available_at"}]
    return found[["user_id", "author_id", "at", "available_at", *extra]].reset_index(drop=True)


def load_relational_tables(db_dir: Path) -> RelationalTables:
    users = pd.read_parquet(db_dir / "Users.parquet", columns=["user_id", "login"])
    repos = pd.read_parquet(db_dir / "Repositories.parquet", columns=["repo_id", "owner_login"])
    login_to_user = (
        users.dropna(subset=["login"])
        .assign(login=lambda frame: frame["login"].str.lower())
        .drop_duplicates("login", keep="last")
        .set_index("login")["user_id"]
    )
    owners = repos["owner_login"].str.lower().map(login_to_user)
    repo_owner = pd.Series(owners.to_numpy(), index=repos["repo_id"]).dropna().astype("int64")

    pushes = pd.read_parquet(
        db_dir / "Pushes.parquet",
        columns=["user_id", "repo_id", "branch", "created_at"],
    ).rename(columns={"created_at": "at"})

    prs = pd.read_parquet(
        db_dir / "PullRequests.parquet",
        columns=[
            "pr_id", "author_id", "repo_id", "head_ref", "created_at", "first_seen_at",
            "closed_at", "merged_at", "merged_by_id",
        ],
    )
    # A PR's opening is unseen when it predates the data window; first_seen_at
    # is then the earliest moment it is known to exist.
    prs = pd.DataFrame(
        {
            "pr_id": prs["pr_id"],
            "author_id": prs["author_id"].astype("Int64"),
            "repo_id": prs["repo_id"],
            "head_ref": prs["head_ref"],
            "opened_at": prs["created_at"].fillna(prs["first_seen_at"]),
            "available_at": prs["first_seen_at"],
            "closed_at": prs["closed_at"],
            "merged_at": prs["merged_at"],
            "merged_by_id": prs["merged_by_id"].astype("Int64"),
        }
    )

    issues = pd.read_parquet(
        db_dir / "Issues.parquet",
        columns=[
            "issue_id", "author_id", "assignee_id", "repo_id", "is_pull_request",
            "created_at", "first_seen_at", "closed_at", "body",
        ],
    )
    issue_bodies = pd.DataFrame(
        {
            "body": issues["body"],
            "author_id": issues["author_id"].astype("Int64"),
            "at": issues["created_at"].fillna(issues["first_seen_at"]),
            "available_at": issues["first_seen_at"],
            "node_type": 2,
            "entity_id": issues["issue_id"],
        }
    )
    issues = pd.DataFrame(
        {
            "issue_id": issues["issue_id"],
            "author_id": issues["author_id"].astype("Int64"),
            "assignee_id": issues["assignee_id"].astype("Int64"),
            "repo_id": issues["repo_id"],
            "is_pr": issues["is_pull_request"].fillna(False).astype(bool),
            "opened_at": issues["created_at"].fillna(issues["first_seen_at"]),
            "available_at": issues["first_seen_at"],
            "closed_at": issues["closed_at"],
        }
    )

    issue_comments = pd.read_parquet(
        db_dir / "IssueComments.parquet",
        columns=["issue_id", "repo_id", "author_id", "created_at", "body"],
    ).rename(columns={"created_at": "at"})
    reviews = pd.read_parquet(
        db_dir / "PRReviews.parquet",
        columns=["pr_id", "repo_id", "reviewer_id", "state", "submitted_at", "first_seen_at", "body"],
    ).rename(columns={"submitted_at": "at", "first_seen_at": "available_at"})
    review_comments = pd.read_parquet(
        db_dir / "PRReviewComments.parquet",
        columns=["pr_id", "repo_id", "author_id", "created_at", "body"],
    ).rename(columns={"created_at": "at"})

    # node_type/entity_id say which PR (1) or issue (2) a text belongs to,
    # matching the node types of experiments.relgt.data.
    def comment_bodies(frame: pd.DataFrame, node_type: int, key: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "body": frame["body"],
                "author_id": frame["author_id"].astype("Int64"),
                "at": frame["at"],
                "available_at": frame["at"],
                "node_type": node_type,
                "entity_id": frame[key],
            }
        )

    review_bodies = pd.DataFrame(
        {
            "body": reviews["body"],
            "author_id": reviews["reviewer_id"].astype("Int64"),
            "at": reviews["at"],
            "available_at": reviews["available_at"],
            "node_type": 1,
            "entity_id": reviews["pr_id"],
        }
    )
    mentions = extract_mentions(
        pd.concat(
            [
                issue_bodies,
                comment_bodies(issue_comments, 2, "issue_id"),
                comment_bodies(review_comments, 1, "pr_id"),
                review_bodies,
            ],
            ignore_index=True,
        ).dropna(subset=["body", "at", "available_at"]),
        login_to_user,
    )

    return RelationalTables(
        repo_owner=repo_owner,
        pushes=pushes,
        prs=prs,
        issues=issues,
        issue_comments=issue_comments.drop(columns="body"),
        reviews=reviews.drop(columns="body").dropna(subset=["at"]),
        review_comments=review_comments.drop(columns="body"),
        mentions=mentions,
    )


def feature_names(groups: tuple[str, ...] | list[str]) -> list[str]:
    unknown = sorted(set(groups) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"Unknown relational feature group(s): {', '.join(unknown)}")
    return [name for group in FEATURE_GROUPS if group in groups for name in FEATURE_GROUPS[group]]


def _between(times: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return (times >= start) & (times < end)


def _is_open(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    return frame["closed_at"].isna() | (frame["closed_at"] >= cutoff)


def _count_by(frame: pd.DataFrame, column: str, user_index: pd.Index) -> np.ndarray:
    counts = frame.groupby(column).size()
    counts.index = counts.index.astype("int64")
    return counts.reindex(user_index, fill_value=0).to_numpy(dtype=np.float32)


def _nunique_by(frame: pd.DataFrame, column: str, value: str, user_index: pd.Index) -> np.ndarray:
    counts = frame.groupby(column)[value].nunique()
    counts.index = counts.index.astype("int64")
    return counts.reindex(user_index, fill_value=0).to_numpy(dtype=np.float32)


def _events_on_user_repos(
    events: pd.DataFrame,
    actor: str,
    user_repos: pd.DataFrame,
    user_index: pd.Index,
) -> np.ndarray:
    """Count events by someone else on repos linked to each user (``user_repos``: user_id, repo_id)."""
    actors = events[["repo_id", actor]].rename(columns={actor: "_actor"})
    joined = user_repos.merge(actors, on="repo_id")
    joined = joined.loc[joined["_actor"].isna() | (joined["_actor"] != joined["user_id"])]
    return _count_by(joined, "user_id", user_index)


def relational_features(
    cutoff: pd.Timestamp,
    tables: RelationalTables,
    histories: dict[str, pd.DataFrame],
    user_ids: list[int],
    groups: tuple[str, ...] | list[str] = tuple(FEATURE_GROUPS),
) -> pd.DataFrame:
    """One row per user with the selected relational features, all cutoff-safe.

    ``histories`` are the cohort action histories from the LightGBM baseline
    (``user_id``, ``repo_id``, ``action_at``, ``available_at`` per label).
    """
    names = feature_names(groups)
    user_index = pd.Index(user_ids, dtype="int64")
    recent = cutoff - pd.Timedelta(days=RECENT_DAYS)
    context = cutoff - pd.Timedelta(days=CONTEXT_DAYS)
    open_window = cutoff - pd.Timedelta(days=OPEN_ITEM_DAYS)

    # Rows known before the cutoff.
    pushes = tables.pushes.loc[tables.pushes["at"] < cutoff]
    prs = tables.prs.loc[(tables.prs["available_at"] < cutoff) & (tables.prs["opened_at"] < cutoff)]
    issues = tables.issues.loc[
        (tables.issues["available_at"] < cutoff) & (tables.issues["opened_at"] < cutoff)
    ]
    plain_issues = issues.loc[~issues["is_pr"]]
    comments = tables.issue_comments.loc[tables.issue_comments["at"] < cutoff]
    reviews = tables.reviews.loc[
        (tables.reviews["at"] < cutoff) & (tables.reviews["available_at"] < cutoff)
    ]
    review_comments = tables.review_comments.loc[tables.review_comments["at"] < cutoff]

    pushes_7d = pushes.loc[pushes["at"] >= recent]
    pushes_30d = pushes.loc[pushes["at"] >= context]
    new_prs_7d = prs.loc[prs["opened_at"] >= recent]
    new_issues_7d = plain_issues.loc[plain_issues["opened_at"] >= recent]
    comments_7d = comments.loc[comments["at"] >= recent]
    reviews_7d = reviews.loc[reviews["at"] >= recent]
    review_comments_7d = review_comments.loc[review_comments["at"] >= recent]
    open_prs = prs.loc[_is_open(prs, cutoff) & (prs["available_at"] >= open_window)]
    open_issues = plain_issues.loc[
        _is_open(plain_issues, cutoff) & (plain_issues["available_at"] >= open_window)
    ]

    values: dict[str, np.ndarray] = {}
    wanted = set(names)

    if wanted & set(FEATURE_GROUPS["owner"]):
        owned = tables.repo_owner.rename("user_id").rename_axis("repo_id").reset_index()
        owned = owned.loc[owned["user_id"].isin(user_index)]
        pushes_owner = pushes_30d["repo_id"].map(tables.repo_owner)
        own_pushes = pushes_30d.loc[pushes_owner.eq(pushes_30d["user_id"]).fillna(False)]
        total = _count_by(pushes_30d, "user_id", user_index)
        values["own_push_share_30d"] = np.divide(
            _count_by(own_pushes, "user_id", user_index),
            total,
            out=np.zeros(len(user_index), dtype=np.float32),
            where=total > 0,
        )
        values["own_repos_pushed_30d"] = _nunique_by(own_pushes, "user_id", "repo_id", user_index)
        values["own_repo_new_issues_by_others_7d"] = _events_on_user_repos(
            new_issues_7d, "author_id", owned, user_index
        )
        values["own_repo_open_issues_by_others"] = _events_on_user_repos(
            open_issues, "author_id", owned, user_index
        )
        values["own_repo_issue_comments_by_others_7d"] = _events_on_user_repos(
            comments_7d, "author_id", owned, user_index
        )
        values["own_repo_new_prs_by_others_7d"] = _events_on_user_repos(
            new_prs_7d, "author_id", owned, user_index
        )
        values["own_repo_open_prs_by_others"] = _events_on_user_repos(
            open_prs, "author_id", owned, user_index
        )

    if wanted & set(FEATURE_GROUPS["sequence"]):
        feature_pushes = pushes_7d.loc[~pushes_7d["branch"].isin(MAINLINE_BRANCHES)]
        values["feature_branch_pushes_7d"] = _count_by(feature_pushes, "user_id", user_index)
        # Feature branches pushed recently for which the user has no PR yet.
        branches = feature_pushes[["user_id", "branch"]].drop_duplicates()
        paired = (
            prs.dropna(subset=["author_id"])[["author_id", "head_ref"]]
            .drop_duplicates()
            .rename(columns={"author_id": "user_id", "head_ref": "branch"})
            .astype({"user_id": "int64"})
        )
        unpaired = branches.merge(paired, on=["user_id", "branch"], how="left", indicator=True)
        unpaired = unpaired.loc[unpaired["_merge"] == "left_only"]
        values["unpaired_branches_7d"] = _count_by(unpaired, "user_id", user_index)
        pushed_repos = pushes_30d[["user_id", "repo_id"]].drop_duplicates()
        values["new_issues_on_pushed_repos_7d"] = _events_on_user_repos(
            new_issues_7d, "author_id", pushed_repos, user_index
        )

    if wanted & set(FEATURE_GROUPS["assigned"]):
        assigned = plain_issues.dropna(subset=["assignee_id"])
        assigned_open = assigned.loc[
            _is_open(assigned, cutoff) & (assigned["available_at"] >= open_window)
        ]
        values["assigned_open_issues"] = _count_by(assigned_open, "assignee_id", user_index)
        values["assigned_new_issues_7d"] = _count_by(
            assigned.loc[assigned["available_at"] >= recent], "assignee_id", user_index
        )

    if wanted & set(FEATURE_GROUPS["reviewer"]):
        reviews_90d = reviews.loc[reviews["at"] >= open_window]
        reviewed_repos = (
            reviews_90d[["reviewer_id", "repo_id"]]
            .drop_duplicates()
            .rename(columns={"reviewer_id": "user_id"})
        )
        reviewed_repos = reviewed_repos.loc[reviewed_repos["user_id"].isin(user_index)]
        values["reviewed_repos_90d"] = _count_by(reviewed_repos, "user_id", user_index)
        values["new_prs_on_reviewed_repos_7d"] = _events_on_user_repos(
            new_prs_7d, "author_id", reviewed_repos, user_index
        )
        values["open_prs_on_reviewed_repos"] = _events_on_user_repos(
            open_prs.loc[open_prs["available_at"] >= context], "author_id", reviewed_repos, user_index
        )
        reviewed_prs = reviews.loc[reviews["at"] >= context, ["reviewer_id", "pr_id"]].drop_duplicates()
        still_open = reviewed_prs.merge(
            prs.loc[_is_open(prs, cutoff), ["pr_id"]], on="pr_id"
        )
        values["open_prs_reviewed_by_user"] = _count_by(still_open, "reviewer_id", user_index)

    if wanted & set(FEATURE_GROUPS["author"]):
        values["open_authored_prs"] = _count_by(
            open_prs.dropna(subset=["author_id"]), "author_id", user_index
        )
        values["open_authored_issues"] = _count_by(
            open_issues.dropna(subset=["author_id"]), "author_id", user_index
        )
        pr_authors = prs.dropna(subset=["author_id"])[["pr_id", "author_id"]]
        received = reviews_7d.merge(pr_authors, on="pr_id")
        received = received.loc[received["reviewer_id"] != received["author_id"]]
        values["reviews_received_7d"] = _count_by(received, "author_id", user_index)
        received = review_comments_7d.rename(columns={"author_id": "commenter_id"}).merge(
            pr_authors, on="pr_id"
        )
        received = received.loc[received["commenter_id"] != received["author_id"]]
        values["review_comments_received_7d"] = _count_by(received, "author_id", user_index)
        # Issue authors include PR conversations, so this covers comments on own PRs too.
        issue_authors = issues.dropna(subset=["author_id"])[["issue_id", "author_id"]]
        received = comments_7d.rename(columns={"author_id": "commenter_id"}).merge(
            issue_authors, on="issue_id"
        )
        received = received.loc[received["commenter_id"] != received["author_id"]]
        values["issue_comments_received_7d"] = _count_by(received, "author_id", user_index)

    if wanted & set(FEATURE_GROUPS["repo_activity"]):
        touched = []
        for history in histories.values():
            known = history.loc[
                (history["available_at"] < cutoff)
                & (history["action_at"] < cutoff)
                & (history["action_at"] >= context)
            ]
            touched.append(known[["user_id", "repo_id"]])
        active_repos = pd.concat(touched).drop_duplicates()
        active_repos = active_repos.loc[active_repos["user_id"].isin(user_index)]
        values["active_repos_30d"] = _count_by(active_repos, "user_id", user_index)
        values["neighbor_new_prs_7d"] = _events_on_user_repos(
            new_prs_7d, "author_id", active_repos, user_index
        )
        values["neighbor_new_issues_7d"] = _events_on_user_repos(
            new_issues_7d, "author_id", active_repos, user_index
        )
        values["neighbor_issue_comments_7d"] = _events_on_user_repos(
            comments_7d, "author_id", active_repos, user_index
        )
        values["neighbor_reviews_7d"] = _events_on_user_repos(
            reviews_7d, "reviewer_id", active_repos, user_index
        )
        values["neighbor_review_comments_7d"] = _events_on_user_repos(
            review_comments_7d, "author_id", active_repos, user_index
        )
        values["neighbor_pushes_7d"] = _events_on_user_repos(
            pushes_7d, "user_id", active_repos, user_index
        )

    if wanted & set(FEATURE_GROUPS["mentions"]):
        mentions = tables.mentions.loc[
            (tables.mentions["at"] < cutoff) & (tables.mentions["available_at"] < cutoff)
        ]
        values["mentions_received_7d"] = _count_by(
            mentions.loc[mentions["at"] >= recent], "user_id", user_index
        )
        values["mentions_received_30d"] = _count_by(
            mentions.loc[mentions["at"] >= context], "user_id", user_index
        )

    if wanted & set(FEATURE_GROUPS["threads"]):
        # Threads the user joined: issues/PR conversations they commented on in
        # the last 30 days or opened within the open-item window.
        joined = pd.concat(
            [
                comments.loc[comments["at"] >= context, ["author_id", "issue_id"]],
                issues.loc[issues["available_at"] >= open_window, ["author_id", "issue_id"]],
            ]
        ).dropna().drop_duplicates()
        joined = joined.rename(columns={"author_id": "user_id"}).astype({"user_id": "int64"})
        joined = joined.loc[joined["user_id"].isin(user_index)]
        new_comments = comments_7d[["issue_id", "author_id"]].rename(columns={"author_id": "_actor"})
        replies = joined.merge(new_comments, on="issue_id")
        replies = replies.loc[replies["_actor"] != replies["user_id"]]
        values["thread_new_comments_7d"] = _count_by(replies, "user_id", user_index)

    if wanted & set(FEATURE_GROUPS["review_cycle"]):
        merged_7d = prs.loc[_between(prs["merged_at"], recent, cutoff)]
        values["own_prs_merged_7d"] = _count_by(
            merged_7d.dropna(subset=["author_id"]), "author_id", user_index
        )
        by_others = merged_7d.dropna(subset=["merged_by_id"])
        by_others = by_others.loc[
            by_others["author_id"].isna() | (by_others["author_id"] != by_others["merged_by_id"])
        ]
        values["merged_others_prs_7d"] = _count_by(by_others, "merged_by_id", user_index)
        pr_authors = prs.dropna(subset=["author_id"])[["pr_id", "author_id"]]
        received = reviews.loc[reviews["at"] >= context].merge(pr_authors, on="pr_id")
        received = received.loc[received["reviewer_id"] != received["author_id"]]
        values["approvals_received_30d"] = _count_by(
            received.loc[received["state"] == "approved"], "author_id", user_index
        )
        pushed = pushes_7d[["user_id", "branch"]].drop_duplicates()
        active_prs = open_prs.dropna(subset=["author_id"])[["author_id", "head_ref"]]
        active_prs = active_prs.astype({"author_id": "int64"}).merge(
            pushed, left_on=["author_id", "head_ref"], right_on=["user_id", "branch"]
        )
        values["open_prs_with_recent_push"] = _count_by(active_prs, "author_id", user_index)

    if wanted & set(FEATURE_GROUPS["habit"]):
        habit_start = cutoff - pd.Timedelta(weeks=HABIT_WEEKS)
        weekly_frames = []
        for label in LABELS:
            history = histories.get(label)
            if history is None:
                values[f"{label}_active_weeks_8w"] = np.zeros(len(user_index), dtype=np.float32)
                continue
            known = history.loc[
                (history["available_at"] < cutoff)
                & (history["action_at"] < cutoff)
                & (history["action_at"] >= habit_start)
            ]
            weeks = ((cutoff - known["action_at"]) // pd.Timedelta(weeks=1)).astype("int64")
            active = pd.DataFrame({"user_id": known["user_id"], "week": weeks}).drop_duplicates()
            weekly_frames.append(active)
            values[f"{label}_active_weeks_8w"] = _count_by(active, "user_id", user_index)
        any_active = (
            pd.concat(weekly_frames).drop_duplicates()
            if weekly_frames
            else pd.DataFrame({"user_id": [], "week": []})
        )
        values["any_active_weeks_8w"] = _count_by(any_active, "user_id", user_index)

    frame = pd.DataFrame({"user_id": user_ids})
    for name in names:
        frame[name] = values[name]
    return frame


def hypothesis_lift(frame: pd.DataFrame) -> pd.DataFrame:
    """Empirical P(label | feature > 0) vs P(label | feature == 0) for each hypothesis."""
    rows = []
    for hypothesis, feature, label in HYPOTHESES:
        if feature not in frame.columns:
            continue
        active = frame[feature].to_numpy() > 0
        y = frame[label].to_numpy()
        with_rate = float(y[active].mean()) if active.any() else float("nan")
        without_rate = float(y[~active].mean()) if (~active).any() else float("nan")
        rows.append(
            {
                "hypothesis": hypothesis,
                "feature": feature,
                "label": label,
                "rows_with_feature": int(active.sum()),
                "rate_with_feature": with_rate,
                "rate_without_feature": without_rate,
                "lift": with_rate / without_rate if without_rate else float("nan"),
            }
        )
    return pd.DataFrame(rows)
