"""Read the 8 tables and turn them into per-activity event lists of cohort users."""
import numpy as np
import pandas as pd

from .config import DATA_START, DB, FEED_END

TABLE_COLUMNS = {
    "users": ("Users", ["user_id", "github_user_id", "is_cohort"]),
    "prs": ("PullRequests", ["repo_id", "author_id", "created_at", "closed_at", "first_seen_at"]),
    "issues": ("Issues", ["repo_id", "author_id", "is_pull_request", "created_at", "closed_at",
                          "first_seen_at"]),
    "comments": ("IssueComments", ["issue_id", "repo_id", "author_id", "created_at"]),
    "reviews": ("PRReviews", ["pr_id", "repo_id", "reviewer_id", "submitted_at", "first_seen_at"]),
    "rcomments": ("PRReviewComments", ["pr_id", "repo_id", "author_id", "created_at"]),
    "pushes": ("Pushes", ["user_id", "repo_id", "created_at"]),
}


def read_tables():
    """The columns load() uses, per table (separate so an experiment can alter the rows first)."""
    return {k: pd.read_parquet(DB / f"{name}.parquet", columns=cols)
            for k, (name, cols) in TABLE_COLUMNS.items()}


def load(tables=None):
    """Read the tables and turn them into per-source event lists of cohort users.

    Each event has uid (cohort index), t (when it happened), avail (when the row is visible in
    the data, so features at cutoff c may only use avail < c) and repo. `tables` (from
    read_tables(), possibly altered) replaces reading the parquet files.
    """
    t = read_tables() if tables is None else tables
    users, prs, issues, comments, reviews, rcomments, pushes = (t[k] for k in TABLE_COLUMNS)

    cohort = users[users.is_cohort].reset_index(drop=True)
    to_cidx = np.full(len(users), -1, np.int64)
    to_cidx[cohort.user_id.values] = np.arange(len(cohort))

    def ids(s):
        return s.to_numpy(dtype="float64", na_value=np.nan)

    def events(uid, t, avail, repo):
        d = pd.DataFrame({"uid": ids(uid), "t": t.values, "avail": avail, "repo": ids(repo)})
        d = d[d.uid.notna() & d.t.notna() & (d.t >= DATA_START)]
        d["uid"] = to_cidx[d.uid.astype(np.int64).values]
        return d[d.uid >= 0].sort_values("t").reset_index(drop=True)

    def later(*cols):
        return np.maximum.reduce([c.values for c in cols])

    plain = issues[~issues.is_pull_request]
    ev = {
        "open_pr": events(prs.author_id, prs.created_at, later(prs.created_at, prs.first_seen_at),
                          prs.repo_id),
        "open_issue": events(plain.author_id, plain.created_at,
                             later(plain.created_at, plain.first_seen_at), plain.repo_id),
        "issue_comment": events(comments.author_id, comments.created_at, comments.created_at.values,
                                comments.repo_id),
        "pr_review": events(reviews.reviewer_id, reviews.submitted_at,
                            later(reviews.submitted_at, reviews.first_seen_at), reviews.repo_id),
        "pr_review_comment": events(rcomments.author_id, rcomments.created_at,
                                    rcomments.created_at.values, rcomments.repo_id),
        "push": events(pushes.user_id, pushes.created_at, pushes.created_at.values, pushes.repo_id),
    }

    # Relational: activity other users direct at the user's own issues and PRs
    # (comment -> issue -> author, review / review comment -> PR -> author).
    parent_issue = issues[["author_id", "first_seen_at"]].rename(
        columns={"author_id": "owner", "first_seen_at": "parent_seen"})
    parent_pr = prs[["author_id", "first_seen_at"]].rename(
        columns={"author_id": "owner", "first_seen_at": "parent_seen"})
    recv = [
        comments.rename(columns={"author_id": "actor"}).merge(
            parent_issue, left_on="issue_id", right_index=True, how="left"),
        reviews.rename(columns={"reviewer_id": "actor", "submitted_at": "created_at"}).merge(
            parent_pr, left_on="pr_id", right_index=True, how="left"),
        rcomments.rename(columns={"author_id": "actor"}).merge(
            parent_pr, left_on="pr_id", right_index=True, how="left"),
    ]
    parts = []
    for d in recv:
        d = d[d.owner.notna() & d.actor.notna()]
        d = d[ids(d.owner) != ids(d.actor)]
        seen = [d.created_at, d.parent_seen] + ([d.first_seen_at] if "first_seen_at" in d else [])
        parts.append(events(d.owner, d.created_at, later(*seen), d.repo_id))
    ev["received"] = pd.concat(parts).sort_values("t").reset_index(drop=True)

    # Authored PRs / issues with open/closed state (closed_at is only used when it is before the cutoff).
    own = {}
    for k, d in {"pr": prs, "issue": plain}.items():
        d = d[d.author_id.notna() & d.created_at.notna()]
        own[k] = pd.DataFrame({"uid": to_cidx[ids(d.author_id).astype(np.int64)],
                               "avail": later(d.created_at, d.first_seen_at),
                               "closed_at": d.closed_at.values})
        own[k] = own[k][own[k].uid >= 0].reset_index(drop=True)

    # Last moment each table has data; the lag of a source = TEST_START - that moment.
    data_end = {
        "open_pr": prs.first_seen_at.max(),
        "open_issue": issues.first_seen_at.max(),
        "issue_comment": comments.created_at.max(),
        "pr_review": reviews.first_seen_at.max(),
        "pr_review_comment": rcomments.created_at.max(),
        "push": pushes.created_at.max(),
    }
    for k, end in FEED_END.items():
        data_end[k] = min(data_end[k], end)
    data_end["received"] = min(data_end[k] for k in ["issue_comment", "pr_review", "pr_review_comment"])
    return cohort, ev, own, data_end
