"""Question 1: schema graph of the 8 tables (Graphviz).

Nodes are tables, edges are foreign keys (FK column -> referenced PK). Each node lists its PK and
FK columns; the remaining columns are only counted. Keys follow assignment2/SCHEMA.md.

Run from the repo root (needs `pip install graphviz` and the Graphviz `dot` binary on PATH):
    python analysis/schema_graph.py               # writes figures/schema_graph.{dot,png,svg}
"""
import os
import shutil
from pathlib import Path

import graphviz
import pyarrow.parquet as pq

# The Windows installer does not add Graphviz to PATH; fall back to its default location.
GRAPHVIZ_BIN = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Graphviz" / "bin"
if shutil.which("dot") is None and (GRAPHVIZ_BIN / "dot.exe").exists():
    os.environ["PATH"] = f"{GRAPHVIZ_BIN}{os.pathsep}{os.environ['PATH']}"

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "db"
OUT = ROOT / "figures" / "schema_graph"

# table -> (primary key, {foreign key column: referenced table})
SCHEMA = {
    "Users": ("user_id", {}),
    "Repositories": ("repo_id", {}),
    "PullRequests": ("pr_id", {"repo_id": "Repositories", "author_id": "Users",
                               "merged_by_id": "Users"}),
    "Issues": ("issue_id", {"repo_id": "Repositories", "author_id": "Users",
                            "pull_request_id": "PullRequests", "assignee_id": "Users"}),
    "IssueComments": ("comment_id", {"issue_id": "Issues", "repo_id": "Repositories",
                                     "author_id": "Users"}),
    "PRReviews": ("review_id", {"pr_id": "PullRequests", "repo_id": "Repositories",
                                "reviewer_id": "Users"}),
    "PRReviewComments": ("comment_id", {"pr_id": "PullRequests", "repo_id": "Repositories",
                                        "author_id": "Users", "pull_request_review_id": "PRReviews",
                                        "in_reply_to_id": "PRReviewComments"}),
    "Pushes": ("push_id", {"user_id": "Users", "repo_id": "Repositories"}),
}

HEADER, PK_FILL, FK_FILL, MUTED = "#2a78d6", "#fdf3d8", "#e8f1fc", "#898781"


def node_label(table, pk, fks, n_cols, n_rows):
    """HTML-like table: header, PK row, one row per FK (with a port for edges), other columns."""
    rows = [f'<tr><td bgcolor="{HEADER}" colspan="2"><font color="white"><b>{table}</b></font></td></tr>',
            f'<tr><td colspan="2"><font point-size="9" color="{MUTED}">{n_rows:,} rows</font></td></tr>',
            f'<tr><td bgcolor="{PK_FILL}" port="{pk}" align="left"><b>PK</b></td>'
            f'<td bgcolor="{PK_FILL}" align="left"><b><u>{pk}</u></b></td></tr>']
    for col, ref in fks.items():
        rows.append(f'<tr><td bgcolor="{FK_FILL}" align="left"><b>FK</b></td>'
                    f'<td bgcolor="{FK_FILL}" port="{col}" align="left"><i>{col}</i> → {ref}</td></tr>')
    rest = n_cols - 1 - len(fks)
    rows.append(f'<tr><td colspan="2"><font point-size="9" color="{MUTED}">+ {rest} other columns'
                f'</font></td></tr>')
    return f'<<table border="0" cellborder="1" cellspacing="0" cellpadding="5">{"".join(rows)}</table>>'


def build():
    g = graphviz.Digraph("schema", graph_attr=dict(rankdir="LR", splines="spline", nodesep="0.5",
                                                   ranksep="1.4", fontname="Helvetica", labelloc="t",
                                                   label="GitHub activity database: schema graph\n"
                                                         "edge = foreign key → referenced primary key",
                                                   fontsize="16", pad="0.3"),
                         node_attr=dict(shape="plaintext", fontname="Helvetica", fontsize="11"),
                         edge_attr=dict(color="#52514e", arrowsize="0.8"))
    for table, (pk, fks) in SCHEMA.items():
        meta = pq.read_metadata(DB / f"{table}.parquet")
        g.node(table, node_label(table, pk, fks, meta.num_columns, meta.num_rows))
    for table, (_, fks) in SCHEMA.items():
        for col, ref in fks.items():
            ref_pk = SCHEMA[ref][0]
            # Self-reference (reply thread) loops on the east side so it stays readable.
            g.edge(f"{table}:{col}:e", f"{ref}:{ref_pk}:" + ("e" if ref == table else "w"),
                   tooltip=f"{table}.{col} → {ref}.{ref_pk}")
    # Users and Repositories (referenced only) on the right, event tables on the left.
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("Users")
        s.node("Repositories")
    return g


def main():
    g = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.with_suffix(".dot").write_text(g.source, encoding="utf-8")
    for fmt in ["png", "svg"]:
        g.render(OUT, format=fmt, cleanup=True, engine="dot")
        print(f"Wrote {OUT.with_suffix('.' + fmt)}")
    n_edges = sum(len(fks) for _, fks in SCHEMA.values())
    print(f"{len(SCHEMA)} tables, {n_edges} foreign keys")


if __name__ == "__main__":
    main()
