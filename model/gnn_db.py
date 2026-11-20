# Wrap the 8 parquet tables as a RelBench Database.
# Keys and row times follow SCHEMA.md.
import pandas as pd
from pathlib import Path
from relbench.base import Database, Table

DB_DIR = Path("data_4w/db")

# Primary key of each table
PK = {
    "Users": "user_id",
    "Repositories": "repo_id",
    "PullRequests": "pr_id",
    "Issues": "issue_id",
    "IssueComments": "comment_id",
    "PRReviews": "review_id",
    "PRReviewComments": "comment_id",
    "Pushes": "push_id",
}

# Row time: when each row first appears in the data
TIME = {
    "Users": "first_seen_at",
    "Repositories": "first_seen_at",
    "PullRequests": "first_seen_at",
    "Issues": "first_seen_at",
    "IssueComments": "created_at",
    "PRReviews": "first_seen_at",
    "PRReviewComments": "created_at",
    "Pushes": "created_at",
}

# Foreign keys: (child table, FK column, parent table)
FK = [
    ("PullRequests", "repo_id", "Repositories"),
    ("PullRequests", "author_id", "Users"),
    ("Issues", "repo_id", "Repositories"),
    ("Issues", "author_id", "Users"),
    ("Issues", "assignee_id", "Users"),
    ("Issues", "pull_request_id", "PullRequests"),
    ("IssueComments", "issue_id", "Issues"),
    ("IssueComments", "repo_id", "Repositories"),
    ("IssueComments", "author_id", "Users"),
    ("PRReviews", "pr_id", "PullRequests"),
    ("PRReviews", "repo_id", "Repositories"),
    ("PRReviews", "reviewer_id", "Users"),
    ("PRReviewComments", "pr_id", "PullRequests"),
    ("PRReviewComments", "repo_id", "Repositories"),
    ("PRReviewComments", "author_id", "Users"),
    ("PRReviewComments", "pull_request_review_id", "PRReviews"),
    ("PRReviewComments", "in_reply_to_id", "PRReviewComments"),
    ("Pushes", "user_id", "Users"),
    ("Pushes", "repo_id", "Repositories"),
]

# Columns known only later (assignment rule): drop them to avoid leakage
KNOWN_LATER = ["merged_at", "merged_by_id", "closed_at", "state_reason", "dismissed_at"]


def load_table(name):
    df = pd.read_parquet(DB_DIR / f"{name}.parquet")
    df = df.drop(columns=KNOWN_LATER, errors="ignore")
    fks = {col: parent for child, col, parent in FK if child == name}
    return Table(df=df, fkey_col_to_pkey_table=fks, pkey_col=PK[name], time_col=TIME[name])


def load_db():
    return Database({name: load_table(name) for name in PK})


if __name__ == "__main__":
    db = load_db()
    for name, table in db.table_dict.items():
        times = table.df[table.time_col]
        print(f"{name:18s} rows = {len(table.df):6d} {times.min()} ~ {times.max()}")
    print("DB time range: ", db.min_timestamp, "~", db.max_timestamp)