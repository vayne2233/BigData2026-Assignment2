# Verify foreign keys: share of FK values found 
# in the parent table's PK (1.0 = valid).

import pandas as pd
from pathlib import Path

T = {f.stem: pd.read_parquet(f) for f in Path("data/db").glob("*.parquet")}

# (chile table, FK column, parent table, PK column)
FK = [
    ("PullRequests", "repo_id", "Repositories", "repo_id"),
    ("PullRequests", "author_id", "Users", "user_id"),
    ("IssueComments", "issue_id", "Issues", "issue_id"), 
    ("PRReviews", "pr_id", "PullRequests", "pr_id"),
]

for c, col, p, pk in FK:
    print(c, col, "->", p, T[c][col].dropna().isin(T[p][pk]).mean())

FK = [
    # -> Users
    ("Issues", "author_id", "Users", "user_id"),
    ("Issues", "assignee_id", "Users", "user_id"),
    ("IssueComments", "author_id", "Users", "user_id"),
    ("PullRequests", "author_id", "Users", "user_id"),
    ("PullRequests", "merged_by_id", "Users", "user_id"),
    ("PRReviews", "reviewer_id", "Users", "user_id"),
    ("PRReviewComments", "author_id", "Users", "user_id"),
    ("Pushes", "user_id", "Users", "user_id"),
    # -> Repositories
    ("Issues", "repo_id", "Repositories", "repo_id"),
    ("IssueComments", "repo_id", "Repositories", "repo_id"),
    ("PullRequests", "repo_id", "Repositories", "repo_id"),
    ("PRReviews", "repo_id", "Repositories", "repo_id"),
    ("PRReviewComments", "repo_id", "Repositories", "repo_id"),
    ("Pushes", "repo_id", "Repositories", "repo_id"),
    # -> PullRequests / Issues / PRReviews / PRReviewComments
    ("Issues", "pull_request_id", "PullRequests", "pr_id"),
    ("IssueComments", "issue_id", "Issues", "issue_id"),
    ("PRReviews", "pr_id", "PullRequests", "pr_id"),
    ("PRReviewComments", "pr_id", "PullRequests", "pr_id"),
    ("PRReviewComments", "pull_request_review_id", "PRReviews", "review_id"),
    ("PRReviewComments", "in_reply_to_id", "PRReviewComments", "comment_id"),
]

for c, col, p, pk in FK:
    print(c, col, "->", p, T[c][col].dropna().isin(T[p][pk]).mean())