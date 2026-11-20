import pandas as pd
from pathlib import Path

T = {f.stem: pd.read_parquet(f) for f in Path("data/db").glob("*.parquet")}

FK = [
    ("PullRequests", "repo_id", "Repositories", "repo_id"),
    ("PullRequests", "author_id", "Users", "user_id"),
    ("IssueComments", "issue_id", "Issues", "issue_id"), 
    ("PRReviews", "pr_id", "PullRequests", "pr_id"),
]

for c, col, p, pk in FK:
    print(c, col, "->", p, T[c][col].dropna().isin(T[p][pk]).mean())