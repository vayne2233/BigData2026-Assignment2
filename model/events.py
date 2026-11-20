# Build one event table: (user_id, time, activity) 
# for the 6 target activities.
import pandas as pd
from pathlib import Path

T = {f.stem: pd.read_parquet(f) for f in Path("data_4w/db").glob("*.parquet")}

# issues only, not PRs
T["Issues"] = T["Issues"][~T["Issues"]["is_pull_request"]]  

# (table, user column, time column, activity)
ACT = [
    ("PullRequests", "author_id", "created_at", "open_pr"),
    ("Issues", "author_id", "created_at", "open_issue"),
    ("IssueComments", "author_id", "created_at", "issue_comment"),
    ("PRReviews", "reviewer_id", "submitted_at", "pr_review"),
    ("PRReviewComments", "author_id", "created_at", "pr_review_comment"),
    ("Pushes", "user_id", "created_at", "push"),
]

events = pd.concat(
    T[t][[u, c]].set_axis(["user_id", "time"], axis=1).assign(activity=a)
    for t, u, c, a in ACT
).dropna()

events = events[events["time"] >= "2026-10-25"]

print(events.groupby("activity").size(), events["time"].agg(["min", "max"]))
events.to_parquet("data_4w/events.parquet")