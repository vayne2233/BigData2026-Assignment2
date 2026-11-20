# Features = activity counts in the 7/28 days before t
# labels = did each activity in the 7 days after t.
import pandas as pd

ACTS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
ev = pd.read_parquet("data_4w/events.parquet")
users = pd.read_parquet("data_4w/db/Users.parquet")
cohort = users.loc[users["is_cohort"], "user_id"]
D = pd.Timedelta(days=1)

def count(start, end):
    w = ev[(ev["time"] >= start) & (ev["time"] < end)]
    return pd.crosstab(w["user_id"], w["activity"]).reindex(index=cohort, columns=ACTS, fill_value=0)

def features(t):
    last = ev[ev["time"] < t].groupby("user_id")["time"].max()
    X = pd.concat([count(t - 7*D, t).add_suffix("_7d"), count(t - 28*D, t).add_suffix("_28d")], axis=1)
    X["days_since_last"] = (t - last).dt.days.reindex(cohort).fillna(99)
    return X

def labels(t):
    return count(t, t + 7*D).clip(upper=1)

if __name__ == "__main__":
    t = pd.Timestamp("2026-11-13 19:00")
    X, y = features(t), labels(t)
    print(X.shape, y.shape)
    print(y.mean())   # share of cohort users doing each activity next week