# Retrain on all weeks, predict the week after the data ends,
# and write the submission file.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from dataset import features, labels, ACTS, users
from train import thresholds, t_val, t_train

t_end = pd.Timestamp("2026-11-20 19:00")      # end of downloaded data
cuts = t_train + [t_val]                       # use all labeled weeks
X = pd.concat([features(t) for t in cuts])
y = pd.concat([labels(t) for t in cuts])
X_te = features(t_end)

pred = pd.DataFrame(index=X_te.index)
for a in ACTS:
    m = LogisticRegression(max_iter=1000).fit(np.log1p(X), y[a])
    pred[a] = (m.predict_proba(np.log1p(X_te))[:, 1] >= thresholds[a]).astype(int)

# map row ids to the GitHub user codes used in test_input.csv; unseen users get 0
pred.index = users.set_index("user_id").loc[pred.index, "github_user_id"]
test = pd.read_csv("assignment2/submission/test_input.csv")
out = pred.reindex(test["UserID"], fill_value=0).reset_index(names="UserID")
out.to_csv("output_315552057.csv", index=False)
print(out[ACTS].mean())