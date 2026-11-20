# Train one logistic regression per activity
# validate on a later week and pick the F1-best threshold.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from dataset import features, labels, ACTS

D = pd.Timedelta(days=1)
t_val = pd.Timestamp("2026-11-13 19:00")    # validation: predict 11/13-11/20
t_train = [t_val - 7 * D, t_val - 14 * D]      # training: two earlier weeks

X_tr = pd.concat([features(t) for t in t_train])
y_tr = pd.concat([labels(t) for t in t_train])
X_va, y_va = features(t_val), labels(t_val)

thresholds = {}
for a in ACTS:
    m = LogisticRegression(max_iter=1000).fit(np.log1p(X_tr), y_tr[a])
    p = m.predict_proba(np.log1p(X_va))[:, 1]
    best = max(np.arange(0.05, 0.9, 0.05), key=lambda th: f1_score(y_va[a], p >= th))
    thresholds[a] = best
    print(f"{a:18s} threshold={best:.2f}  F1={f1_score(y_va[a], p >= best):.3f}")