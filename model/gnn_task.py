# Training tables for the GNN: one row per
# (user, cut-off time t) with 6 labels for the 7 days after t.
import pandas as pd
from dataset import labels, ACTS

D = pd.Timedelta(days=1)
T_VAL = pd.Timestamp("2026-11-13 19:00")    # validation week: 11/13 - 11/20
T_TEST = pd.Timestamp("2026-11-20 19:00")   # predict the week after the data ends

# Training cut-offs: one per day, each followed by a full 7-day label window before T_VAL
TRAIN_TIMES = pd.date_range(T_VAL - 12 * D, T_VAL - 7 * D, freq="D")


def make_table(times):
    rows = []
    for t in times:
        y = labels(t).reset_index()          # columns: user_id + 6 activities
        y["timestamp"] = t
        rows.append(y)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    train = make_table(TRAIN_TIMES)
    val = make_table([T_VAL])
    print("train", train.shape, "val", val.shape)
    print(train[ACTS].mean().round(3))