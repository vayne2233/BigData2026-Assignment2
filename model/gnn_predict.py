# Load the trained RelGNN, pick thresholds on the validation week
# predict the week after the data ends.
import sys
import pandas as pd
import torch
from gnn_db import load_db
from gnn_graph import build_graph
from gnn_task import make_table, T_VAL, T_TEST, ACTS
from gnn_train import RelGNNModel, make_loader, predict, best_thresholds
from dataset import users

data, col_stats = build_graph(load_db())
model = RelGNNModel(data, col_stats)
model.load_state_dict(torch.load("model/relgnn.pt"))

# thresholds from the validation week
val_tab = make_table([T_VAL])
th = best_thresholds(predict(model, make_loader(data, val_tab, shuffle=False)), val_tab[ACTS].values)

# predict at the end of the data (labels here are unknown, only used as placeholders)
test_tab = make_table([T_TEST])
prob = predict(model, make_loader(data, test_tab, shuffle=False))
pred = pd.DataFrame({a: (prob[:, i] >= th[a]).astype(int) for i, a in enumerate(ACTS)})

# map row ids to GitHub user codes; users not in our data get 0
pred.index = users.set_index("user_id").loc[test_tab["user_id"], "github_user_id"].values
test = pd.read_csv("assignment2/submission/test_input.csv")
out = pred.reindex(test["UserID"], fill_value=0).rename_axis("UserID").reset_index()
out.to_csv("output_315552057.csv", index=False)
print(out[ACTS].mean())
