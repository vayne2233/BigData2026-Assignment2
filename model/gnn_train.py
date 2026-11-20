# Train a RelBench GraphSAGE model (as in RelBench examples)
# to predict the 6 activities.
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
from gnn_db import load_db
from gnn_graph import build_graph
from gnn_task import make_table, TRAIN_TIMES, T_VAL, ACTS

torch.manual_seed(0)
CHANNELS, LAYERS, EPOCHS = 64, 2, 20


class Model(torch.nn.Module):
    """Encode each table's columns, add time encoding, pass messages over FK edges, predict 6 logits."""
    def __init__(self, data, col_stats):
        super().__init__()
        self.encoder = HeteroEncoder(
            channels=CHANNELS,
            node_to_col_names_dict={n: data[n].tf.col_names_dict for n in data.node_types},
            node_to_col_stats=col_stats)
        self.time_encoder = HeteroTemporalEncoder(
            node_types=[n for n in data.node_types if "time" in data[n]], channels=CHANNELS)
        self.gnn = HeteroGraphSAGE(node_types=data.node_types, edge_types=data.edge_types,
                                   channels=CHANNELS, aggr="sum", num_layers=LAYERS)
        self.head = MLP(CHANNELS, out_channels=len(ACTS), norm="batch_norm", num_layers=1)

    def forward(self, batch):
        seed_time = batch["Users"].seed_time
        x = self.encoder(batch.tf_dict)
        for n, rel_time in self.time_encoder(seed_time, batch.time_dict, batch.batch_dict).items():
            x[n] = x[n] + rel_time            # time encoding: how long before t each row happened
        x = self.gnn(x, batch.edge_index_dict)
        return self.head(x["Users"][: seed_time.size(0)])

        
class RelGNNModel(torch.nn.Module):
    """
    Official RelGNN (snap-standford/RelGNN): composite
    message passing over atomic routes.
    """
    def __init__(self, data, col_stats):
        super().__init__()
        from relgnn_model import RelGNN_Model
        from atomic_routes import get_atomic_routes
        self.net = RelGNN_Model(data=data, col_stats_dict=col_stats, num_model_layers=LAYERS,
                                channels=CHANNELS, out_channels=len(ACTS), aggr="sum",
                                norm="batch_norm", num_heads=8,
                                atomic_routes=get_atomic_routes(data.edge_types))

    def forward(self, batch):
        return self.net(batch, "Users")
        

def make_loader(data, table, shuffle):
    users = torch.tensor(table["user_id"].values)
    times = torch.tensor(((table["timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).values)
    y = torch.tensor(table[ACTS].values, dtype=torch.float)

    def attach_y(batch):
        batch["Users"].y = y[batch["Users"].input_id]
        return batch

    return NeighborLoader(data, num_neighbors=[64, 32], time_attr="time",
                          input_nodes=("Users", users), input_time=times,
                          transform=attach_y, batch_size=512, shuffle=shuffle,
                          temporal_strategy="uniform")


@torch.no_grad()
def predict(model, loader):
    model.eval()
    return torch.cat([torch.sigmoid(model(b)) for b in loader]).numpy()


def best_thresholds(prob, y):
    th = {}
    for i, a in enumerate(ACTS):
        th[a] = max(np.quantile(prob[:, i], np.linspace(0.5, 0.9, 50)),
                    key = lambda t: f1_score(y[:, i], prob[:, i] >= t))
    return th


if __name__ == "__main__":
    data, col_stats = build_graph(load_db())
    train_tab, val_tab = make_table(TRAIN_TIMES), make_table([T_VAL])
    train_loader = make_loader(data, train_tab, shuffle=True)
    val_loader = make_loader(data, val_tab, shuffle=False)

    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "sage"
    model = RelGNNModel(data, col_stats) if name == "relgnn" else Model(data, col_stats)

    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    pos = torch.tensor(train_tab[ACTS].values, dtype=torch.float).mean(0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(batch), batch["Users"].y)
            loss.backward()
            opt.step()
        print(f"epoch {epoch:02d}  loss={loss.item():.4f}")

    prob, y_val = predict(model, val_loader), val_tab[ACTS].values
    th = best_thresholds(prob, y_val)
    f1s = [f1_score(y_val[:, i], prob[:, i] >= th[a]) for i, a in enumerate(ACTS)]
    for a, f in zip(ACTS, f1s):
        print(f"{a:18s} threshold={th[a]:.2f}  F1={f:.3f}")
    print("macro-F1 =", round(np.mean(f1s), 3))
    print("AUC", [round(roc_auc_score(y_val[:, i], prob[:, i]), 3)
                  for i in range(len(ACTS))])
    torch.save(model.state_dict(), f"model/{name}.pt")
