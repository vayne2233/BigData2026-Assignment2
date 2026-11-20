# Turn the RelBench Database into a heterogeneous
# graph: rows -> nodes, foreign keys -> edges.
from torch_frame import stype
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from gnn_db import load_db

# skip masked text columns
KEEP = {stype.numerical, stype.categorical, stype.timestamp}


def build_graph(db):
    col_to_stype = get_stype_proposal(db)
    col_to_stype = {table: {col: st for col, st in cols.items() if st in KEEP}
                    for table, cols in col_to_stype.items()}
    data, col_stats = make_pkey_fkey_graph(db, col_to_stype_dict=col_to_stype,
                                           text_embedder_cfg=None, cache_dir=None)
    return data, col_stats


if __name__ == "__main__":
    data, col_stats = build_graph(load_db())
    print(data)