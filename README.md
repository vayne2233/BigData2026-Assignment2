# BigData Assignment 2: Predicting GitHub User Activity

## Data Structure
GH Archive: https://www.gharchive.org/

Period used: 2026-10-25 00:00 ~ 2026-11-20 19:00 (UTC)

## Setup
uv sync

## How to run
```bash
# 1. Download raw data into raw/, then build the 8 tables
uv run python assignment2/scripts/preprocess.py --raw raw --out data_4w

# 2. Build the event table (used for labels and history features)
uv run python model/events.py

# 3. Train RelGNN; also: sage, hybrid
uv run python model/gnn_train.py relgnn

# 4. Predict and check the format
uv run model/gnn_predict.py
uv run python assignment2/scripts/evaluate.py --pred output_315552057.csv --input assignment2/submission/test_input.csv
```

# Files 
- `q1`: schema graph (Q1)
- `model/dataset.py`, `train.py`, `predict.py`: logistic regression baseline
- `model/gnn_*.py`: RelBench, database, graph, task and training
- `model/relgnn_*.py`, `atomic_routes.py`: from https://github.com/snap-stanford/RelGNN
