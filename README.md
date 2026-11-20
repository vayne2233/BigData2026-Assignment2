# BigData Assignment 2: Predicting GitHub User Activity

## Environment
- OS: Win11, CPU only (no GPU)
- Python 3.10 (managed by uv)
- torch 2.14+cpu, torch-geometric 2.8.1, relbench, pytorch-frame, pyg-lib
- Exact package versions are pinned in `uv.lock`

## Data Structure
- GH Archive: https://www.gharchive.org/
- Period used: 2026-10-25 00:00 ~ 2026-11-20 19:00 (UTC)

## Setup
Requires [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/vayne2233/BigData2026-Assignment2
cd BigData2026-Assignment2

# course repo
git clone https://github.com/nycubigdata-2026fall/BigData2026-Assignment2 assignment2

uv sync  

# pyg-lib is not on PyPI; install the build matching torch 2.14 (CPU)j
uv pip install pyg-lib --find-links https://data.pyg.org/whl/torch-2.14.0+cpu.html
```


> **Windows note:** if the project path contains non-ASCII characters (e.g. Chinese), 
> Python 3.10 may fail with `UnicodeDecodeError: 'cp950'`. Clone into a path with 
> only English characters, e.g. `C:\asg2`.

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
