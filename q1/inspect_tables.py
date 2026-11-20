# Inspect the 8 tables: row count and unique values per 
# column (to find PK/FK candidates).

import pandas as pd
from pathlib import Path

for f in Path("data/db").glob("*.parquet"):
    df = pd.read_parquet(f)
    print(f.stem, len(df), df.nunique().to_dict())
