"""
sample_generator.py

This script is used to generate samples directly from the main dataset. 
These samples are used to test the `/predict/file` route. 
To run it, enter this in your command line:
```
python -m Datasets.sample_generator.py
```
"""

import pandas as pd
from pathlib import Path

DATASET_PATH = Path("Datasets","SDSS_DR18.csv")
OUTPUT_PATH = Path("Datasets","samples.csv")

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["objid", "specobjid", "run", "rerun", "camcol",
                 "field", "plate", "mjd", "fiberid"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df["class"] = df["class"].map({"GALAXY": 0, "STAR": 1, "QSO": 2})

    df = df[["ra", "dec", "redshift","psfMag_r",  "u", "g", "r", "i", "z", "class"]].copy()

    return df


def stratified_sample(df: pd.DataFrame, total_samples: int, class_col: str = "class", random_state: int = 42) -> pd.DataFrame:
    class_counts = df[class_col].value_counts(normalize=True)
    class_n = (class_counts * total_samples).round().astype(int)

    diff = total_samples - class_n.sum()
    class_n.iloc[0] += diff

    return df.groupby(class_col, group_keys=False).apply(
        lambda x: x.sample(n=min(class_n[x.name], len(x)), random_state=random_state)
    )


df_raw = pd.read_csv(DATASET_PATH)

df_processed = preprocess(df_raw)

sample = stratified_sample(df_processed, total_samples=100)

sample = sample.drop(columns=["class"])

sample.to_csv(OUTPUT_PATH,index=False)