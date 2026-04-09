import pandas as pd
from pathlib import Path

DATASET_PATH = Path("Datasets","SDSS_DR18.csv")
OUTPUT_PATH = Path("Datasets","samples.csv")

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop irrelevant columns (ignore if already absent)
    drop_cols = ["objid", "specobjid", "run", "rerun", "camcol",
                 "field", "plate", "mjd", "fiberid"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Map string classes to numeric labels
    df["class"] = df["class"].map({"GALAXY": 0, "STAR": 1, "QSO": 2})

    # Feature reduction
    df = df[["ra", "dec", "redshift", "u", "g", "r", "i", "z", "psfMag_r", "class"]].copy()

    # Feature engineering — color contrast columns
    df["u_g_color"] = df["u"] - df["g"]
    df["g_r_color"] = df["g"] - df["r"]
    df["r_i_color"] = df["r"] - df["i"]
    df["i_z_color"] = df["i"] - df["z"]
    df = df.drop(columns=["u", "g", "r", "i", "z"])

    # Move `class` to the end
    popped_class = df.pop("class")
    df.insert(len(df.columns), "class", popped_class)

    return df


def stratified_sample(df: pd.DataFrame, total_samples: int, class_col: str = "class", random_state: int = 42) -> pd.DataFrame:
    class_counts = df[class_col].value_counts(normalize=True)
    class_n = (class_counts * total_samples).round().astype(int)

    # Fix rounding drift
    diff = total_samples - class_n.sum()
    class_n.iloc[0] += diff

    return df.groupby(class_col, group_keys=False).apply(
        lambda x: x.sample(n=min(class_n[x.name], len(x)), random_state=random_state)
    )


# --- Usage ---
df_raw = pd.read_csv(DATASET_PATH)

df_processed = preprocess(df_raw)

sample = stratified_sample(df_processed, total_samples=100)

sample = sample.drop(columns=["class"])

sample.to_csv(OUTPUT_PATH,index=False)