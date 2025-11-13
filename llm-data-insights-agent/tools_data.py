import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any
import os


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df

def quick_eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    summary["shape"] = df.shape
    summary["dtypes"] = df.dtypes.astype(str).to_dict()
    missing_perc = df.isna().mean() * 100
    summary["missing_perc"] = missing_perc.round(2).to_dict()
    numeric_desc = df.describe(include="number").round(3)
    summary["describe_num"] = numeric_desc.to_dict()

    cat_cols = df.select_dtypes(include="object").columns[:3]
    top_categories = {}
    for col in cat_cols:
        top_categories[col] = df[col].value_counts(dropna=True).head(5).to_dict()

    summary["top_categories"] = top_categories
    return summary

def plot_numeric_hist(df: pd.DataFrame, out_path: str = "outputs/histograms.png") -> str:
    os.makedirs("outputs", exist_ok=True)
    num_cols = df.select_dtypes(include="number").columns[:4]
    if len(num_cols) == 0:
        return ""
    df[num_cols].hist(figsize=(10, 6))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
