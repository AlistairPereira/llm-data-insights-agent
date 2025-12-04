import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any
import os

import os

def get_dataset_key_from_path(path: str) -> str:
    """
    Turn 'sample_data/cars.csv' -> 'cars'
    Turn 'iris.csv'              -> 'iris'
    Turn 'My Data.csv'           -> 'my_data'
    """
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name.replace(" ", "_").lower()


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

def compute_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a correlation matrix for numeric columns.

    Returns a dict like:
    {
      "numeric_columns": [...],
      "corr_matrix": { col1: {col1: 1.0, col2: 0.5, ...}, ... }
    }
    """
    result: Dict[str, Any] = {}

    num_cols = df.select_dtypes(include="number").columns
    result["numeric_columns"] = list(num_cols)

    if len(num_cols) < 2:
        # Not enough numeric columns to compute correlations
        result["corr_matrix"] = {}
        return result

    corr = df[num_cols].corr().round(3)
    result["corr_matrix"] = corr.to_dict()

    return result


def plot_corr_heatmap(df: pd.DataFrame, out_path: str = "outputs/corr_heatmap.png") -> str:
    """
    Create and save a simple correlation heatmap for numeric columns.
    Returns the path if created, or an empty string if not.
    """
    os.makedirs("outputs", exist_ok=True)

    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) < 2:
        return ""

    corr = df[num_cols].corr()

    plt.figure(figsize=(8, 6))
    # Simple matplotlib heatmap (no seaborn)
    im = plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
    plt.yticks(range(len(num_cols)), num_cols)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return out_path

def detect_outliers_iqr(df: pd.DataFrame, multiplier: float = 1.5) -> Dict[str, Any]:
    """
    Detect outliers in numeric columns using the IQR rule.

    For each numeric column:
      - Compute Q1, Q3, IQR
      - Define lower = Q1 - multiplier * IQR
      - Define upper = Q3 + multiplier * IQR
      - Count how many values are below/above those bounds

    Returns a dict like:
    {
      "multiplier": 1.5,
      "per_column": {
        "price": {
          "q1": ...,
          "q3": ...,
          "lower_bound": ...,
          "upper_bound": ...,
          "n_outliers": ...,
          "outlier_ratio": ...
        },
        ...
      }
    }
    """
    result: Dict[str, Any] = {"multiplier": multiplier, "per_column": {}}

    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        return result

    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        mask_out = (series < lower) | (series > upper)
        n_outliers = int(mask_out.sum())
        n_total = int(series.shape[0])
        ratio = round(n_outliers / n_total, 3) if n_total > 0 else 0.0

        result["per_column"][col] = {
            "q1": round(float(q1), 3),
            "q3": round(float(q3), 3),
            "lower_bound": round(float(lower), 3),
            "upper_bound": round(float(upper), 3),
            "n_outliers": n_outliers,
            "n_total": n_total,
            "outlier_ratio": ratio,
        }

    return result


