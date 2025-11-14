# tools_models.py

import pandas as pd
from typing import Tuple, Dict, Any, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
)


def choose_target_column(df: pd.DataFrame, target_col: Optional[str] = None) -> str:
    """
    If the user provides a target column, validate it.
    Otherwise, default to the last column in the dataframe.
    """
    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")
        return target_col

    # Fallback: use the last column as the target
    return df.columns[-1]


def infer_problem_type(y: pd.Series) -> str:
    """
    Decide whether this is a regression or classification problem.

    - If the target is non-numeric  -> classification
    - If the target is numeric:
        - If it has few unique values (<= 10) -> classification (like 0/1/2 labels)
        - Else                               -> regression
    """
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    if y.nunique() <= 10:
        return "classification"

    return "regression"


def prepare_features_and_target(
    df: pd.DataFrame, target_col: str
) -> Tuple[Any, Any, str, Optional[OrdinalEncoder], Optional[LabelEncoder]]:
    """
    Prepare X (features) and y (target) for modeling:

    1. Split df into X and y
    2. Infer problem type (regression / classification)
    3. Encode target for classification using LabelEncoder
    4. Ordinal-encode categorical features
    5. Scale ONLY numeric features using StandardScaler
    """
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Decide if regression vs classification
    problem_type = infer_problem_type(y)

    # Encode target for classification
    label_encoder: Optional[LabelEncoder] = None
    if problem_type == "classification":
        label_encoder = LabelEncoder()
        y_processed = label_encoder.fit_transform(y.astype(str))
    else:
        y_processed = y.astype(float)

    # Identify categorical and numeric feature columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.select_dtypes(include="number").columns

    # Ordinal encode categorical features (turn them into numbers)
    ord_encoder: Optional[OrdinalEncoder] = None
    if len(cat_cols) > 0:
        ord_encoder = OrdinalEncoder()
        X[cat_cols] = ord_encoder.fit_transform(X[cat_cols].astype(str))

    # Scale ONLY numeric features
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    # Return numpy array for X, processed y, and encoders/problem type
    return X.values, y_processed, problem_type, ord_encoder, label_encoder


def train_model(
    X, y, problem_type: str
) -> Tuple[Any, Any, Any]:
    """
    Train a simple baseline model:

    - regression     -> RandomForestRegressor
    - classification -> RandomForestClassifier

    Returns:
        model, y_test, y_pred
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if problem_type == "regression":
        model = RandomForestRegressor(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, y_test, y_pred


def evaluate_model(
    y_true, y_pred, problem_type: str
) -> Dict[str, Any]:
    """
    Evaluate the model based on the problem type and
    return a clean dictionary with metrics for the LLM.
    """
    if problem_type == "regression":
        # Older sklearn versions don't support squared=False, so compute RMSE manually
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_true, y_pred)
        return {
            "type": "regression",
            "rmse": round(rmse, 3),
            "r2": round(r2, 3),
            "n_test_samples": len(y_true),
        }

    # classification
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return {
        "type": "classification",
        "accuracy": round(acc, 3),
        "f1_weighted": round(f1, 3),
        "n_test_samples": len(y_true),
    }
