# # tools_model.py

from typing import Tuple, Literal, Dict
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    f1_score,
)

ProblemType = Literal["regression", "classification"]


def choose_target_column(df: pd.DataFrame, target_hint: str | None) -> str:
    """
    If user gives a target and it exists, use it.
    Otherwise:
      - pick the last numeric column if available
      - else pick the last column.
    """
    if target_hint and target_hint in df.columns:
        print(f"[Model Agent] Using user-provided target column: {target_hint}")
        return target_hint

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        print(f"[Model Agent] No valid target hint. Using last numeric column: {num_cols[-1]}")
        return num_cols[-1]

    print(f"[Model Agent] No numeric columns. Using last column as target: {df.columns[-1]}")
    return df.columns[-1]


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series, ProblemType, object, object, list[str]]:
    """
    Prepares X, y using your unsupervised-style logic:

    - Separate target column.
    - Categorical features:
        * if number of cat columns > 4 → LabelEncode each column separately
        * else (1–4 cat cols) → OrdinalEncode all of them together
    - Scale numeric features with StandardScaler.
    - Detect problem type:
        * classification if y is object or has <=10 unique values
        * else regression
    - For classification, label-encode y.

    Returns:
        X              : processed feature DataFrame
        y              : target (possibly encoded)
        problem_type   : "regression" or "classification"
        cat_encoder    : dict of LabelEncoders OR a single OrdinalEncoder OR None
        y_encoder      : LabelEncoder for y (classification) or None
        feature_names  : list of feature names after preprocessing
    """

    # Separate target
    y = df[target_col]
    X = df.drop(columns=[target_col]).copy()

    # Identify categorical and numeric columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()

    print(f"[Model Agent] Categorical columns: {cat_cols}")
    print(f"[Model Agent] Numeric columns: {num_cols}")

    # ---------------------------
    # Encode categorical features
    # ---------------------------
    cat_encoder: object | None = None

    if len(cat_cols) > 4:
        print("[Model Agent] Using LabelEncoder for categorical features (per column)...")
        encoders: Dict[str, LabelEncoder] = {}
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        cat_encoder = encoders
    else:
        if len(cat_cols) > 0:
            print("[Model Agent] Using OrdinalEncoder for categorical features...")
            ord_enc = OrdinalEncoder()
            X[cat_cols] = ord_enc.fit_transform(X[cat_cols].astype(str))
            cat_encoder = ord_enc
        else:
            print("[Model Agent] No categorical columns found.")
            cat_encoder = None

    # ---------------------------
    # Scale numeric features
    # ---------------------------
    if len(num_cols) > 0:
        print("[Model Agent] Scaling numeric features...")
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
    else:
        print("[Model Agent] No numeric columns found.")
        scaler = None  # kept for completeness, not returned

    # ---------------------------
    # Detect problem type
    # ---------------------------
    if y.dtype == "object" or y.nunique() <= 10:
        problem_type: ProblemType = "classification"
        print("[Model Agent] Detected classification problem.")
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y.astype(str))
    else:
        problem_type = "regression"
        print("[Model Agent] Detected regression problem.")
        y_encoder = None

    feature_names = list(X.columns)
    return X, y, problem_type, cat_encoder, y_encoder, feature_names


# ---------------------------
#  Model selection helpers
# ---------------------------

def _get_regressor(algo: str):
    algo = algo.lower()
    if algo in ("linear", "linreg", "lr"):
        print("[Model Agent] Using LinearRegression...")
        return LinearRegression()
    if algo in ("gbr", "gb", "gradient_boosting"):
        print("[Model Agent] Using GradientBoostingRegressor...")
        return GradientBoostingRegressor(random_state=42)
    if algo in ("knn", "kneighbors", "k_neighbors"):
        print("[Model Agent] Using KNeighborsRegressor...")
        return KNeighborsRegressor()
    # default
    print("[Model Agent] Using RandomForestRegressor (default)...")
    return RandomForestRegressor(random_state=42)


def _get_classifier(algo: str):
    algo = algo.lower()
    if algo in ("logreg", "logistic", "lr"):
        print("[Model Agent] Using LogisticRegression...")
        return LogisticRegression(max_iter=1000)
    if algo in ("gboost", "gbr", "gradient_boosting"):
        print("[Model Agent] Using GradientBoostingClassifier...")
        return GradientBoostingClassifier(random_state=42)
    if algo in ("knn", "kneighbors", "k_neighbors"):
        print("[Model Agent] Using KNeighborsClassifier...")
        return KNeighborsClassifier()
    # default
    print("[Model Agent] Using RandomForestClassifier (default)...")
    return RandomForestClassifier(random_state=42)


def train_model(
    X,
    y,
    problem_type: ProblemType,
    algo: str = "random_forest",
):
    """
    Split into train/test and train a model.

    Regression algos:
        - 'linear', 'linreg', 'lr'
        - 'random_forest', 'rf' (default)
        - 'gbr', 'gb', 'gradient_boosting'
        - 'knn', 'kneighbors', 'k_neighbors'

    Classification algos:
        - 'logreg', 'logistic', 'lr'
        - 'random_forest', 'rf' (default)
        - 'gboost', 'gbr', 'gradient_boosting'
        - 'knn', 'kneighbors', 'k_neighbors'
    """
    print("[Model Agent] Splitting into train/test ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if problem_type == "regression":
        model = _get_regressor(algo)
    else:
        model = _get_classifier(algo)

    print("[Model Agent] Fitting model ...")
    model.fit(X_train, y_train)
    print("[Model Agent] Predicting on test set ...")
    y_pred = model.predict(X_test)

    return model, y_test, y_pred


def evaluate_model(y_test, y_pred, problem_type: ProblemType) -> dict:
    """
    Compute metrics for regression or classification.
    Uses manual RMSE computation to avoid sklearn 'squared' argument issues.
    """
    if problem_type == "regression":
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        return {
            "type": "regression",
            "rmse": float(rmse),
            "r2": float(r2_score(y_test, y_pred)),
        }
    else:
        return {
            "type": "classification",
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        }
