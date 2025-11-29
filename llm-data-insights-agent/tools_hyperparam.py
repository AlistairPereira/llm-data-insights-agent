# tools_hyperparam.py

from typing import List, Dict, Any, Literal
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score, f1_score


ProblemType = Literal["regression", "classification"]


def _regression_search_space(algo: str):
    algo = algo.lower()

    if algo in ("rf", "random_forest", "randomforest"):
        model = RandomForestRegressor(random_state=42)
        params = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        algo_name = "RandomForestRegressor"

    elif algo in ("gbr", "gb", "gradient_boosting"):
        model = GradientBoostingRegressor(random_state=42)
        params = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4],
        }
        algo_name = "GradientBoostingRegressor"

    elif algo in ("ridge",):
        model = Ridge()
        params = {"alpha": np.logspace(-3, 2, 10)}
        algo_name = "Ridge"

    elif algo in ("lasso",):
        model = Lasso()
        params = {"alpha": np.logspace(-3, 1, 10)}
        algo_name = "Lasso"

    elif algo in ("elasticnet", "enet"):
        model = ElasticNet()
        params = {
            "alpha": np.logspace(-3, 1, 10),
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
        algo_name = "ElasticNet"

    elif algo in ("knn", "kneighbors", "k_neighbors"):
        model = KNeighborsRegressor()
        params = {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
        }
        algo_name = "KNeighborsRegressor"

    else:
        # default to RF
        model = RandomForestRegressor(random_state=42)
        params = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        algo_name = "RandomForestRegressor (default)"

    # for older sklearn, use neg MSE and compute RMSE manually
    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    metric_name = "rmse"
    higher_is_better = False

    return model, params, scoring, metric_name, higher_is_better, algo_name


def _classification_search_space(algo: str):
    algo = algo.lower()

    if algo in ("rf", "random_forest", "randomforest"):
        model = RandomForestClassifier(random_state=42)
        params = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        algo_name = "RandomForestClassifier"

    elif algo in ("gboost", "gbr", "gradient_boosting"):
        model = GradientBoostingClassifier(random_state=42)
        params = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4],
        }
        algo_name = "GradientBoostingClassifier"

    elif algo in ("logreg", "logistic", "lr"):
        model = LogisticRegression(max_iter=1000)
        params = {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
        }
        algo_name = "LogisticRegression"

    elif algo in ("knn", "kneighbors", "k_neighbors"):
        model = KNeighborsClassifier()
        params = {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
        }
        algo_name = "KNeighborsClassifier"

    else:
        model = RandomForestClassifier(random_state=42)
        params = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        algo_name = "RandomForestClassifier (default)"

    # we'll optimise accuracy by default
    scoring = make_scorer(accuracy_score)
    metric_name = "accuracy"
    higher_is_better = True

    return model, params, scoring, metric_name, higher_is_better, algo_name


def run_hyperparam_experiments(
    X,
    y,
    problem_type: ProblemType,
    algo_list: List[str] | None = None,
    n_iter: int = 15,
    cv: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run hyperparameter search over one or more algorithms.

    Returns a dict:
    {
      "problem_type": ...,
      "experiments": [
        {
          "algo_key": "rf",
          "algo_name": "RandomForestRegressor",
          "metric_name": "rmse",
          "metric_value": 3.45,
          "higher_is_better": false,
          "best_params": {...}
        },
        ...
      ],
      "best_experiment": {...}
    }
    """
    if algo_list is None:
        if problem_type == "regression":
            algo_list = ["rf", "gbr", "ridge", "lasso"]
        else:
            algo_list = ["rf", "gboost", "logreg", "knn"]

    experiments: List[Dict[str, Any]] = []

    for algo in algo_list:
        print(f"[Hyperparam] Running search for algo = {algo} ...")

        if problem_type == "regression":
            model, params, scoring, metric_name, higher_is_better, algo_name = _regression_search_space(
                algo
            )
        else:
            model, params, scoring, metric_name, higher_is_better, algo_name = _classification_search_space(
                algo
            )

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
        )

        search.fit(X, y)

        # For regression, scoring is negative MSE; convert to RMSE
        if problem_type == "regression":
            mse_cv = -float(search.best_score_)
            metric_value = float(mse_cv ** 0.5)  # RMSE
        else:
            metric_value = float(search.best_score_)  # e.g., accuracy

        exp = {
            "algo_key": algo,
            "algo_name": algo_name,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "higher_is_better": higher_is_better,
            "best_params": search.best_params_,
        }
        experiments.append(exp)

    # pick best experiment
    if not experiments:
        return {
            "problem_type": problem_type,
            "experiments": [],
            "best_experiment": None,
        }

    if problem_type == "regression":
        best = min(experiments, key=lambda e: e["metric_value"])
    else:
        best = max(experiments, key=lambda e: e["metric_value"])

    results: Dict[str, Any] = {
        "problem_type": problem_type,
        "experiments": experiments,
        "best_experiment": best,
    }
    return results
