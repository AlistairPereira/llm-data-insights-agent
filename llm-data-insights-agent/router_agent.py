
import sys
import json

from llm_local import run_llm
from eda_agent import run_eda_agent
from model_agent import run_model_agent
from unsupervised_model_agent import run_unsupervised_agent
from hyperparam_agent import run_hyperparam_agent


# -----------------------------------------------------------
#  RULE-BASED OVERRIDE (STRONGER THAN LLM)
# -----------------------------------------------------------
TUNING_KEYWORDS = [
    "tune", "tuning", "hyperparam", "hyperparameter", "fine tune",
    "fine-tune", "optimize", "best parameters", "find best",
    "improve model", "optimize model"
]


def user_wants_tuning(msg: str) -> bool:
    msg_lower = msg.lower()
    return any(k in msg_lower for k in TUNING_KEYWORDS)


# -----------------------------------------------------------
#  LLM ROUTER PROMPT
# -----------------------------------------------------------
def build_router_prompt(user_message: str) -> str:
    return f"""
You are a routing assistant for a data analysis system.

The user will describe what they want to do with a dataset.

You have access to four high-level tools:

1) "eda"          - Exploratory data analysis.
2) "model"        - Supervised model training.
3) "unsupervised" - PCA or clustering.
4) "hyperparam"   - Train + hyperparameter tuning.

RULES:
- If the user mentions any of the following: 
  "tune", "tuning", "hyperparameter", "fine-tune", 
  "optimize", "best parameters", "improve model",
  then ALWAYS pick: "hyperparam".

- If they mention only training → "model".
- If they mention PCA or clusters → "unsupervised".
- If they mention EDA → "eda".

Supervised algorithms map as:
  linear model → linear
  random forest → random_forest
  gradient boosting / gboost → gbr
  logistic regression → logreg
  knn → knn

Return STRICT JSON:

{{
  "action": "eda" | "model" | "unsupervised" | "hyperparam",
  "file_path": "<path>",
  "target_column": "<column or null>",
  "model_algo": "<linear|random_forest|gbr|logreg|knn|null>",
  "algorithm": "<pca|kmeans|dbscan|agglomerative|null>",
  "n_components": <int or null>,
  "n_clusters": <int or null>,
  "eps": <float or null>,
  "min_samples": <int or null>,
  "linkage": "<string or null>"
}}

User message:
\"\"\"{user_message}\"\"\"
    """


# -----------------------------------------------------------
#  MAIN ROUTING LOGIC
# -----------------------------------------------------------
def route_user_request(user_message: str) -> None:

    # 1) RULE-BASED OVERRIDE (always wins)
    if user_wants_tuning(user_message):
        print("[Router] 🔥 Rule-based override: user wants hyperparameter tuning.")
        action = "hyperparam"
        # Need file + target → LLM will parse that part only
        prompt = build_router_prompt(user_message)
        router_response = run_llm(prompt)

        try:
            start = router_response.find("{")
            end = router_response.rfind("}") + 1
            route = json.loads(router_response[start:end])
        except:
            print("[Router] Failed to parse JSON in tuning override.")
            return

        # Force action to hyperparam
        route["action"] = "hyperparam"

    else:
        # Normal LLM routing
        prompt = build_router_prompt(user_message)
        print("[Router] Asking LLM to route the request ...")
        router_response = run_llm(prompt)

        try:
            start = router_response.find("{")
            end = router_response.rfind("}") + 1
            route = json.loads(router_response[start:end])
        except Exception as e:
            print("[Router] Failed to parse routing JSON:", e)
            return

    # Extract
    action = route.get("action")
    file_path = route.get("file_path")
    target_col = route.get("target_column")
    model_algo = route.get("model_algo")
    algo = route.get("algorithm")
    n_components = route.get("n_components")
    n_clusters = route.get("n_clusters")
    eps = route.get("eps")
    min_samples = route.get("min_samples")
    linkage = route.get("linkage")

    # Normalize
    def norm(x):
        if isinstance(x, str) and x.lower() in ("null", "none", ""):
            return None
        return x

    file_path = norm(file_path)
    target_col = norm(target_col)
    model_algo = norm(model_algo) or "random_forest"
    algo = norm(algo)
    n_components = int(n_components) if n_components else None
    n_clusters = int(n_clusters) if n_clusters else None
    eps = float(eps) if eps else None
    min_samples = int(min_samples) if min_samples else None
    linkage = norm(linkage)

    # ---------------- ROUTER ACTIONS -----------------

    if action == "eda":
        run_eda_agent(file_path, use_llm=True)

    elif action == "model":
        run_model_agent(file_path, target_col=target_col, algo=model_algo)

    elif action == "unsupervised":
        if algo is None:
            algo = "kmeans"
        run_unsupervised_agent(
            file_path=file_path,
            algo=algo,
            n_components=n_components,
            n_clusters=n_clusters,
            eps=eps,
            min_samples=min_samples,
            linkage=linkage,
        )

    elif action == "hyperparam":
        print("[Router] Running baseline model before tuning...")
        run_model_agent(file_path, target_col=target_col, algo=model_algo)

        print("[Router] Running hyperparameter tuning...")
        run_hyperparam_agent(
            file_path=file_path,
            target_col=target_col,
            algo_list=None
        )

    else:
        print(f"[Router] Unknown action: {action}")


# -----------------------------------------------------------
#  CLI ENTRY
# -----------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python router_agent.py "<your_request>"')
        sys.exit(1)

    user_msg = " ".join(sys.argv[1:])
    route_user_request(user_msg)
