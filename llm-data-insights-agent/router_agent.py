# router_agent.py

import sys
import json

from llm_local import run_llm
from eda_agent import run_eda_agent
from model_agent import run_model_agent


def build_router_prompt(user_message: str) -> str:
    """
    Ask the LLM to decide which action to take based on the user request.
    It must respond with STRICT JSON.
    """
    return f"""
You are a routing assistant for a data analysis system.

The user will describe what they want to do with a dataset.

You have access to two tools:

1) "eda"   - Run exploratory data analysis (EDA) on a CSV file.
2) "model" - Train and evaluate a machine learning model on a CSV file.

Your job:
- Read the user message.
- Decide which single action to take: "eda" or "model".
- Extract the file path mentioned by the user (for example: "sample_data/cars.csv").
- Optionally, if the user mentions a target column (like "price"), include it.
- If no target column is mentioned and action is "model", set target_column to null.

Return ONLY a single JSON object with this structure:

{{
  "action": "eda" or "model",
  "file_path": "<path_to_csv>",
  "target_column": "<name or null>"
}}

User message:
\"\"\"{user_message}\"\"\"
    """


def route_user_request(user_message: str) -> None:
    """
    1. Ask the LLM to decide which tool to use.
    2. Parse the JSON.
    3. Call the correct agent with the right arguments.
    """
    prompt = build_router_prompt(user_message)
    print("[Router] Asking LLM to route the request ...")
    router_response = run_llm(prompt)

    print("\n[Router] Raw LLM routing response:")
    print(router_response)
    print()

    # Try to parse JSON from the LLM output
    try:
        # In case the model outputs extra text, try to find the JSON part
        start = router_response.find("{")
        end = router_response.rfind("}") + 1
        json_str = router_response[start:end]

        route = json.loads(json_str)
    except Exception as e:
        print("[Router] Failed to parse routing JSON:", e)
        print("[Router] Aborting.")
        return

    action = route.get("action")
    file_path = route.get("file_path")
    target_col = route.get("target_column")

    print(f"[Router] Parsed action: {action}")
    print(f"[Router] File path: {file_path}")
    print(f"[Router] Target column: {target_col}")

    if not file_path:
        print("[Router] No file_path provided in routing output. Aborting.")
        return

    if action == "eda":
        print("[Router] Calling EDA Agent ...")
        run_eda_agent(file_path, use_llm=True)

    elif action == "model":
        print("[Router] Calling Model Agent ...")
        # If target_col is null in JSON, Python sees None
        if isinstance(target_col, str) and target_col.lower() in ("null", "none", ""):
            target_col = None
        run_model_agent(file_path, target_col=target_col)

    else:
        print(f"[Router] Unknown action '{action}'. Aborting.")


if __name__ == "__main__":
    # Usage:
    #   python router_agent.py "do EDA on sample_data/cars.csv"
    #   python router_agent.py "train a model on sample_data/cars.csv to predict price"
    if len(sys.argv) < 2:
        print("Usage: python router_agent.py \"<your_request_here>\"")
        sys.exit(1)

    user_msg = " ".join(sys.argv[1:])
    route_user_request(user_msg)
