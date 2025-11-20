# LLM Data Insights Agent

A small **multi-agent, tool-using system** that can:

- Run **EDA with plots, correlations & outliers**
- Train & evaluate **supervised models** (regression / classification)
- Run **unsupervised learning** (PCA, KMeans, DBSCAN, Agglomerative)
- Use a **Router Agent** to decide which agent to call based on a natural-language request  
- Ask a **local LLM (Ollama / llama3.2)** to explain all results in plain English

---

## ✨ Features

- 🧹 **EDA Agent**
  - Cleaning (drop empty cols, forward/backward fill)
  - Summary stats, missing %, dtypes, top categories
  - Correlation matrix + correlation heatmap (`corr_heatmap.png`)
  - IQR-based **outlier detection** for numeric columns
  - EDA insights explained by a local LLM

- 🤖 **Supervised Model Agent**
  - Uses cleaned data + EDA summary
  - Prepares features/target (encoding + scaling)  
  - Trains model (baseline version now; easily extendable)
  - Evaluates with metrics (e.g. RMSE, R², etc.)
  - Uses LLM to explain model performance, feature importance & EDA together

- 🧩 **Unsupervised Model Agent**
  - Shared preprocessing for unsupervised tasks  
  - Algorithms:
    - PCA (components + explained variance)
    - KMeans (labels, centers, metrics)
    - DBSCAN (clusters, noise ratio)
    - Agglomerative clustering
  - LLM explains structure, clusters & metrics using EDA + algorithm report

- 🧭 **Router Agent**
  - Takes a **natural-language request**
  - Asks LLM to output a strict JSON:
    - `action` → `"eda" | "model" | "unsupervised"`
    - `file_path`, `target_column`, `algorithm`, and algorithm params
  - Calls the correct agent with parsed parameters

- 🧠 **Local LLM integration (Ollama)**
  - Uses `ollama run llama3.2` via `subprocess`
  - All heavy data work is done in Python tools;  
    LLM is used only for **reasoning & explanation**.

---

## 🏗️ Project Structure

```text
llm-data-insights-agent/
├── eda_agent.py                 # EDA Agent (summary + correlations + outliers + plots + LLM)
├── model_agent.py               # Supervised model agent
├── unsupervised_model_agent.py  # Unsupervised model agent (PCA / KMeans / DBSCAN / Agglomerative)
├── router_agent.py              # Router agent: chooses which agent to call based on user request
├── tools_data.py                # Low-level EDA tools (cleaning, summary, correlations, outliers, plots)
├── tools_model.py               # Supervised model tools (prep, training, metrics, feature importances)
├── tools_unsupervised.py        # Unsupervised tools (prep, PCA, KMeans, DBSCAN, Agglomerative)
├── llm_local.py                 # Thin wrapper around `ollama run <model>`
├── sample_data/
│   └── cars.csv                 # Example dataset
├── outputs/
│   ├── histograms.png
│   ├── corr_heatmap.png
│   ├── eda_insights.txt
│   ├── model_insights.txt
│   └── unsupervised_insights.txt
├── requirements.txt
└── .gitignore
