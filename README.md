**LLM Data Insights Agent**
A multi-agent, tool-using system that can:
Run EDA with plots, correlations & outliers
Train & evaluate supervised models (regression / classification)
Run unsupervised learning (PCA, KMeans, DBSCAN, Agglomerative)
Perform hyperparameter tuning
Use a Router Agent to decide which agent to call based on a natural-language request
Generate a full PDF analytics report
Ask a local LLM (Ollama / llama3.2) to explain all results in plain English

**✨ Features**


**🧹 EDA Agent**
Cleans dataset (drop empty cols, forward/backward fill)
Summary stats, missing %, dtypes, top categories
Correlation matrix + correlation heatmap (corr_heatmap.png)
Numeric histograms
IQR-based outlier detection
LLM insights explained by a local mode
Saves all outputs inside:
outputs/<dataset>/

**🤖 Supervised Model Agent**

**Supports:**
Linear Regression
Random Forest
Gradient Boosting
Logistic Regression
KNN (Regressor/Classifier)

**Features:**
Uses cleaned data + EDA summary
Prepares features/target (encoding + scaling)
Trains the selected model
Evaluates performance (RMSE, R², Accuracy, F1)
LLM explanation of model + EDA together

**Saves:**
model_report.json
model_insights.txt

**🧩 Unsupervised Model Agent**

**Supports:**
PCA
KMeans
DBSCAN
Agglomerative Clustering

**LLM explains:**
PCA variance and component meaning
Cluster structures and metrics
Correlation insights
Outlier influence
Practical use cases

**Saves:**
unsupervised_insights.txt

**⚙️ Hyperparameter Tuning Agent**
Auto-detects algorithm from last model run
Or user can manually specify algorithms
Uses RandomizedSearchCV

**Saves:**
hyperparam_results.json
hyperparam_insights.txt

**🧭 Router Agent**

**Understands natural-language commands such as:**

"Run EDA on cars.csv"
"Train a linear model to predict price"
"Cluster using kmeans with 3 groups"
"Tune the model on iris.csv"
Chooses correct agent → executes → saves results.

**📄 Report Agent (PDF Generator)**

Creates a clean PDF including:
EDA summary + plots
Model results + insights
Hyperparameter tuning summary
Unsupervised insights

**Outputs:**

outputs/report_cars.pdf
outputs/report_iris.pdf

**Uses dataset folders:**

outputs/cars/
outputs/iris/

**🏗️ Project Structure**
llm-data-insights-agent/
├── eda_agent.py
├── model_agent.py
├── unsupervised_model_agent.py
├── hyperparam_agent.py
├── router_agent.py
├── report_agent.py
│
├── tools_data.py
├── tools_model.py
├── tools_unsupervised.py
├── tools_hyperparam.py
├── llm_local.py
│
├── sample_data/
│   ├── cars.csv
│   └── iris.csv
│
├── outputs/
│   ├── cars/
│   ├── iris/
│   ├── report_cars.pdf
│   ├── report_iris.pdf
│   └── ...
│
├── requirements.txt
└── .gitignore
