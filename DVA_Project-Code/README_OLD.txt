DESCRIPTION

This package predicts whether an MLB umpire will make a wrong ball-or-strike call and
evaluates the model's performance. It combines pybaseball Statcast pitch data with
umpire assignments, cleans and engineers pitch-location and game-context features,
and trains baseline and final machine-learning models.

The project contains notebooks for data retrieval and exploratory data analysis,
feature engineering, baseline XGBoost and neural-network models, a tuned final
XGBoost model, and visualization. Processed data is stored in data/processed as
Parquet and CSV files. Model predictions and summary metrics are stored in
results/raw and results/stats. Static charts are stored in figs, and 2025
prediction outputs are stored in viz.

The target variable, wrong_call, identifies a called strike outside the rule-book
strike zone or a ball inside the zone. The main demo uses 2022-2024 training data
and held-out 2025 data to evaluate umpire-call predictions. The source notebooks
also show exploratory plots, model diagnostics, confusion matrices, precision-recall
and ROC curves, calibration, and 2025 visualization outputs.

INSTALLATION

1. Install Python 3.11 and Jupyter Notebook or JupyterLab. An Anaconda or Miniconda
   environment is recommended.

2. Open a terminal in the project root and create an environment:

   conda create -n umpire-call python=3.11
   conda activate umpire-call

3. Install the required Python packages:

   python -m pip install jupyterlab pandas polars pyarrow numpy scipy matplotlib
   seaborn scikit-learn tqdm pybaseball requests pytz shap xgboost

4. Start Jupyter from the project root:

   jupyter lab

Keep the supplied data/ and results/ folders with the notebooks. All notebook paths
are relative to their containing directory and assume this folder structure.

Data Collection + EDA/data_retrevial.ipynb requires internet access for pybaseball
and the MLB Stats API. Re-running collection can take significant time. The
Final Model/xgb.ipynb is configured to use an NVIDIA GPU through XGBoost
(device="cuda"). GPU execution is optional: to run on CPU, change that setting to
device="cpu", or remove the device argument.

EXECUTION

For the fastest demo, do not re-download or retrain data. The processed datasets,
model predictions, metrics, and visualizations are included.

1. Open viz/viz.ipynb in JupyterLab and run all cells. This loads the supplied
   2025 prediction data and train/test Parquet files and produces the project's
   visual analysis.

2. To inspect the final model, open Final Model/xgb.ipynb and run all cells.
   This trains and tunes an XGBoost classifier, reports train/test log loss,
   ROC-AUC, PR-AUC, and F1, writes diagnostics to figs, and writes predictions
   to results/raw/xgb_predictions_vs_actuals.csv and metrics to
   results/stats/xgb_metrics.csv.

To rebuild the package from source:

1. Run "Data Collection + EDA/data_retrevial.ipynb". Set or keep the requested
   year range in the notebook. This creates Statcast Parquet files in
   data/processed/.
2. Run "Data Collection + EDA/feature_engr.ipynb". This creates the feature and
   response CSV files plus train/test Parquet files under data/processed/.
3. Run one or both notebooks in "Baseline Model Eval/" to compare MLP or XGBoost
   baseline performance.
4. Run "Final Model/xgb.ipynb" and then "viz/viz.ipynb".

DEMO VIDEO

No demo video is included.
