# Description

This package predicts wehter an MLB umpire will make a wrong call, evaluates how differnt models preform and visulizes the result. It uses Statcast pitch data with umpire assignments.

The target variable "wrong_call" marks when a true strike is called a ball or when a true ball is called a strike. The MLB season 2022 was used for intial model baselining and chosing a final model. Then the for the chosen model (XGBoost) the 2022-2024 seasons were used for training and the 2025 season was held for testing and evaulations.

The code is organized in a series of Jupyter notebooks. All notebook paths are relative to the notebooks own folder, so keep this structure.

### Repo Structure
#### After everything has run

 ```
📦Code
 ┣ 📂Baseline Model Eval
 ┃ ┣ 📜baseline_model_nn.ipynb
 ┃ ┗ 📜baseline_model_xgb.ipynb
 ┣ 📂data
 ┃ ┗ 📂processed
 ┃ ┃ ┣ 📂features
 ┃ ┃ ┣ 📂response_cols
 ┃ ┃ ┣ 📂test
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┣ 📜statcast_2022.parquet
 ┃ ┃ ┣ 📜statcast_2023.parquet
 ┃ ┃ ┣ 📜statcast_2024.parquet
 ┃ ┃ ┗ 📜statcast_2025.parquet
 ┣ 📂Data Collection + EDA
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┗ 📜api_scraper.cpython-311.pyc
 ┃ ┣ 📜api_scraper.py
 ┃ ┣ 📜data_retrevial.ipynb
 ┃ ┣ 📜eda_mmcdowell.ipynb
 ┃ ┣ 📜eda_vfitz.ipynb
 ┃ ┗ 📜feature_engr.ipynb
 ┣ 📂Extra
 ┃ ┗ 📜polars-cheatsheet.pdf
 ┣ 📂figs
 ┣ 📂Final Model
 ┃ ┗ 📜xgb.ipynb
 ┣ 📂results
 ┃ ┣ 📂raw
 ┃ ┗ 📂stats
 ┣ 📂viz
 ┃ ┣ 📜2025_preds_data.csv
 ┃ ┣ 📜2025_preds_data.parquet
 ┃ ┣ 📜2025_preds_summary.csv
 ┃ ┗ 📜viz.ipynb
 ┣ 📜README.md
```

# Installation

Requires Python 3.11 (Option A installs it for you). Chose either of the enviroment options (our team chose option A), run from the project root.

### 0. Unzip submission and open a terminal in the project root (make sure you can see README.md and reuirements.txt)
```
 cd path/to/unzipped/project/folder
```



## Option A: Conda (Miniconda or Anaconda)
```
conda create -n ENV_NAME python=3.11
conda activate ENV_NAME
```

## Option B: Virtual enviroment (venv)

### Install python 3.11 (if necessary):
#### Windows (PowerShell):
```
winget install Python.Python.3.11
```
#### macOS:
```
 brew install python@3.11
```
#### Linux:
```
 sudo apt install python3.11
```
### Create Enviroment:

#### Windows (PowerShell):
```
 py -3.11 -m venv venv
 ```
#### macOS/Linux:
 ```
 python3.11 -m venv venv
```
### Activate:
#### Windows (PowerShell):
```
 venv\Scripts\Activate.ps1
 ```
#### macOS/Linux: 
 ```
source venv/bin/activate
```
## For both Options:

### 1. Install required packages (install all packages from scratch takes ~30 min):

```
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
```
### 2. Start jupyter lab from project root:

```
jupyter lab
```

#### Notes:
- Data reterival can take between 1-2 hrs.
- The final model (Model/xgb.ipynb) is set to use an NVIDA GPU
    - To run on a CPU change param: device="cpu"

# Execution

To rebuild everything run the notebooks in this order.

1. Go to Data Collection + EDA Folder:
    1. [Data Collection + EDA/data_retrevial.ipynb](<Data Collection + EDA/data_retrevial.ipynb>)
    2. EDA Files
        1. [Data Collection + EDA/eda_vfitz.ipynb](<Data Collection + EDA/eda_vfitz.ipynb>)
        2. [Data Collection + EDA/eda_mmcdowell.ipynb](<Data Collection + EDA/eda_mmcdowell.ipynb>)
    3. [Data Collection + EDA/feature_engr.ipynb](<Data Collection + EDA/feature_engr.ipynb>)

2. Go to Baseline Model Eval folder (Optional -for model comparison only):
    1. [Baseline Model Eval/baseline_model_nn.ipynb](<Baseline Model Eval/baseline_model_nn.ipynb>)
    2. [Baseline Model Eval/baseline_model_xgb.ipynb](<Baseline Model Eval/baseline_model_xgb.ipynb>)

3. Go to Final Model folder:
    1. [Final Model/xgb.ipynb](<Final Model/xgb.ipynb>)

4. Go to viz folder:
    1. [viz/viz.ipynb](viz/viz.ipynb)