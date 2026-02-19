# Machine Learning Courseware

A comprehensive 10-module machine learning course delivered as Jupyter notebooks. Covers foundational mathematics through neural networks, with hands-on code examples, visualizations, and exercises in every module.

---

## Prerequisites

- Python 3.8 or higher
- Basic familiarity with Python, NumPy, and Pandas

---

## Environment Setup

### Option A: Using venv (recommended)

```bash
# 1. Create a virtual environment
python -m venv ml_env

# 2. Activate the environment
# Windows
ml_env\Scripts\activate
# macOS/Linux
source ml_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Option B: Using Conda

```bash
# 1. Create a conda environment
conda create -n ml_course python=3.10 -y

# 2. Activate it
conda activate ml_course

# 3. Install core packages
conda install numpy pandas matplotlib seaborn scikit-learn scipy jupyter -y

# 4. Install optional packages
pip install xgboost imbalanced-learn
conda install tensorflow -y   # or: pip install tensorflow
```

---

## Running the Notebooks

```bash
# Start Jupyter Notebook
jupyter notebook

# Or use Jupyter Lab (install first: pip install jupyterlab)
jupyter lab
```

Open any `.ipynb` file from the file browser to get started.

---

## Module Overview

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_Introduction_to_ML.ipynb` | What is ML, types of learning, the ML workflow |
| 02 | `02_Mathematical_Foundations.ipynb` | Linear algebra, statistics, probability, calculus intuition |
| 03 | `03_Data_Preprocessing.ipynb` | Missing values, encoding, scaling, pipelines |
| 04 | `04_Regression.ipynb` | Linear, polynomial, and regularized regression |
| 05 | `05_Classification.ipynb` | Logistic Regression, KNN, SVM, Decision Trees, Naive Bayes |
| 06 | `06_Model_Evaluation.ipynb` | Confusion matrix, ROC/AUC, cross-validation, tuning |
| 07 | `07_Unsupervised_Learning.ipynb` | K-Means, DBSCAN, PCA, t-SNE |
| 08 | `08_Ensemble_Methods.ipynb` | Random Forest, AdaBoost, Gradient Boosting, XGBoost, Stacking |
| 09 | `09_Neural_Networks.ipynb` | Perceptron, MLP from scratch, Keras/TensorFlow |
| 10 | `10_Advanced_Topics.ipynb` | Imbalanced data, feature engineering, end-to-end pipelines |

Work through the modules in order. Each builds on concepts from the previous ones.

---

## Dependencies

**Core** (required for all modules):

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualization |
| `scikit-learn` | ML algorithms, preprocessing, evaluation |
| `scipy` | Statistical functions |

**Optional** (needed only for specific sections):

| Package | Used In | Purpose |
|---------|---------|---------|
| `xgboost` | Module 08 | XGBoost classifier |
| `tensorflow` | Module 09 | Keras neural networks and CNNs |
| `imbalanced-learn` | Module 10 | SMOTE oversampling |

Sections that require optional packages include `try/except` blocks and will print installation instructions if the package is missing. The rest of each module will still run without them.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install <package_name>` |
| Matplotlib plots not showing | Add `%matplotlib inline` at the top of a notebook |
| TensorFlow installation fails | See [TensorFlow install guide](https://www.tensorflow.org/install) |
| Jupyter kernel not found | Run `python -m ipykernel install --user --name ml_env` |
