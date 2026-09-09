<div align="center">

# 📞 Bank Marketing Call Classifier

### Predicting term deposit subscription from telemarketing call data using Random Forest

[![Python](https://img.shields.io/badge/Python-3.12-000000?style=for-the-badge&logo=python&logoColor=00FF7F)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-000000?style=for-the-badge&logo=scikitlearn&logoColor=00FF7F)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-000000?style=for-the-badge&logo=pandas&logoColor=00FF7F)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-000000?style=for-the-badge&logo=jupyter&logoColor=00FF7F)](https://jupyter.org/)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Model](#-model)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🚀 Overview

This project builds a **binary classification model** that predicts whether a client will subscribe to a term deposit after being contacted through a bank's telemarketing campaign. It covers the full machine learning pipeline: from raw data exploration and cleaning, through feature engineering and model training, to evaluation and insights extraction.

The goal is not only to classify outcomes accurately, but to identify **which factors most influence a client's decision** — turning a raw call log into actionable business insight.

---

## 🎯 Problem Statement

Direct marketing campaigns (phone calls) are costly and time-consuming. Contacting every client in a database is inefficient when only a fraction actually subscribes to the offered product.

**Business question:** Can we predict, before or during a call, which clients are most likely to say "yes" to a term deposit — so the bank can prioritize outreach and improve campaign efficiency?

---

## 📊 Dataset

- **Source:** [Bank Marketing Dataset — Kaggle](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing) (`bank-additional-full.csv`), originally published on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Description:** Real data from direct marketing campaigns (phone calls) of a Portuguese banking institution, collected to determine whether a client would subscribe to a term deposit.
- **Size:** 41,188 records × 21 columns (no missing values — verified column by column).
- **Target variable:** `y` — did the client subscribe to a term deposit? (`yes` / `no`), highly imbalanced: **~88.7% "no" vs ~11.3% "yes"**.

### Original features included:

| Category | Features |
|---|---|
| Client info | `age`, `job`, `marital`, `education`, `default`, `housing`, `loan` |
| Contact info | `contact`, `month`, `day_of_week`, `duration` |
| Campaign info | `campaign`, `pdays`, `previous`, `poutcome` |
| Socioeconomic context | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` |

## 🗂️ Project Structure

```
📦 bank-marketing-classifier
├── 📂 data/
│   └── bank-additional-full.csv        # Raw dataset (not included)
├── 📂 notebooks/
│   └── bank_marketing_analysis.ipynb   # EDA, preprocessing, training & evaluation
├── 📂 models/
│   ├── rf_model.joblib                 # Trained Random Forest model
│   └── scaler_bank_marketing.joblib    # Fitted StandardScaler
├── 📂 src/
│   └── predict.py                      # Script to load model & run predictions
└── README.md
```

> ⚠️ This structure is designed for the final phase of the project; at the moment it is in a development/exploratory scenario (see [Future Improvements](#-future-improvements)).

---

## 🔬 Methodology

1. **Data Cleaning** — handled missing/unknown values across categorical columns.
2. **Exploratory Data Analysis (EDA)** — analyzed distributions and correlations between client attributes, campaign variables, and the subscription outcome.
3. **Feature Engineering** — encoded categorical variables (`marital`, `default`, `housing`, `loan`, `contact`, `poutcome`) using one-hot encoding.
4. **Feature Scaling** — standardized numerical features with `StandardScaler` to normalize the input space.
5. **Train/Test Split** — split the dataset to evaluate generalization on unseen data.
6. **Model Training** — trained a `RandomForestClassifier` and tuned its hyperparameters.
7. **Evaluation** — assessed performance with classification metrics and a confusion matrix.

---

## 🌲 Model

The final model is a **Random Forest Classifier** trained with the following configuration:

| Parameter | Value |
|---|---|
| `n_estimators` | 100 |
| `criterion` | gini |
| `max_features` | sqrt |
| `random_state` | 42 |
| `n_jobs` | -1 |

**Final feature set (post-encoding, 26 features):**

`age`, `job`, `education`, `month`, `day_of_week`, `duration`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`, `marital_married`, `marital_single`, `marital_divorced`, `default_no`, `default_yes`, `housing_no`, `housing_yes`, `loan_no`, `loan_yes`, `contact_cellular`, `contact_telephone`, `poutcome_success`

Both the trained model (`rf_model.joblib`) and the fitted scaler (`scaler_bank_marketing.joblib`) are serialized with `joblib` so they can be reused directly for inference without retraining.

---

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | `0.9199` |
| Precision | `0.6811` |
| Recall | `0.5431` |
| F1-Score | `0.60` |

### Key Insights:
- **High Accuracy** (91.99%): The model correctly classifies clients in the vast majority of cases.
- **Moderate Precision** (68.11%): When the model predicts "subscription," it's correct about 68% of the time.
- **Lower Recall** (54.31%): The model identifies only about 54% of actual positive cases—there's room to improve detection of true subscriptions.
- **Imbalanced Dataset Impact**: The high accuracy is partly due to class imbalance; the model's conservative approach reduces false positives at the cost of some false negatives.

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/DuvanL-bot/Dataset-Bank-Marketing.git
cd Dataset-Bank-Marketing

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`** should include at minimum:

```
pandas
numpy
scikit-learn
matplotlib
jupyterlab
joblib
```

---

## ▶️ Usage

### Run the notebook

```bash
jupyter lab notebooks/bank_marketing_analysis.ipynb
```

### Use the trained model for inference

```python
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load("models/rf_model.joblib")
scaler = joblib.load("models/scaler_bank_marketing.joblib")

# new_data must have the same 26 columns used during training,
# in the same order, after applying the same encoding steps
scaled_data = scaler.transform(new_data)
prediction = model.predict(scaled_data)
probability = model.predict_proba(scaled_data)

print("Prediction:", prediction)          # 0 = No subscription | 1 = Subscription
print("Probability:", probability)
```

---

## 🧰 Tech Stack

<div align="center">

<img src="https://img.shields.io/badge/Python-000000?style=for-the-badge&logo=python&logoColor=00FF7F" />
<img src="https://img.shields.io/badge/Pandas-000000?style=for-the-badge&logo=pandas&logoColor=00FF7F" />
<img src="https://img.shields.io/badge/NumPy-000000?style=for-the-badge&logo=numpy&logoColor=00FF7F" />
<img src="https://img.shields.io/badge/scikit--learn-000000?style=for-the-badge&logo=scikitlearn&logoColor=00FF7F" />
<img src="https://img.shields.io/badge/Matplotlib-000000?style=for-the-badge&logo=plotly&logoColor=00FF7F" />
<img src="https://img.shields.io/badge/Jupyter-000000?style=for-the-badge&logo=jupyter&logoColor=00FF7F" />

</div>

---

## 🧭 Future Improvements

- [ ] Compare against other models (Logistic Regression, XGBoost, Gradient Boosting)
- [ ] Handle class imbalance explicitly (e.g. `class_weight="balanced"` or SMOTE)
- [ ] Deploy the model behind a simple API (Flask/FastAPI) for real-time predictions
- [ ] Build a small dashboard (Streamlit / Power BI) to visualize campaign insights
- [ ] Add cross-validation for more robust performance estimates
- [ ] Perform hyperparameter tuning with GridSearchCV or RandomizedSearchCV

---

## 👤 Author

**Duvan** — Junior Full Stack Developer & aspiring Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-000000?style=for-the-badge&logo=linkedin&logoColor=00FF7F)](https://www.linkedin.com/in/felipe-lizarazo-contreras-ba974836b)
[![GitHub](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=00FF7F)](https://github.com/DuvanL-bot)

If this project was useful to you, consider giving it a ⭐!

---

**Last Updated:** September 2026
