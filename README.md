# Customer Churn Prediction for Subscription Services Using Machine Learning

Predicting which subscription customers are likely to cancel (churn) using a **Random Forest** classifier, so a business can act *before* the customer leaves.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Why Random Forest?](#why-random-forest)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [How to Run](#how-to-run)
7. [Workflow](#workflow)
8. [Results](#results)
9. [Model Serialization and Reusability](#model-serialization-and-reusability)
10. [Output Files](#output-files)
11. [Limitations and Future Work](#limitations-and-future-work)
12. [Credits](#credits)

---

## Problem Statement

Subscription businesses lose revenue when customers cancel, and they usually find out only after the customer has already left. Acquiring a new customer costs far more than keeping an existing one, so early identification of at-risk customers is valuable.

This project applies **binary classification** (`Churned` / `Retained`) to customer account, service and billing data to:

- predict the **probability of churn** for every customer,
- group customers into **Low / Medium / High risk** tiers,
- identify the **main factors that drive churn**,
- allow predictions for **new customers without retraining** (saved, reusable model).

## Dataset

| Property | Details |
|---|---|
| **Name** | IBM Telco Customer Churn |
| **Source** | [Kaggle – blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **File** | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Rows** | 7,043 customers |
| **Columns** | 21 (1 ID + 19 input features + 1 target) |
| **Size** | ~977 KB |
| **Target** | `Churn` (Yes / No) – about 26.5% of customers churned |

### Data Dictionary

| Feature | Type | Description |
|---|---|---|
| `customerID` | Categorical | Unique customer identifier (not used for training) |
| `gender` | Categorical | Male / Female |
| `SeniorCitizen` | Binary (0/1) | Whether the customer is a senior citizen |
| `Partner` | Categorical | Has a partner (Yes/No) |
| `Dependents` | Categorical | Has dependents (Yes/No) |
| `tenure` | Integer | Months the customer has stayed with the company |
| `PhoneService` | Categorical | Has phone service (Yes/No) |
| `MultipleLines` | Categorical | Multiple phone lines (Yes/No/No phone service) |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | Categorical | Online security add-on |
| `OnlineBackup` | Categorical | Online backup add-on |
| `DeviceProtection` | Categorical | Device protection add-on |
| `TechSupport` | Categorical | Tech support add-on |
| `StreamingTV` | Categorical | Streaming TV add-on |
| `StreamingMovies` | Categorical | Streaming movies add-on |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | Categorical | Paperless billing (Yes/No) |
| `PaymentMethod` | Categorical | Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic) |
| `MonthlyCharges` | Float | Amount charged per month |
| `TotalCharges` | Float | Total amount charged over the customer's tenure |
| `Churn` | Categorical (**target**) | Whether the customer churned (Yes/No) |

## Why Random Forest?

- Churn depends on **combinations** of factors (for example: month-to-month contract + low tenure + no tech support), which a single linear model struggles to capture.
- It builds many decision trees on random subsets of the data and averages their votes, which **reduces overfitting**.
- It handles a mix of numerical and categorical features **without feature scaling**.
- It provides **feature importance**, which explains *what drives churn*.

## Project Structure

```
.
├── Churn_Prediction_Random_Forest.ipynb   # Main notebook (all steps, with explanations)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset (download from Kaggle)
├── README.md                              # This file
│
│   # Generated when the notebook is run:
├── churn_rf_model.joblib                  # Saved model bundle (model + encoders + settings)
├── churn_predictor.py                     # Reusable prediction module
├── customer_risk_predictions.csv          # Churn probability, risk tier and reason per test customer
├── batch_predictions.csv                  # Example batch prediction output
└── model_results.png                      # Feature importance chart
```

## Installation

**Requirements:** Python 3.8 or higher.

```bash
pip install pandas numpy matplotlib scikit-learn joblib jupyter
pip install shap        # optional – gives a per-customer "top reason"
```

> If you use Anaconda, most of these packages are already installed. `shap` is optional; without it the notebook falls back to a simpler reason and does not crash.

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
2. Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` and `Churn_Prediction_Random_Forest.ipynb` in the **same folder**.
3. Open the notebook:
   ```bash
   jupyter notebook Churn_Prediction_Random_Forest.ipynb
   ```
4. Choose **Kernel → Restart & Run All** (in VS Code: *Restart*, then *Run All*).

The notebook searches for the CSV in the current folder, its parent folder, Downloads, Desktop, Documents and sub-folders, and shows a clear message if it cannot find it.

## Workflow

| Step | Description |
|---|---|
| 1 | Import libraries |
| 2 | Load the data |
| 3 | Exploratory data analysis (churn distribution, churn by contract / tenure / monthly charges) |
| 4 | Clean the data (fix 11 blank `TotalCharges` values, drop `customerID`, encode target as 1/0) |
| 5 | Encode categorical features with `LabelEncoder` (encoders are kept for later reuse) |
| 6 | Stratified 80/20 train/test split (5,634 train / 1,409 test) |
| 7 | Train the Random Forest (200 trees, `max_depth=10`, `class_weight='balanced'`) |
| 8 | Evaluate: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve |
| 9 | Feature importance |
| 10 | Per-customer churn probability, risk tier and top reason (SHAP) |
| 11 | Predict for a single new customer |
| 12 | **Model serialization** – save the model bundle with `joblib` |
| 13 | **Reusability** – load the saved model and predict without retraining |
| 14 | Conclusion |

## Results

Evaluated on the **1,409 unseen test customers**:

| Metric | Score |
|---|---|
| Accuracy | **76.65%** |
| Precision | 54.84% |
| Recall | 68.18% |
| F1-Score | 60.79% |
| ROC-AUC | **83.75%** |

**Confusion matrix**

| | Predicted: Retained | Predicted: Churned |
|---|---|---|
| **Actual: Retained** | 825 (true negatives) | 210 (false alarms) |
| **Actual: Churned** | 119 (missed churners) | 255 (true positives) |

**Accuracy = (825 + 255) / 1,409 = 76.65%**

> Because only ~26.5% of customers churn, accuracy alone can be misleading (always predicting "No churn" would already score ~73%). That is why precision, recall, F1 and ROC-AUC are reported as well.

### Top churn drivers (Random Forest feature importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `Contract` | 16.6% |
| 2 | `tenure` | 15.0% |
| 3 | `TotalCharges` | 13.4% |
| 4 | `MonthlyCharges` | 12.6% |
| 5 | `OnlineSecurity` | 8.4% |
| 6 | `TechSupport` | 6.5% |

**Insight:** new customers on flexible month-to-month contracts, with higher bills and without support add-ons, are the highest churn risk.

### Risk tiers

| Churn probability | Risk tier | Suggested action | Customers (test set) |
|---|---|---|---|
| 0.00 – 0.30 | Low Risk | No action needed | 747 |
| 0.30 – 0.60 | Medium Risk | Monitor / soft engagement | 295 |
| 0.60 – 1.00 | High Risk | Immediate retention outreach | 367 |

Of the 367 customers flagged **High Risk**, about **61%** actually churned.

## Model Serialization and Reusability

The trained model is saved once and can be reused any number of times without retraining or access to the training data.

`churn_rf_model.joblib` is a single bundle containing:

- the trained Random Forest,
- the label encoders (so raw text values are converted exactly as during training),
- the feature column order,
- the risk thresholds (0.30 / 0.60),
- metadata (training date, scikit-learn version, test metrics).

### Using the saved model in any script or notebook

Keep `churn_rf_model.joblib` and `churn_predictor.py` in the same folder:

```python
from churn_predictor import predict_churn

customer = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0,
}

print(predict_churn(customer))
```

Example output:

```
 Churn_Probability Risk_Tier                                           Recommended_Action
               0.9 High Risk Immediate retention outreach (e.g. discount or plan upgrade)
```

`predict_churn()` also accepts a **pandas DataFrame** to score many customers at once, and it rejects invalid input (for example an unseen `Contract` value or missing fields) with a clear error message.

> **Notes:** load the model with the **same scikit-learn version** it was saved with (stored in the metadata), and only load `.joblib` files from sources you trust.

## Output Files

| File | Description |
|---|---|
| `customer_risk_predictions.csv` | One row per test customer: `customerID`, `Churn_Probability`, `Risk_Tier`, `Top_Reason`, `Actual_Churn` (sorted by highest risk) |
| `churn_rf_model.joblib` | Saved model bundle |
| `churn_predictor.py` | Reusable prediction module |
| `batch_predictions.csv` | Example batch predictions for 10 customers (demo only – some rows were seen in training) |
| `model_results.png` | Top-10 feature importance chart |

## Limitations and Future Work

**Limitations**
- Precision is moderate (~55%): because only ~26% of customers churn, some loyal customers are flagged as at-risk (false alarms).
- The dataset comes from a single telecom company, so results may not transfer directly to other subscription businesses.
- A single train/test split was used; results can vary slightly with a different split.

**Future work**
- Hyper-parameter tuning (grid / random search) and cross-validation.
- Decision-threshold tuning to trade precision for higher recall.
- Handling class imbalance with SMOTE or other resampling.
- Comparing with gradient boosting models such as XGBoost or LightGBM.
- Deploying `predict_churn()` behind a simple web app or API.

## Credits

- Dataset: *IBM Telco Customer Churn*, published on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- Libraries: pandas, NumPy, matplotlib, scikit-learn, joblib, SHAP.
