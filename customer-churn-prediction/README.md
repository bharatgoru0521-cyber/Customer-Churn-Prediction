# Customer Churn Prediction

A complete machine learning pipeline to predict customer churn using the Telco Customer Churn dataset. Covers feature engineering, model comparison, and evaluation — built to be interview-ready and production-quality.

---

## Project Structure

```
customer-churn-prediction/
│
├── data/                    # Raw and processed data
│   └── .gitkeep
│
├── notebooks/               # Exploratory & step-by-step notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling_and_evaluation.ipynb
│
├── src/                     # Reusable source modules
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
│
├── outputs/                 # Saved models, plots, metrics
│   └── .gitkeep
│
├── tests/                   # Unit tests
│   └── test_features.py
│
├── main.py                  # Run full pipeline end-to-end
├── requirements.txt
└── README.md
```

---

## Models Used

| Model               | Strengths                          |
|---------------------|------------------------------------|
| Logistic Regression | Fast, interpretable, good baseline |
| Decision Tree       | Visual, explainable to stakeholders|
| Random Forest       | Best accuracy, handles non-linearity|

---

## Key Features Engineered

- `tenure_group` — bucketed tenure (new / mid / long-term)
- `charges_ratio` — monthly charges ÷ (total charges + 1)
- `num_services` — count of active add-on services
- `is_month_to_month` — flag for highest-churn contract type

---

## Evaluation Metrics

- **Precision & Recall** (with threshold tuning)
- **F1 Score** (primary metric — imbalanced classes)
- **ROC-AUC** (model discrimination)
- **Confusion Matrix**

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Get **Telco Customer Churn** from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` inside the `data/` folder.

### 4. Run the full pipeline
```bash
python main.py
```

Or explore step-by-step in the notebooks:
```bash
jupyter notebook notebooks/
```

---

## Interview Talking Points

- Why F1 score beats accuracy on imbalanced data
- Precision vs Recall tradeoff and business cost of false negatives
- Why Random Forest outperforms a single Decision Tree (variance reduction via bagging)
- Threshold tuning: shifting the decision boundary to favor recall
- SMOTE for handling class imbalance

---

## Results (typical on Telco dataset)

| Model               | Accuracy | F1 (Churn) | ROC-AUC |
|---------------------|----------|------------|---------|
| Logistic Regression | ~78%     | ~0.58      | ~0.84   |
| Decision Tree       | ~79%     | ~0.55      | ~0.73   |
| Random Forest       | ~80%     | ~0.60      | ~0.85   |

> Results vary slightly with random seed and preprocessing choices.

---

## Tech Stack

- Python 3.9+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- imbalanced-learn (SMOTE)
- joblib (model persistence)
