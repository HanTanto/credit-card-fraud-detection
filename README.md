# credit-card-fraud-detection

## Overview
This project builds a machine learning model to detect fraudulent credit card transactions using a large-scale, highly imbalanced dataset (fraud rate <1%). The goal is to explore how standard classification metrics behave under severe class imbalance and to identify approaches for detecting rare fraud events.

## Dataset
- Source: [Credit Card Transactions Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- ~370K transactions, with only ~1,900 labeled as fraud (~0.5% of the data)

## Approach
1. Data cleaning and exploratory analysis of transaction patterns
2. Feature engineering from transaction, merchant, and cardholder attributes
3. Model training using [isi algoritma yang kamu pakai, misal: Random Forest / XGBoost / Logistic Regression]
4. Evaluation focused on precision-recall trade-offs rather than raw accuracy, given the severe class imbalance

## Results
| Metric | Class 0 (Non-Fraud) | Class 1 (Fraud) |
|---|---|---|
| Precision | 1.00 | 0.02 |
| Recall | 0.93 | 0.33 |
| F1-score | 0.96 | 0.04 |

Overall accuracy: 92% — but this figure is misleading given the ~0.5% fraud rate, since a naive model predicting "no fraud" for every transaction would already score ~99.5% accuracy. The model currently catches 33% of actual fraud cases, at the cost of a high false-positive rate (precision 0.02).

## Key Learnings
- Accuracy is not a reliable metric for highly imbalanced classification problems
- Precision-recall trade-off is a more meaningful lens for fraud detection use cases
- Further improvement is needed to make this model production-viable

## Next Steps / Planned Improvements
- Address class imbalance using SMOTE, undersampling, or `class_weight='balanced'`
- Threshold tuning to optimize the precision-recall trade-off for business needs
- Try ensemble/boosting methods (e.g., XGBoost, LightGBM) for better minority-class detection
- Cost-sensitive learning to weigh false negatives (missed fraud) more heavily than false positives

## Tech Stack
Python | pandas | scikit-learn | [tambahkan library lain yang kamu pakai]

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook ANN_Fraud_Detection.ipynb
```
