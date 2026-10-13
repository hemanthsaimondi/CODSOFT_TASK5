# Task 5: Credit Card Fraud Detection

Detects fraudulent credit card transactions in a highly imbalanced dataset.

## Dataset
Kaggle "Credit Card Fraud Detection" (`creditcard.csv`): 284,807 transactions by European cardholders, 31 columns
(Time, Amount, anonymised PCA features V1-V28, Class). The file is ~150 MB, too large for GitHub, so it is **not included**.
Download it from Kaggle and place `creditcard.csv` in this folder before running.

## Approach
1. **Cleaning**: no missing values; removed 1,081 duplicate rows (so identical rows can't appear in both train and test) -> 283,726 transactions, 473 frauds (0.167%)
2. **Why accuracy is misleading**: always predicting "genuine" is already 99.83% accurate, so I evaluate with precision, recall, F1 and PR-AUC
3. **Split first, then normalise**: stratified 80/20 split; `RobustScaler` fitted on the training set only, applied to Time and Amount
4. **Imbalance strategies** (applied to training data only): none, class weights, SMOTE oversampling, random undersampling
5. **Models**: Logistic Regression and Random Forest

## Results (test set: 56,746 transactions, 95 frauds, threshold 0.5)
| Model | Precision | Recall | F1 | PR-AUC | Caught | Missed | False alarms |
|---|---|---|---|---|---|---|---|
| LR, no handling | 0.862 | 0.589 | 0.700 | 0.695 | 56 | 39 | 9 |
| LR, class weights | 0.056 | 0.874 | 0.106 | 0.672 | 83 | 12 | 1393 |
| LR, SMOTE | 0.054 | 0.874 | 0.101 | 0.673 | 83 | 12 | 1463 |
| LR, undersampling | 0.051 | 0.874 | 0.096 | 0.600 | 83 | 12 | 1552 |
| **RF, no handling** | **0.972** | 0.737 | **0.838** | 0.789 | 70 | 25 | 2 |
| RF, class weights | 0.909 | 0.737 | 0.814 | **0.805** | 70 | 25 | 7 |
| RF, undersampling (10:1) | 0.647 | 0.789 | 0.711 | 0.690 | 75 | 20 | 41 |

## Key insights
- Random Forest clearly beats Logistic Regression on F1 and PR-AUC.
- Resampling and class weights raise **recall** (more fraud caught) but cost a lot of **precision** (many false alarms). For Logistic Regression that trade is poor: 1,400+ false alarms for 27 extra frauds caught.
- Which model is best depends on business cost: if a missed fraud is far more expensive than a false alarm, the higher-recall models or a lower decision threshold are preferable.
- The test set has only 95 frauds, so small differences between models are noisy.

## Run
```
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
python fraud_detection.py
```
