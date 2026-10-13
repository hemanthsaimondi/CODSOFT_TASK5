"""
CODSOFT Data Science Internship - Task 5: Credit Card Fraud Detection
Detect fraudulent transactions in a highly imbalanced dataset (~0.17% fraud).

Dataset: Kaggle "Credit Card Fraud Detection" (creditcard.csv, 284,807 transactions).
Download it and place creditcard.csv in this folder (it is too large for GitHub).
Usage: python fraud_detection.py [path/to/creditcard.csv]
"""
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

RS = 42
sns.set_theme(style="whitegrid")
path = sys.argv[1] if len(sys.argv) > 1 else "creditcard.csv"

# ---------------------------------------------------------------- 1. Load & inspect
df = pd.read_csv(path)
print("Shape:", df.shape, "| Missing values:", int(df.isnull().sum().sum()))
dups = int(df.duplicated().sum())
df = df.drop_duplicates()        # identical rows could land in both train and test (leakage)
print(f"Dropped {dups} duplicate rows -> {len(df)} left")
vc = df["Class"].value_counts()
print(f"Genuine: {vc[0]:,} | Fraud: {vc[1]:,} ({vc[1] / len(df) * 100:.3f}%)")
print("A model that always says 'genuine' would be %.2f%% accurate - accuracy is misleading here." % (vc[0] / len(df) * 100))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(x="Class", data=df, ax=ax[0]); ax[0].set_yscale("log"); ax[0].set_title("Class balance (log scale)")
sns.boxplot(x="Class", y="Amount", data=df, showfliers=False, ax=ax[1]); ax[1].set_title("Amount by class")
corr = df.corr()["Class"].drop("Class").sort_values()
pd.concat([corr.head(5), corr.tail(5)]).plot(kind="barh", ax=ax[2], title="Strongest correlations with fraud")
plt.tight_layout(); plt.savefig("eda.png", dpi=110); plt.close()

# ---------------------------------------------------------------- 2. Split first, then scale (no leakage)
X, y = df.drop(columns="Class"), df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)

# V1-V28 are already PCA-scaled; Time and Amount are on very different scales
scaler = RobustScaler().fit(X_train[["Time", "Amount"]])    # robust to Amount's huge outliers
for part in (X_train, X_test):
    part[["Time", "Amount"]] = scaler.transform(part[["Time", "Amount"]])
print(f"\nTrain: {len(X_train):,} ({int(y_train.sum())} fraud) | Test: {len(X_test):,} ({int(y_test.sum())} fraud)")

# ---------------------------------------------------------------- 3. Imbalance strategies (applied to TRAIN only)
def lr(**kw): return LogisticRegression(max_iter=1000, **kw)
def rf(**kw): return RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=RS, **kw)

print("Resampling training data...")
X_sm, y_sm = SMOTE(random_state=RS).fit_resample(X_train, y_train)
X_us, y_us = RandomUnderSampler(random_state=RS).fit_resample(X_train, y_train)
X_us2, y_us2 = RandomUnderSampler(sampling_strategy=0.1, random_state=RS).fit_resample(X_train, y_train)  # 10 genuine : 1 fraud

experiments = [
    ("LR - no imbalance handling",          lr(),                                   X_train, y_train),
    ("LR - class weights",                  lr(class_weight="balanced"),            X_train, y_train),
    ("LR - SMOTE oversampling",             lr(),                                   X_sm, y_sm),
    ("LR - random undersampling",           lr(),                                   X_us, y_us),
    ("RF - no imbalance handling",          rf(),                                   X_train, y_train),
    ("RF - class weights",                  rf(class_weight="balanced_subsample"),  X_train, y_train),
    ("RF - undersampling (10:1)",           rf(),                                   X_us2, y_us2),
]

# ---------------------------------------------------------------- 4. Train & evaluate on the untouched test set
rows, probs = [], {}
for name, model, Xt, yt in experiments:
    t0 = time.time()
    model.fit(Xt, yt)
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    rows.append(dict(Model=name, Precision=precision_score(y_test, pred, zero_division=0),
                     Recall=recall_score(y_test, pred), F1=f1_score(y_test, pred),
                     PR_AUC=average_precision_score(y_test, p), ROC_AUC=roc_auc_score(y_test, p),
                     Caught=tp, Missed=fn, FalseAlarms=fp))
    probs[name] = p
    print(f"  done: {name:32s} ({time.time() - t0:.0f}s)")

res = pd.DataFrame(rows).round(3)
res.to_csv("results.csv", index=False)
print("\n=== Test set results (threshold 0.5) ===")
print(res.to_string(index=False))

# ---------------------------------------------------------------- 5. Plots
plt.figure(figsize=(8, 6))
for name, p in probs.items():
    pr, rc, _ = precision_recall_curve(y_test, p)
    plt.plot(rc, pr, label=f"{name} (AP={average_precision_score(y_test, p):.2f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curves (test set)")
plt.legend(fontsize=7); plt.tight_layout(); plt.savefig("precision_recall_curves.png", dpi=110); plt.close()

best = res.sort_values("PR_AUC", ascending=False).iloc[0]["Model"]
best_f1 = res.sort_values("F1", ascending=False).iloc[0]["Model"]
pred = (probs[best_f1] >= 0.5).astype(int)
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix\n{best_f1}")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=110); plt.close()
print("\nBest by PR-AUC:", best, "| Best by F1:", best_f1)
