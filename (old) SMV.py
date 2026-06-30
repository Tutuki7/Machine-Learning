### Linear SVM (LinearSVC) with numerical + dummy features (df_B_encoded).
### Similar preprocessing to other models (median + *_unknown, "Unknown" for categoricals),
### with optional standardization and class balancing.

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, classification_report, roc_auc_score
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np

df_original = pd.read_csv("Donors_dataset.csv")
df_cls = df_original.copy()

STANDARDIZE = True
BALANCE = True

numeric_features = [
    "RECENT_AVG_GIFT_AMT", 
    "LAST_GIFT_AMT", 
    "LIFETIME_AVG_GIFT_AMT",
    "RECENT_AVG_CARD_GIFT_AMT", 
    "LIFETIME_MAX_GIFT_AMT", 
    "NUMBER_PROM_12",
    "MONTHS_SINCE_LAST_GIFT", 
    "MONTHS_SINCE_FIRST_GIFT", 
    "LIFETIME_GIFT_COUNT",
    "PER_CAPITA_INCOME", 
    "MEDIAN_HOUSEHOLD_INCOME",
    "RECENT_RESPONSE_PROP", 
    "RECENT_RESPONSE_COUNT", 
    "WEALTH_RATING"
]

categorical_features = [
    "URBANICITY", 
    "SES", "DONOR_GENDER", 
    "RECENCY_STATUS_96NK"
]

# Keeping only TARGET_B + selected predictors
df_B = df_cls[['TARGET_B'] + numeric_features + categorical_features].copy()

# Imputação de missing values

# Numéricas: mediana (+ coluna *_unknown)
for col in numeric_features:
    df_B[col + "_unknown"] = df_B[col].isna().astype(int)
    med = df_B[col].median()
    df_B[col] = df_B[col].fillna(med)

# Categóricas: "Unknown"
for col in categorical_features:
    df_B[col] = df_B[col].fillna("Unknown")

# One-hot encoding das categóricas
df_B_encoded = pd.get_dummies(
    df_B,
    columns=categorical_features,
    drop_first=True  # usa uma categoria como referência
)

X_B_full = df_B_encoded.drop('TARGET_B', axis=1)
y_B_full = df_B_encoded['TARGET_B']

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    X_B_full, y_B_full, random_state=42
)

# STANDARDIZE / BALANCE options
if BALANCE:
    class_weight = "balanced"
else:
    class_weight = None

steps = []
if STANDARDIZE:
    steps.append(("scaler", StandardScaler()))
steps.append(("svm", LinearSVC(class_weight=class_weight, random_state=42)))

svm_lin = Pipeline(steps)

svm_lin.fit(X_train_svm, y_train_svm)

print("TRAIN accuracy =", svm_lin.score(X_train_svm, y_train_svm))
print("TEST  accuracy =", svm_lin.score(X_test_svm, y_test_svm))

# Confusion matrices
y_train_svm_pred = svm_lin.predict(X_train_svm)
y_test_svm_pred  = svm_lin.predict(X_test_svm)

cm_train_svm = confusion_matrix(y_train_svm, y_train_svm_pred)
disp_train_svm = ConfusionMatrixDisplay(cm_train_svm, display_labels=svm_lin.named_steps["svm"].classes_)
disp_train_svm.plot()
plt.title("Linear SVM - Confusion matrix (TRAIN)")
plt.show()

cm_test_svm = confusion_matrix(y_test_svm, y_test_svm_pred)
disp_test_svm = ConfusionMatrixDisplay(cm_test_svm, display_labels=svm_lin.named_steps["svm"].classes_)
disp_test_svm.plot()
plt.title("Linear SVM - Confusion matrix (TEST)")
plt.show()

# Evaluation metrics
f1_train_svm = f1_score(y_train_svm, y_train_svm_pred, pos_label=1)
f1_test_svm  = f1_score(y_test_svm, y_test_svm_pred, pos_label=1)

print("\nTRAIN classification report (Linear SVM):")
print(classification_report(y_train_svm, y_train_svm_pred))

print("TEST classification report (Linear SVM):")
print(classification_report(y_test_svm, y_test_svm_pred))

print("F1 (class 1) - TRAIN:", f1_train_svm)
print("F1 (class 1) - TEST :", f1_test_svm)

# AUC com decision_function
scores_test_svm = svm_lin.decision_function(X_test_svm)
auc_test_svm = roc_auc_score(y_test_svm, scores_test_svm)
print("TEST AUC (Linear SVM) =", auc_test_svm)
