"""
COMPAS Fairness Audit
Author: Egrah Savai
Objective: Audit COMPAS recidivism dataset for racial bias using IBM AI Fairness 360.
Deliverable: Fairness metrics + visualizations.
"""

# --- Setup ---
# Make sure to install dependencies before running:
# pip install aif360 pandas scikit-learn matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# AIF360 imports
from aif360.datasets import CompasDataset, BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# --- Load COMPAS dataset ---
compas = CompasDataset()
print("Data loaded successfully.")
print("Shape:", compas.features.shape)
print(compas.feature_names[:10], "...\n")

# --- Protected groups ---
protected_attribute = 'race'
privileged_groups = [{'race': 1}]   # Caucasian
unprivileged_groups = [{'race': 0}] # Non-Caucasian

# --- Baseline fairness metrics ---
metric = BinaryLabelDatasetMetric(
    compas, 
    unprivileged_groups=unprivileged_groups, 
    privileged_groups=privileged_groups
)
print("=== Baseline Fairness Metrics ===")
print("Disparate Impact:", round(metric.disparate_impact(), 3))
print("Mean Positive Label Rate:", round(metric.mean_positive_rate(), 3))
print()

# --- Train-test split ---
X = compas.features
y = compas.labels.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Train a simple logistic regression model ---
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_s, y_train)
y_pred = clf.predict(X_test_s)

# --- Convert to AIF360 datasets for fairness metrics ---
test_bld = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=pd.DataFrame(np.hstack((X_test, y_test.reshape(-1,1))),
                    columns=compas.feature_names + ['label']),
    label_names=['label'],
    protected_attribute_names=[protected_attribute]
)
pred_bld = test_bld.copy()
pred_bld.labels = y_pred.reshape(-1, 1)

# --- Classification metrics ---
cm = ClassificationMetric(
    test_bld, pred_bld,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

fpr_priv = cm.false_positive_rate(privileged=True)
fpr_unpriv = cm.false_positive_rate(privileged=False)
fnr_priv = cm.false_negative_rate(privileged=True)
fnr_unpriv = cm.false_negative_rate(privileged=False)
disp_impact = cm.disparate_impact()

print("=== Model Fairness Metrics ===")
print(f"False Positive Rate (Privileged):   {fpr_priv:.3f}")
print(f"False Positive Rate (Unprivileged): {fpr_unpriv:.3f}")
print(f"False Negative Rate (Privileged):   {fnr_priv:.3f}")
print(f"False Negative Rate (Unprivileged): {fnr_unpriv:.3f}")
print(f"Disparate Impact (Predictions):     {disp_impact:.3f}\n")

# --- Visualization ---
groups = ['Privileged (Caucasian)', 'Unprivileged (Non-Caucasian)']
fprs = [fpr_priv, fpr_unpriv]
fnrs = [fnr_priv, fnr_unpriv]

x = np.arange(len(groups))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, fprs, width, label='False Positive Rate', color='tomato')
plt.bar(x + width/2, fnrs, width, label='False Negative Rate', color='skyblue')
plt.xticks(x, groups)
plt.ylabel('Rate')
plt.title('Error Rate Disparities by Race (COMPAS Logistic Regression)')
plt.legend()
plt.tight_layout()
plt.savefig('fairness_audit_results.png')
plt.show()

# --- Bias Mitigation (Reweighing) ---
print("Applying bias mitigation using Reweighing...")
RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
compas_transf = RW.fit_transform(compas)

metric_after = BinaryLabelDatasetMetric(
    compas_transf, 
    unprivileged_groups=unprivileged_groups, 
    privileged_groups=privileged_groups
)
print("=== After Reweighing ===")
print("Disparate Impact:", round(metric_after.disparate_impact(), 3))
print("Mean Positive Label Rate:", round(metric_after.mean_positive_rate(), 3))
print("\nFairness audit complete. Visualization saved as 'fairness_audit_results.png'.")
