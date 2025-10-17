"""
COMPAS Fairness Audit - Fixed Version for Colab
Author: Egrah Savai
Objective: Audit COMPAS recidivism dataset for racial bias using IBM AI Fairness 360.
"""

# --- Step 1: Installation & Setup ---
print("📦 Installing required packages...")
!pip install 'aif360[all]' pandas scikit-learn matplotlib seaborn

print("🔧 Setting up dataset directory...")
!mkdir -p /usr/local/lib/python3.12/dist-packages/aif360/data/raw/compas

print("📥 Downloading COMPAS dataset...")
!wget -q https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv -P /usr/local/lib/python3.12/dist-packages/aif360/data/raw/compas/

print("✅ Setup complete! Now importing libraries...")

# --- Step 2: Import Libraries ---
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

print("✅ All libraries imported successfully!")

# --- Step 3: Load COMPAS Dataset ---
print("📊 Loading COMPAS dataset...")
try:
    compas = CompasDataset()
    print("✅ COMPAS dataset loaded successfully!")
    print(f"Dataset shape: {compas.features.shape}")
    print(f"Features: {compas.feature_names[:10]}...")
    print(f"Label distribution: {np.unique(compas.labels, return_counts=True)}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# --- Step 4: Define Protected Groups ---
protected_attribute = 'race'
privileged_groups = [{'race': 1}]   # Caucasian
unprivileged_groups = [{'race': 0}] # Non-Caucasian

print(f"🔒 Protected attribute: {protected_attribute}")
print(f"Privileged groups: {privileged_groups}")
print(f"Unprivileged groups: {unprivileged_groups}")

# --- Step 5: Baseline Fairness Metrics ---
print("\n" + "="*50)
print("📈 BASELINE FAIRNESS METRICS")
print("="*50)

metric_orig = BinaryLabelDatasetMetric(
    compas, 
    unprivileged_groups=unprivileged_groups, 
    privileged_groups=privileged_groups
)

print(f"Disparate Impact: {metric_orig.disparate_impact():.3f}")
print(f"Statistical Parity Difference: {metric_orig.statistical_parity_difference():.3f}")

# Calculate mean positive rates manually
privileged_rate = compas.labels[compas.protected_attributes[:,0] == 1].mean()
unprivileged_rate = compas.labels[compas.protected_attributes[:,0] == 0].mean()

print(f"Mean Positive Label Rate - Privileged: {privileged_rate:.3f}")
print(f"Mean Positive Label Rate - Unprivileged: {unprivileged_rate:.3f}")

# Interpretation
di = metric_orig.disparate_impact()
if di < 0.8:
    print("🚨 WARNING: Disparate Impact < 0.8 indicates potential bias")
elif di > 1.25:
    print("⚠️  NOTE: Disparate Impact > 1.25 may indicate reverse bias")
else:
    print("✅ Disparate Impact within acceptable range (0.8 - 1.25)")

# --- Step 6: Train-Test Split & Model Training ---
print("\n" + "="*50)
print("🤖 MODEL TRAINING")
print("="*50)

X = compas.features
y = compas.labels.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

print(f"Model accuracy: {clf.score(X_test_scaled, y_test):.3f}")

# --- Step 7: Model Fairness Evaluation ---
print("\n" + "="*50)
print("📊 MODEL FAIRNESS METRICS")
print("="*50)

# Create test dataset for fairness evaluation
test_dataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=pd.DataFrame(
        np.hstack([X_test, y_test.reshape(-1, 1)]),
        columns=compas.feature_names + ['label']
    ),
    label_names=['label'],
    protected_attribute_names=[protected_attribute]
)

# Create prediction dataset
pred_dataset = test_dataset.copy()
pred_dataset.labels = y_pred.reshape(-1, 1)

# Calculate classification metrics
classification_metric = ClassificationMetric(
    test_dataset, 
    pred_dataset,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

# Extract key metrics
fpr_priv = classification_metric.false_positive_rate(privileged=True)
fpr_unpriv = classification_metric.false_positive_rate(privileged=False)
fnr_priv = classification_metric.false_negative_rate(privileged=True)
fnr_unpriv = classification_metric.false_negative_rate(privileged=False)
tpr_priv = classification_metric.true_positive_rate(privileged=True)
tpr_unpriv = classification_metric.true_positive_rate(privileged=False)
disp_impact_pred = classification_metric.disparate_impact()

print("=== ERROR RATES ===")
print(f"False Positive Rate (Privileged):   {fpr_priv:.3f}")
print(f"False Positive Rate (Unprivileged): {fpr_unpriv:.3f}")
print(f"FPR Ratio (Unpriv/Priv):           {fpr_unpriv/fpr_priv:.3f}")
print()
print(f"False Negative Rate (Privileged):   {fnr_priv:.3f}")
print(f"False Negative Rate (Unprivileged): {fnr_unpriv:.3f}")
print(f"FNR Ratio (Unpriv/Priv):           {fnr_unpriv/fnr_priv:.3f}")
print()
print(f"True Positive Rate (Privileged):    {tpr_priv:.3f}")
print(f"True Positive Rate (Unprivileged):  {tpr_unpriv:.3f}")
print(f"TPR Ratio (Unpriv/Priv):           {tpr_unpriv/tpr_priv:.3f}")
print()
print(f"Disparate Impact (Predictions):     {disp_impact_pred:.3f}")

# --- Step 8: Visualization ---
print("\n" + "="*50)
print("📊 CREATING VISUALIZATIONS")
print("="*50)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Error rates by group
groups = ['Privileged\n(Caucasian)', 'Unprivileged\n(Non-Caucasian)']
fprs = [fpr_priv, fpr_unpriv]
fnrs = [fnr_priv, fnr_unpriv]

x = np.arange(len(groups))
width = 0.35

ax1.bar(x - width/2, fprs, width, label='False Positive Rate', color='tomato', alpha=0.8)
ax1.bar(x + width/2, fnrs, width, label='False Negative Rate', color='skyblue', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(groups)
ax1.set_ylabel('Rate')
ax1.set_title('Error Rate Disparities by Race Group\n(COMPAS Logistic Regression)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Fairness metrics comparison
metrics_before = ['Disparate Impact', 'FPR Ratio', 'FNR Ratio']
values_before = [disp_impact_pred, fpr_unpriv/fpr_priv, fnr_unpriv/fnr_priv]

colors = ['green' if 0.8 <= x <= 1.25 else 'red' for x in values_before]

ax2.barh(metrics_before, values_before, color=colors, alpha=0.7)
ax2.axvline(x=0.8, color='red', linestyle='--', alpha=0.7, label='Fairness Threshold (0.8)')
ax2.axvline(x=1.25, color='red', linestyle='--', alpha=0.7, label='Fairness Threshold (1.25)')
ax2.axvline(x=1.0, color='green', linestyle='-', alpha=0.5, label='Perfect Fairness (1.0)')
ax2.set_xlabel('Ratio')
ax2.set_title('Fairness Metrics Overview\n(Green = Fair, Red = Biased)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fairness_audit_results.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Step 9: Bias Mitigation ---
print("\n" + "="*50)
print("⚖️ APPLYING BIAS MITIGATION")
print("="*50)

print("Applying Reweighing pre-processing...")
RW = Reweighing(
    unprivileged_groups=unprivileged_groups, 
    privileged_groups=privileged_groups
)
compas_transformed = RW.fit_transform(compas)

# Calculate metrics after mitigation
metric_after = BinaryLabelDatasetMetric(
    compas_transformed, 
    unprivileged_groups=unprivileged_groups, 
    privileged_groups=privileged_groups
)

print("=== BEFORE REWEIGHING ===")
print(f"Disparate Impact: {metric_orig.disparate_impact():.3f}")
print(f"Statistical Parity Difference: {metric_orig.statistical_parity_difference():.3f}")

print("\n=== AFTER REWEIGHING ===")
print(f"Disparate Impact: {metric_after.disparate_impact():.3f}")
print(f"Statistical Parity Difference: {metric_after.statistical_parity_difference():.3f}")

# Calculate improvement
improvement_di = abs(metric_after.disparate_impact() - 1.0) - abs(metric_orig.disparate_impact() - 1.0)
improvement_spd = abs(metric_after.statistical_parity_difference()) - abs(metric_orig.statistical_parity_difference())

print(f"\n📈 Improvement in Disparate Impact: {improvement_di:+.3f}")
print(f"📈 Improvement in Statistical Parity: {improvement_spd:+.3f}")

if improvement_di < 0:
    print("✅ Reweighing improved fairness (moved disparate impact closer to 1.0)")
else:
    print("⚠️  Reweighing did not improve disparate impact")

# --- Step 10: Summary ---
print("\n" + "="*50)
print("🎯 FAIRNESS AUDIT SUMMARY")
print("="*50)

print("🔍 KEY FINDINGS:")
print(f"   • Baseline disparate impact: {metric_orig.disparate_impact():.3f}")
print(f"   • Model prediction disparate impact: {disp_impact_pred:.3f}")
print(f"   • FPR ratio (Unprivileged/Privileged): {fpr_unpriv/fpr_priv:.3f}")
print(f"   • FNR ratio (Unprivileged/Privileged): {fnr_unpriv/fnr_priv:.3f}")

print("\n📋 RECOMMENDATIONS:")
if disp_impact_pred < 0.8 or disp_impact_pred > 1.25:
    print("   • Significant bias detected - consider additional mitigation techniques")
    print("   • Explore other pre-processing methods (Disparate Impact Remover)")
    print("   • Consider in-processing or post-processing fairness approaches")
else:
    print("   • Model shows reasonable fairness properties")
    print("   • Continue monitoring for demographic shifts")

print("\n💾 Output saved: 'fairness_audit_results.png'")
print("✅ Fairness audit complete!")
