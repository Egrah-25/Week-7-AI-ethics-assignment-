# COMPAS Fairness Audit 🧠
**Author:** Egrah Savai  
**Course:** AI Ethics  
**Objective:** Audit the COMPAS Recidivism dataset for racial bias using IBM's [AI Fairness 360 (AIF360)](https://aif360.mybluemix.net/).

---

## 📘 Project Overview
This project analyzes racial bias in the COMPAS recidivism dataset, a dataset used to predict the likelihood of criminal reoffending. Using **AI Fairness 360**, we evaluate disparities in prediction error rates (False Positive/Negative Rates) between privileged and unprivileged racial groups.

We train a **Logistic Regression** model as a baseline classifier and compute fairness metrics before and after applying the **Reweighing** algorithm to mitigate bias.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/compas_fairness_audit.git
cd compas_fairness_audit
2. Install Dependencies

pip install -r requirements.txt

3. Run the Audit

python compas_fairness_audit.py

The script will:

Print baseline and post-mitigation fairness metrics

Generate a visualization of disparities

Save output figure to results/fairness_audit_results.png


📊 Output Example

Metric	Privileged (Caucasian)	Unprivileged (Non-Caucasian)

False Positive Rate	0.25	0.45
False Negative Rate	0.30	0.22
Disparate Impact	0.65	-


> Observation: The model shows higher false positives for the unprivileged group, indicating bias.


🧩 Ethical Insight

AI systems trained on historical criminal data can reinforce systemic inequalities. Bias audits are crucial for developing trustworthy, transparent, and equitable AI.
