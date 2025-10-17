# COMPAS Dataset Fairness Audit – Report

**Author:** Egrah Savai  
**Word Count:** ~300

The COMPAS dataset, widely used for predicting criminal recidivism, was analyzed using IBM’s AI Fairness 360 toolkit to evaluate racial bias. A logistic regression model was trained to predict reoffending risk, treating race as a protected attribute. Privileged and unprivileged groups were defined as Caucasian and Non-Caucasian, respectively.

Preliminary dataset analysis revealed disparities in outcome distributions, reflecting potential historical bias. The fairness audit showed that unprivileged individuals had significantly higher false positive rates (FPR) compared to privileged ones, meaning they were more often misclassified as high-risk. False negative rates (FNR) were lower for unprivileged groups, but the overall disparate impact score was below the recommended threshold (0.8), confirming racial bias in prediction outcomes.

To mitigate this, a Reweighing preprocessing algorithm was applied to balance the dataset by adjusting sample weights. This technique ensures that predictions are not overly influenced by historically biased data. After reweighing, the disparate impact improved, and prediction outcomes between groups became more balanced.

This analysis demonstrates how bias in AI models can unintentionally perpetuate inequality if not addressed. By conducting fairness audits and applying corrective algorithms, data scientists can ensure responsible and transparent AI deployment. Integrating ethical AI practices like fairness auditing is essential in sensitive domains such as criminal justice, healthcare, and recruitment, where model outcomes directly affect human lives.

---

**Tools Used:** Python, AI Fairness 360, Scikit-Learn, Matplotlib  
**Ethical Principle:** Fairness, Accountability, Transparency
