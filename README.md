# SVM Classification & Manual Feature Engineering

**Tools:** Python · scikit-learn · SVM · RBF Kernel · GridSearchCV · Logistic Regression

---

## Problem

Two-class data arranged in a radial pattern was not linearly separable, causing logistic regression to achieve only 58.9% accuracy, well below a useful decision threshold.

## Approach

Compared an RBF kernel SVM with gamma tuning via GridSearchCV against logistic regression. Also engineered a radial distance feature (x3) to make the classes linearly separable without a kernel, achieving perfect separation.

## Impact

Both the SVM and the feature-engineered logistic model achieved 100% accuracy and precision, demonstrating the value of kernel methods and domain-driven feature engineering for non-linear classification problems.

---

## Files

| File | Description |
|------|-------------|
| `svm_classification_feature_engineering.ipynb` | Jupyter notebook with full analysis |
| `svm_classification_feature_engineering.html` | Rendered HTML version |

---

*Part of my data & analytics portfolio — [cameronbatts.github.io](https://cameronbatts.github.io)*
