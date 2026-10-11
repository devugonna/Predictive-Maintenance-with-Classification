# Predictive-Maintenance-with-Classification

Predicting machine failure with the AI4I 2020 dataset, focussing on feature engineering and assessing viability of the additional engineering metrics created through data modelling. Utilises scikit-learn for machine learning library, Pandas and NumPy for database and mathematical functions and Matplotlib for figure generation.
Seperate model trained and tuned without feature engineered metrics for comparison to main model. 

# Overview

Predictive maintenance is used to flag machines with the likelihood of failure, prioritising intelligent predictions that do not take advantage of a seemingly high accuracy obtained when predicting no failure. only 3.4% of rows in the dataset are failure, so the metric that is the most misleading is accuracy, with a 96.6% rate of a model that guesses no failures. 
- **Missed failure (false negative):** unplanned downtime and possible damage.
- **False alarm (false positive):** an unnecessary inspection.
Both these possibilities are detrimental to good resource management, with additional inspections and unplanned damages being costly. Due to this, model baselines are assessed with average precision **(PR-AUC)**.

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
Model information can be found here for further information on the dataset

# Approach

- Uses stratified 5 fold cross-validation before any model training. Trained on a baseline assortment of 4 models: A dummy classifier, Logistic Regression, KNeighbors and Random Forest, in which the model with the highest **(PR-AUC)** is used. In theory any model and pipeline can be inserted in the code if compatible with scikit-learn.
- Introduces *Power [W]*, *Temp diff [K]*, *Strain* and an encoded *Type (L/M/H)* column that is utilised or dropped, dependant on the model that is being assessed.
- Coded with a configured 80/20 stratified split, with seed 42.
- Hyperparameter tuned with *GridSearchCV* against average precision and threshold tuned on *TunedThresholdClassifierCV*, optimising for F1 on training data only.
- Test set imported and ran once, alongside permutation importances.
- Pipeline ran on the raw measurements, without the engineered features, has its own training, to prevent advantages.

# Baseline Cross-Validation results

| Model | Avg. precision | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| Dummy (always "no failure") | 0.034 | 0.000 | 0.000 | 0.000 | 0.966 |
| Logistic regression | 0.454 | 0.845 | 0.163 | 0.273 | 0.847 |
| k-nearest neighbours | 0.583 | 0.428 | 0.834 | 0.563 | 0.978 |
| Random forest | 0.883 | 0.753 | 0.982 | 0.850 | 0.991 |

# Final evaluation on held out test set

| Model | Threshold | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Tuned random forest, engineered features | 0.46 | 0.871 | 0.794 | 0.831 | 0.879 |
| Tuned random forest, raw features | n/a | 0.746 | 0.735 | 0.741 | 0.795 
