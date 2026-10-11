# Predictive-Maintenance-with-Classification

Predicting machine failure with the AI4I 2020 dataset, focussing on feature engineering and assessing viability of the additional engineering metrics created through data modelling.

# Overview

Predictive maintenance is used to flag machines with the likelihood of failure, prioritising intelligent predictions that do not take advantage of a seemingly high accuracy obtained when predicting no failure. only 3.4% of rows in the dataset are failure, so the metric that is the most misleading is accuracy, with a 96.6% rate of a model that guesses all failure. 
- **Missed failure (false negative):** unplanned downtime and possible damage.
- **False alarm (false positive):** an unnecessary inspection.
Both these possibilities are detrimental to good resource management, with additional inspections and unplanned damages being costly. Due to this, model baselines are assessed with average precision **(PR-AUC)**.

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
Model information can be found here for further information on the dataset

# Approach

- Introduces *Power [W]*, *Temp diff [K]*,*Strain* and an encoded *Type (L/M/H)* column that is utilised or dropped, dependant on the model that is being assesed.
