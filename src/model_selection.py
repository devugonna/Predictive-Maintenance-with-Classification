'''
    model_selection.py

compares the baseline of 4 models: a dummy, logistic regression,
Kneighbors, and random forest classifiers, picking the model
with the highest average precision as the winner
the baseline metrics are saved as a dataframe and the best model
is used in training and evaluation, with its pipeline being saved
for tuning. Theoretically any classifier from scikit or an xgboost
could be placed and modified by adding it and its pipeline in the
dictionary of models, however tuning might need different values.
Warning filtered for dummy model.
'''


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from config import(random_state, test_size, n_splits, STATE)
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    
def baseline_scoring(X_train, y_train):
    '''
    creates scoring metrics that is iterated through the cross validation
    function, and later sorted by avg precision.
    models holds the pipeline and name for all models, with a standard
    scaler.
    scores saved in row and appended to metric, which is then saved as
    a df. I found when appending, a list of dicts would be better than
    appending to a dataframe. I later used an empty df with columns and
    using .loc to assign and fill the df. both methods are viable.
    '''
    scoring = ['average_precision', 'recall', 'precision', 'f1', 'accuracy']
    models = {
    'dummy':	DummyClassifier(strategy='most_frequent'),
    'logreg':	make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state)),
    'knn':	make_pipeline(StandardScaler(), KNeighborsClassifier()),
    'rf':	RandomForestClassifier(class_weight='balanced', random_state=random_state)
    }
    metric = []
    for name, model in models.items():
        valscore = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        row = {'model': name}
        for s in scoring:
            row[s] = valscore['test_' + s].mean()
        metric.append(row)
    eval_for_models = pd.DataFrame(metric).set_index('model')
    
    return eval_for_models, models
    
def find_best_model(eval_for_models, models):
    '''
    sorts by avg precision and finds the first index to find the 
    winning model, and its pipeline, still in models dict,
    name from both df and dict are the same so search never breaks.
    '''
    metric_sort = eval_for_models.sort_values(by=['average_precision'], ascending=False)
    winning_model = metric_sort.index[0]
    winning_pipeline = models[winning_model]
    return winning_pipeline, winning_model
    

    

def run():
    '''
    may be verbose in bigger projects, but STATE dict is unpacked then
    updated, grabbing X, y for model baseline cross validation and 
    non eng x to find the training splits for both training data using 
    config metrics and train_test_split; y being the same in both data sets
    so not needing a unique variable - denoted in _'s.
    
    '''
    X, y, non_eng_X = STATE['X'], STATE['y'], STATE['non_eng_X']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
    stratify=y, random_state=random_state)
    non_eng_X_train, non_eng_X_test, _, _ = train_test_split(non_eng_X, y, test_size=test_size, stratify=y, random_state=random_state)
    
    eval_for_models, models = baseline_scoring(X_train, y_train)
    winning_pipeline, winning_model = find_best_model(eval_for_models, models)
    
    STATE.update(cv=cv,
                 X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                 non_eng_X_train=non_eng_X_train, non_eng_X_test=non_eng_X_test,
                 eval_for_models=eval_for_models,
                 winning_pipeline=winning_pipeline, winning_model=winning_model)




if __name__ == "__main__":
    import data_load
    data_load.run()
    run()
    print(STATE['eval_for_models'].round(3))
    print("winner:", STATE['winning_model'])
