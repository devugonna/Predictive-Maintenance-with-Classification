'''
    training.py
tunes the winning model with an assortment of tested hyperparameters
and fitted, displaying the best params and threshold. for a more
fair experiment, the function is non specific so the non
engineering metrics are put through the same function for different 
tuning if neccessary.
Simply fit and saved, only using training split of data, keeping
test split unseen.
'''
import pandas as pd
from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV

from config import random_state, STATE



def hyperparam_tuning(winning_pipeline, X_train, y_train, cv):
    '''
    returns best hyperparams for a given model
    '''
    grid_tune = {'n_estimators': [10, 25, 50, 100, 200], 'max_depth':[None, 5, 10], 'min_samples_split': [2,3,5,8]}
    grid = GridSearchCV(winning_pipeline, grid_tune, n_jobs=-1, cv=cv, scoring='average_precision')
    grid.fit(X_train, y_train)
    print("best params:", grid.best_params_)
    print("best CV PR-AUC:", round(grid.best_score_, 3))
    return grid.best_estimator_
    
    
def threshold_tuning(best_model, X_train, y_train, cv):
    '''
    returns best threshold for a given model
    '''
    tune = TunedThresholdClassifierCV(best_model, scoring='f1', cv=cv, thresholds=200, random_state=random_state)
    tune.fit(X_train, y_train)
    print("threshold:", round(tune.best_threshold_, 3), " CV F1:", round(tune.best_score_, 3))

    return tune
    
def run():
    '''
    tunes and saves the models for the model with engineering features,
    and without using the derived features.
    '''
    winning_pipeline, X_train, y_train, cv, non_eng_X_train = STATE['winning_pipeline'], STATE['X_train'], STATE['y_train'], STATE['cv'], STATE['non_eng_X_train']
    best_model = hyperparam_tuning(winning_pipeline, X_train, y_train, cv)
    best_tuning = threshold_tuning(best_model, X_train, y_train, cv)
    non_eng_model = hyperparam_tuning(winning_pipeline, non_eng_X_train, y_train, cv)
    non_eng_tuning = threshold_tuning(non_eng_model, non_eng_X_train, y_train, cv)
    STATE.update(best_model=best_model, best_tuning=best_tuning, non_eng_tune=non_eng_tuning, non_eng_model=non_eng_model)

if __name__ == "__main__":
    import data_load
    import model_selection
    data_load.run()
    model_selection.run()
    run()
    
    
    


    
