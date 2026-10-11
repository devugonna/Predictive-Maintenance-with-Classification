'''
    evaluate.py
runs and collates evaluation data of the tuned models,
appending them to the dataframe with the baselines, and storing them
seperately with extra metrics, such as confusion matrix.
Runs permutation importance to assess most important column for failure,
and saves the data to be used in creating visual results with matplotlib.  
'''
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
average_precision_score, confusion_matrix)

from config import random_state, STATE

def evaluation(name,best_tuning, final_results, uidX_test, y_test):	#X_test changes, want to reuse this block
    '''
    after model is fit in model training, it is ran alongside metrics,
    and the until now unseen test data. fit to a dataframe final_results
    through the .loc assigning, with index and other columns preset, and 
    assigned in run(). .predict used for the X_test but _proba used for
    collecting accuracy score. metrics saved to df and then returned.
    '''
    prediction = best_tuning.predict(uidX_test)
    p = best_tuning.predict_proba(uidX_test)[:,1]
    matrix = confusion_matrix(y_test, prediction)
    
    final_results.loc[name] = [best_tuning.best_threshold_,
    precision_score(y_test,prediction),
    recall_score(y_test,prediction),
    f1_score(y_test,prediction),
    accuracy_score(y_test,prediction),
    average_precision_score(y_test, p),
    y_test.mean(),
    matrix[0,0],
    matrix[0,1],
    matrix[1,0],
    matrix[1,1]]
    
    return 
    
  
def insert_results_to_eval(name, final_results, eval_for_models):
    '''
    for a non vital comparison, a model in final_results dataframe can be 
    put into the earlier eval_for_models with the model baselines
    '''
    eval_for_models.loc[name] = [final_results.loc[name]['pr_auc'], final_results.loc[name]['recall'], 
    final_results.loc[name]['precision'], final_results.loc[name]['f1'], final_results.loc[name]['accuracy']]
    
    return 
    
    
def cat_permutation(best_model, uidX_test, y_test):
    '''
    finds and returns the permutation importance of all used columns in a model for their
    importance and correlation to the target. compiled in a pandas series array
    '''
    perm = permutation_importance(best_model.estimator_, uidX_test, y_test,scoring='average_precision', n_repeats=30, random_state=random_state).importances_mean
    perm2 = pd.Series(perm, index=uidX_test.columns).sort_values(ascending=False)
    return perm2
    
    
def run():
    '''
    creates the col_names for final_results so it can be appended with
    .loc. probably easier as i am used to assigning an empty array []
    then iterating in a loop for assigns for a non pandas compilation.
    '''
    X_test, y_test, non_eng_X_test, best_tuning, non_eng_tune,eval_for_models =  STATE['X_test'], STATE['y_test'],STATE['non_eng_X_test'],STATE['best_tuning'], STATE['non_eng_tune'], STATE['eval_for_models']
    col_names = ['threshold', 'precision', 'recall','f1', 
    'accuracy', 'pr_auc', 'no_skill', 'TN', 'FP', 'FN', 'TP']
    final_results = pd.DataFrame(columns = col_names)
    
    evaluation('Tuned model', best_tuning, final_results, X_test, y_test)
    insert_results_to_eval('Tuned model', final_results, eval_for_models)

    evaluation('Non Engineered metrics', non_eng_tune, final_results, non_eng_X_test, y_test)
    insert_results_to_eval('Non Engineered metrics', final_results,eval_for_models)

    final_results = final_results.astype(float)
    eng_permutation = cat_permutation(best_tuning, X_test, y_test)
    non_eng_permutation = cat_permutation(non_eng_tune, non_eng_X_test, y_test)

    STATE.update(final_results=final_results, eval_for_models=eval_for_models,
	eng_permutation=eng_permutation, non_eng_permutation=non_eng_permutation)



if __name__ == "__main__":
    import data_load, model_selection, training
    data_load.run()
    model_selection.run()
    training.run()
    run()
    print(STATE['final_results'].round(3))
    print(STATE['eval_for_models'].round(3))
    print(STATE['eng_permutation'].round(3))
    print(STATE['non_eng_permutation'].round(3))
    
    
