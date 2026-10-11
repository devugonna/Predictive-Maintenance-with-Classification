'''
    data_load.py
    
loads data from data folder, includes data modeling and 
derivations of different engineering metrics, used to 
assess the importance of using the metrics vs without
later on. 

'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from config import DATA_PATH, STATE


def load_raw_data(path=DATA_PATH):
    #loads dataframe as df
    df = pd.read_csv(path)
    return df
    
    
def compile_engineering_metrics(df):
    '''
    added power (torque x rotational speed x 2π/60)
    added temperature difference, between process and air temps
    added strain, tool wear x torque
    encoded through ordinalencoder type, with low, med, high type
    being represented by 0.0, 1.0, 2.0 with a fit_transform
    the engineering metrics are not used in the comparitive model
    and assesed there.
    
    '''
    df_eng = df
    df_eng.insert(3, 'Power [W]', 0.0)
    df_eng.insert(4, 'Temp diff [K]', 0.0)
    df_eng.insert(5, 'Strain', 0.0)
    df_eng['Power [W]'] = df_eng['Torque [Nm]'] * df_eng['Rotational speed [rpm]'] * 2 * np.pi / 60
    df_eng['Temp diff [K]'] = df_eng['Process temperature [K]'] - df_eng['Air temperature [K]']
    df_eng['Strain'] = df_eng['Tool wear [min]'] * df_eng['Torque [Nm]']
    
    encodetype = OrdinalEncoder(categories = [['L','M','H']])
    df['Type'] = encodetype.fit_transform(df[['Type']])
    
    return df_eng

def inspect(df):
    '''
    prints database inspections
    '''
    print("Shape: ",df.shape)
    print("\n Dtpye counts:\n", df.dtypes.value_counts())
    print("\n Summary: ", df.describe())
    print("\n Information: \n")
    df.info()
    print("\n The UID is/is not unique: ", df['UDI'].is_unique)
    print(df.head(3),"\n",df.tail(3))

def model_categorisation(df):
    '''
    the main model uses columns 2:6 and 8:11 as its X category, and the binary
    Machine Failure column as its target cat for classification. Non eng uses 
    different categories in its model for comparison
    ''' 
    model_columns = df.columns
    eng_feature_categories = df[list(model_columns[2:6]) + list(model_columns[8:11])]
    target_category = df['Machine failure']
    non_eng_feature_categories = df[[model_columns[2]] + list(model_columns[6:11])]
    
    return eng_feature_categories, target_category, non_eng_feature_categories

def run():
    '''
    alongside STATE dict, there is a run to update state with all variables that
    will need to be fetched later on.
    saves the X, y and the X cats for the non engineered feature models too
    '''
    df = load_raw_data()
    df_eng = compile_engineering_metrics(df)
    X, y, non_eng_X = model_categorisation(df_eng)
    inspect(df_eng)
    STATE.update(df_eng=df_eng, X=X, y=y, non_eng_X=non_eng_X
    )
if __name__ == "__main__":
    run()
    
