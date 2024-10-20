from loaddata import loadData
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preProcess():
    categories = {
        'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 
                      'State-gov', 'Without-pay', 'Never-worked'],
        'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 
                      'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', 
                      '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
        'marital.status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 
                           'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
        'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
                       'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
                       'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
                       'Transport-moving', 'Priv-house-serv', 'Protective-serv', 
                       'Armed-Forces'],
        'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 
                         'Other-relative', 'Unmarried'],
        'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
        'sex': ['Female', 'Male'],
        'native.country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 
                           'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 
                           'India', 'Japan', 'Greece', 'South', 'China', 
                           'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 
                           'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 
                           'Ireland', 'France', 'Dominican-Republic', 'Laos', 
                           'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 
                           'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 
                           'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 
                           'Peru', 'Hong', 'Holand-Netherlands']
    }

    df_train, df_test = loadData()
    
    df_train.replace('?', np.nan, inplace=True)

    for col in df_train.columns[df_train.isnull().any()].tolist():
        df_train[col] = df_train[col].fillna(df_train[col].mode()[0])
    
    for col, cats in categories.items():
        df_train[col] = pd.Categorical(df_train[col], categories=cats)
        df_test[col] = pd.Categorical(df_test[col], categories=cats)

    categorical_cols = list(categories.keys())
    df_train_encoded = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)
    df_test_encoded = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)

    continuous_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    scaler = StandardScaler()
    
    df_train_encoded[continuous_cols] = scaler.fit_transform(df_train_encoded[continuous_cols])
    df_test_encoded[continuous_cols] = scaler.transform(df_test_encoded[continuous_cols])

    return df_train_encoded, df_test_encoded

if __name__ == '__main__':
    preProcess()