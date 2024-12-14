import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

def preProcess(df_train, df_test):
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

    #adjusting continuous cols
    continuous_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    continuos_cols_drop = ['fnlwgt', 'capital.gain', 'capital.loss']

    #binning age
    age_bins = [0, 18, 25, 32, 42, 52, 64, 200]
    age_labels = ['0-18', '19-25', '26-32', '33-42', '43-52', '53-64', '65+']
    df_train['age'] = pd.cut(df_train['age'], bins=age_bins, labels=age_labels, right=False)
    df_test['age'] = pd.cut(df_test['age'], bins=age_bins, labels=age_labels, right=False)
    continuous_cols.remove('age')

    #creating net capital column
    df_train['net_capital'] = df_train['capital.gain'] - df_train['capital.loss']
    df_test['net_capital'] = df_test['capital.gain'] - df_test['capital.loss']
    continuous_cols.append('net_capital')

    for col in continuos_cols_drop:
        df_train.drop(col, axis=1, inplace=True)
        df_test.drop(col, axis=1, inplace=True)
        continuous_cols.remove(col)

    target_enc_cols = ['native.country', 'education', 'occupation', 'age']

    categorical_cols = list(categories.keys())
    categorical_cols.append('age')

    for col in target_enc_cols:
        categorical_cols.remove(col)

    target_col = 'income>50K'

    #filling missing values with mode
    df_train.replace('?', np.nan, inplace=True)

    for col in df_train.columns[df_train.isnull().any()].tolist():
        df_train[col] = df_train[col].fillna(df_train[col].mode()[0])

    #target encoding
    for col in target_enc_cols:
        target_encoding = df_train.groupby(col)[target_col].mean()
        df_train[col] = df_train[col].map(target_encoding)
        df_test[col] = df_test[col].map(target_encoding)

    #one hot encoding
    for col, cats in categories.items():
        df_train[col] = pd.Categorical(df_train[col], categories=cats)
        df_test[col] = pd.Categorical(df_test[col], categories=cats)

    categorical_cols = list(categories.keys())
    df_train_encoded = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)
    df_test_encoded = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)

    #scaling continuous columns
    scaler = StandardScaler()
    
    df_train_encoded[continuous_cols] = scaler.fit_transform(df_train_encoded[continuous_cols])
    df_test_encoded[continuous_cols] = scaler.transform(df_test_encoded[continuous_cols])

    return df_train_encoded, df_test_encoded
