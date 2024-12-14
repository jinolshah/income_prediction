import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

def decisionTree(X_train, y_train):
    max_depth = 50
    best_score = 0
    best_depth = 0

    print('Depth, CV Score')
    for i in range(1, max_depth):
        dt_model = DecisionTreeClassifier(random_state=42, max_depth=i)
        cv_score = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='roc_auc').mean()
        print(f'{i}, {cv_score}')
        if cv_score > best_score:
            best_score = cv_score
            best_depth = i

    dt_model = DecisionTreeClassifier(random_state=42, max_depth=best_depth)
    dt_model.fit(X_train, y_train)

    return dt_model

def randomForest(X_train, y_train):
    param_grid = {
        'n_estimators': [200],
        'max_depth': [10, 20, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    rf_model = RandomForestClassifier(random_state=42, n_estimators=300, bootstrap=False, max_features='sqrt', max_depth=30, min_samples_split=10, min_samples_leaf=2)

    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           scoring='roc_auc', cv=5, n_jobs=-1, verbose=2)

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np

def GradientBoosting(X_train, y_train):
    param_dist = {
        'n_estimators': [300],  # Increased range
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 4],
        'subsample': [1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],  # Add max_features for random feature selection
    }

    gb_model = GradientBoostingClassifier(random_state=42)

    # Set up RandomizedSearchCV
    randomized_search = RandomizedSearchCV(
        estimator=gb_model, 
        param_distributions=param_dist,
        scoring='roc_auc', 
        cv=5, 
        n_jobs=-1, 
        verbose=1,
        n_iter=400,  # Number of parameter settings sampled
        random_state=42
    )

    # Fit the randomized search
    randomized_search.fit(X_train, y_train)

    # Retrieve the best model and parameters
    best_model = randomized_search.best_estimator_
    best_params = randomized_search.best_params_
    best_score = randomized_search.best_score_

    print(f'Best Parameters: {best_params}')
    print(f'Best Cross-Validation ROC AUC Score: {best_score:.4f}')

    return best_model

def train(df_train, df_test, modelType = 'DecisionTree'):
    X_train = df_train.drop(columns=['income>50K'])
    y_train = df_train['income>50K']

    if modelType == 'DecisionTree':
        model = decisionTree(X_train, y_train)
    elif modelType == 'RandomForest':
        model = randomForest(X_train, y_train)
    elif modelType == 'GradientBoosting':
        model = GradientBoosting(X_train, y_train)

    print(cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean())

    y_pred_prob = model.predict_proba(df_test.drop(columns=['ID']))[:, 1]

    submission = pd.DataFrame({'ID': df_test['ID'], 'Prediction': y_pred_prob})
    submission.to_csv('submission.csv', index=False)