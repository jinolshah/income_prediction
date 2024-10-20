import preprocess
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def train():
    df_train, df_test = preprocess.preProcess()

    X_train = df_train.drop(columns=['income>50K'])
    y_train = df_train['income>50K']

    dt_model = DecisionTreeClassifier(random_state=42)

    dt_model.fit(X_train, y_train)

    y_pred_prob = dt_model.predict_proba(df_test.drop(columns=['ID']))[:, 1]

    submission = pd.DataFrame({'ID': df_test['ID'], 'Prediction': y_pred_prob})
    submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    train()