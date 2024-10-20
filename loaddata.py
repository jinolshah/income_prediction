import os
import pandas as pd

def loadData():
    currDir = os.getcwd()
    dataDir = os.path.join(currDir, 'data')
    trainFile = os.path.join(dataDir, 'train_final.csv')
    testFile = os.path.join(dataDir, 'test_final.csv')

    df_train = pd.read_csv(trainFile)
    df_test = pd.read_csv(testFile)

    return df_train, df_test

if __name__ == '__main__':
    loadData()