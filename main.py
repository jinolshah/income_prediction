import loaddata
import preprocess
import train

print('1. DecisionTree')
print('2. RandomForest')
print('3. GradientBoosting')
modelType = input('Enter the model number: ')

if modelType == '1':
    modelType = 'DecisionTree'
elif modelType == '2':
    modelType = 'RandomForest'
elif modelType == '3':
    modelType = 'GradientBoosting'
else:
    print('Invalid Input')
    exit()

df_train, df_test = loaddata.loadData()
df_train, df_test = preprocess.preProcess(df_train, df_test)

train.train(df_train, df_test, modelType)