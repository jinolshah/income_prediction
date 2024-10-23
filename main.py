import loaddata
import preprocess
import train

df_train, df_test = loaddata.loadData()
df_train, df_test = preprocess.preProcess(df_train, df_test)

train.train(df_train, df_test, 'GradientBoosting')