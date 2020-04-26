from utils import preprocess, PROCESSED_DATA
import pandas as pd
import joblib

df_train = pd.read_csv('train.csv', parse_dates=['OrderedDate'])
df_val = pd.read_csv('validation.csv', parse_dates=['OrderedDate'])
df_test = pd.read_csv('test.csv', parse_dates=['OrderedDate'])

X_train = preprocess(df_train)
y_train = df_train['RTA']
X_val = preprocess(df_val)
y_val = df_val['RTA']
X_test = preprocess(df_test)

data = (X_train, y_train, X_val, y_val, X_test, df_test['Id'])

joblib.dump(data, PROCESSED_DATA)
