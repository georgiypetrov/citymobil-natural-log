import pandas as pd
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('data/train.csv', parse_dates=['OrderedDate'])
df_train_old = pd.read_csv('old_data/train.csv', parse_dates=['OrderedDate'])

df_valid = pd.read_csv('data/validation.csv', parse_dates=['OrderedDate'])
df_valid_old = pd.read_csv('old_data/validation.csv', parse_dates=['OrderedDate'])

df_test = pd.read_csv('data/test.csv', parse_dates=['OrderedDate'])
df_old_test = pd.read_csv('old_data/test_additional.csv', parse_dates=['OrderedDate', 'GoodArrived', 'ClientCollected'])
df_old_test['RTA'] = (df_old_test['GoodArrived'] - df_old_test['ClientCollected']).dt.seconds

df_val_new, df_old_test = train_test_split(df_old_test, test_size=0.25, random_state=1337)

df_train_new = pd.concat(
    [df_train.sample(150000, random_state=1337), df_train_old.sample(150000, random_state=1337), df_valid,
     df_valid_old, df_old_test], axis=0)

df_test_new = df_test

df_train_new.to_csv('train.csv', index=False)
df_val_new.to_csv('validation.csv', index=False)
df_test_new.to_csv('test.csv', index=False)
