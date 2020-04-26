import pandas as pd
from sklearn.model_selection import train_test_split


def merge_crossroads(df, crossroads):
    if 'id' in crossroads.columns:
        crossroads.rename(columns={'id': 'Id'}, inplace=True)
    crossroads.drop_duplicates(subset='Id', inplace=True)
    new_df = df.merge(crossroads, on='Id', how='left')
    for column in ['p200', 'p500', 'p1000']:
        new_df[column] = new_df[column].fillna(value=new_df[column].mean())
    return new_df


df_train = pd.read_csv('data/train.csv', parse_dates=['OrderedDate'])
df_train_old = pd.read_csv('old_data/train.csv', parse_dates=['OrderedDate'])

df_valid = pd.read_csv('data/validation.csv', parse_dates=['OrderedDate'])
df_valid_old = pd.read_csv('old_data/validation.csv', parse_dates=['OrderedDate'])

df_test = pd.read_csv('data/test.csv', parse_dates=['OrderedDate'])
df_old_test = pd.read_csv('old_data/test_additional.csv', parse_dates=['OrderedDate', 'GoodArrived', 'ClientCollected'])
df_old_test['RTA'] = (df_old_test['GoodArrived'] - df_old_test['ClientCollected']).dt.seconds.astype('float64')

# Добавляем перекрестки
try:
    train_crossroads = pd.read_csv('data/train_crossroads.csv')
    valid_crossroads = pd.read_csv('data/validation_crossroads.csv')
    test_crossroads = pd.read_csv('data/test_crossroads.csv')
    train_crossroads_old = pd.read_csv('old_data/train_crossroads.csv')
    valid_crossroads_old = pd.read_csv('old_data/validation_crossroads.csv')
    test_crossroads_old = pd.read_csv('old_data/test_crossroads.csv')

    df_train = merge_crossroads(df_train, train_crossroads)
    df_valid = merge_crossroads(df_valid, valid_crossroads)
    df_test = merge_crossroads(df_test, test_crossroads)
    df_train_old = merge_crossroads(df_train_old, train_crossroads_old)
    df_valid_old = merge_crossroads(df_valid_old, valid_crossroads_old)
    df_old_test = merge_crossroads(df_old_test, test_crossroads_old)
except Exception as exc:
    print(f'WARNING: Cannot run crossroads merging (passed). Got: {exc}')


df_val_new, df_old_test = train_test_split(df_old_test, test_size=0.25, random_state=1337)

df_train_new = pd.concat(
    [df_train.sample(600000, random_state=1337), df_train_old.sample(30000, random_state=1337), df_valid,
     df_valid_old, df_old_test], axis=0)

df_test_new = df_test

df_train_new.to_csv('train.csv', index=False)
df_val_new.to_csv('validation.csv', index=False)
df_test_new.to_csv('test.csv', index=False)
