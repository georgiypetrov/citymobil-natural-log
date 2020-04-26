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

df_valid = pd.read_csv('data/validation.csv', parse_dates=['OrderedDate'])

df_test = pd.read_csv('data/test_additional.csv', parse_dates=['OrderedDate'])


# Добавляем перекрестки
try:
    train_crossroads = pd.read_csv('data/train_crossroads.csv')
    valid_crossroads = pd.read_csv('data/valid_crossroads.csv')
    test_crossroads = pd.read_csv('data/test_crossroads.csv')

    df_train = merge_crossroads(df_train, train_crossroads)
    df_valid = merge_crossroads(df_valid, valid_crossroads)
    df_test = merge_crossroads(df_test, test_crossroads)
except Exception as exc:
    print(f'WARNING: Cannot run crossroads merging (passed). Got: {exc}')


df_val_old, df_val_new = train_test_split(df_valid, test_size=0.25, random_state=1337)

df_train_new = pd.concat([df_train, df_val_old], axis=0)

df_train_new.to_csv('train.csv', index=False)
df_val_new.to_csv('validation.csv', index=False)
df_test.to_csv('test.csv', index=False)
