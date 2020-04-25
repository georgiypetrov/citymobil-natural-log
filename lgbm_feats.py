import pandas as pd
import numpy as np
import matplotlib as plt
import polyline
import lightgbm as lgb
from datetime import date, time, datetime, timedelta
from pandarallel import pandarallel
from sklearn.preprocessing import OneHotEncoder
pandarallel.initialize()

def mean_absolute_percentage_error(y_true, y_pred):
    """
    MAPE metric eval.
    :param y_true:
    :param y_pred:
    :return: MAPE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

df_train = pd.read_csv('train.csv', parse_dates=['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived'])
df_test = pd.read_csv('validation.csv', parse_dates=['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived'])

df_train[['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived']] = df_train[df_train.main_id_locality == 338][['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived']].apply(lambda x: x - timedelta(hours=3))
df_train[['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived']] = df_train[df_train.main_id_locality == 22402][['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived']].apply(lambda x: x - timedelta(hours=5))
df_train[['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived']] = df_train[df_train.main_id_locality == 22394][['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived']].apply(lambda x: x - timedelta(hours=4))
df_train[['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived']] = df_train[df_train.main_id_locality == 22406][['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived']].apply(lambda x: x - timedelta(hours=5))

df_train = df_train.drop('Id', axis=1)

onehotencoder = OneHotEncoder(dtype=int)
data = df_train[['main_id_locality']]
data = onehotencoder.fit_transform(data).toarray() 
data = data.T

df_train['city1'] = data[0]
df_train['city2'] = data[1]
df_train['city3'] = data[2]
df_train['city4'] = data[3]

df_train['hour'] = df_train.OrderedDate.dt.hour
df_train['dow'] = df_train.OrderedDate.dt.dayofweek

df_train['routeNum'] = df_train.route.parallel_apply(lambda x: 0 if pd.isna(x) else len(polyline.decode(x)))
df_train['trackNum'] = df_train.track.parallel_apply(lambda x: 0 if pd.isna(x) else len(polyline.decode(x)))

df_train['DeltaReady'] = df_train['ReadyForCollection'] - df_train['OrderedDate']
df_train['DeltaCollected'] = df_train['ClientCollected'] - df_train['OrderedDate']
df_train['DeltaArrived'] = df_train['GoodArrived'] - df_train['OrderedDate']
df_train['DeltaReady'] = df_train['DeltaReady'].dt.seconds
df_train['DeltaCollected'] = df_train['DeltaCollected'].dt.seconds
df_train['DeltaArrived'] = df_train['DeltaArrived'].dt.seconds

df_test[['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived']] = df_test[df_test.main_id_locality == 338][['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived']].apply(lambda x: x - timedelta(hours=3))
df_test[['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived']] = df_test[df_test.main_id_locality == 22402][['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived']].apply(lambda x: x - timedelta(hours=5))
df_test[['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived']] = df_test[df_test.main_id_locality == 22394][['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived']].apply(lambda x: x - timedelta(hours=4))
df_test[['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived']] = df_test[df_test.main_id_locality == 22406][['OrderedDate', 'ReadyForCollection', 'ClientCollected', 'GoodArrived']].apply(lambda x: x - timedelta(hours=5))

df_test = df_test.drop('Id', axis=1)

onehotencoder = OneHotEncoder(dtype=int)
data = df_test[['main_id_locality']]
data = onehotencoder.fit_transform(data).toarray() 
data = data.T
df_test['city1'] = data[0]
df_test['city2'] = data[1]
df_test['city3'] = data[2]
df_test['city4'] = data[3]

df_test['hour'] = df_test.OrderedDate.dt.hour
df_test['dow'] = df_test.OrderedDate.dt.dayofweek

df_test['routeNum'] = df_test.route.parallel_apply(lambda x: 0 if pd.isna(x) else len(polyline.decode(x)))

df_test['DeltaReady'] = df_test['ReadyForCollection'] - df_test['OrderedDate']
df_test['DeltaCollected'] = df_test['ClientCollected'] - df_test['OrderedDate']
df_test['DeltaArrived'] = df_test['GoodArrived'] - df_test['OrderedDate']
df_test['DeltaReady'] = df_test['DeltaReady'].dt.seconds
df_test['DeltaCollected'] = df_test['DeltaCollected'].dt.seconds
df_test['DeltaArrived'] = df_test['DeltaArrived'].dt.seconds

feats = ['ETA', 'latitude','del_latitude','longitude','del_longitude','EDA','ready_latitude',
'ready_longitude', 'onway_latitude', 'onway_longitude', 'arrived_latitude', 'arrived_longitude',
'center_latitude', 'center_longitude', 'city1', 'city2', 'city3', 'city4', 'hour', 'dow',
        'routeNum', 'DeltaReady', 'DeltaCollected', 'DeltaArrived']

feats2 = ['ETA', 'latitude','del_latitude','longitude','del_longitude','EDA','ready_latitude',
'ready_longitude', 'onway_latitude', 'onway_longitude', 'arrived_latitude', 'arrived_longitude',
'center_latitude', 'center_longitude', 'city1', 'city2', 'city3', 'city4', 'hour', 'dow',
        'routeNum', 'DeltaReady', 'DeltaCollected', 'DeltaArrived']

df_train_x = df_train[feats2]
df_train_y = df_train[['RTA']]

df_test_x = df_test[feats2]
df_test_y = df_test['RTA']

gbm = lgb.LGBMRegressor(num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=100)
gbm.fit(df_train_x, df_train_y,
        eval_set=[(df_test_x, df_test_y)],
        eval_metric='l1',
        early_stopping_rounds=10)

y_pred = gbm.predict(df_test_x, num_iteration=gbm.best_iteration_)

print('The mape of prediction is:', mean_absolute_percentage_error(df_test_y, y_pred))

print('\nFeature importances:', list(gbm.feature_importances_))
