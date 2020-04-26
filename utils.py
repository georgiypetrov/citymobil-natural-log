from datetime import timedelta

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array

cities = [1078, 22390, 22430, 22438]


def get_distance(lat1, lon1, lat2, lon2):
    """
    Get distance from two coordinates.
    :param lat1: first latitude coord
    :param lon1: first longitude coord
    :param lat2: second latitude coord
    :param lon2: second longitude coord
    :return: distance in km
    """
    KM = 6371.393
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = KM * c
    return distance


def set_city_time_by_timezone(df, city_id, hour_diff):
    """
    Shift time by timezone to city data
    :param df:
    :param city_id:
    :param hour_diff:
    :return: data with updated time
    """
    df[df.main_id_locality == city_id][['OrderedDate']].apply(lambda x: x - timedelta(hours=hour_diff))
    return df


def set_time_by_timezone(df):
    """
    Shift time by timezone to all cities
    :param df:
    :return:
    """
    df = set_city_time_by_timezone(df, 1078, 3)
    df = set_city_time_by_timezone(df, 22390, 4)
    df = set_city_time_by_timezone(df, 22430, 4)
    df = set_city_time_by_timezone(df, 22438, 5)
    return df


def add_time_features(df_kek):
    """
    Extract time related features.
    :param df_kek: init dataframe
    :return: dataframe with time features
    """
    df = pd.DataFrame([])
    df['hour'] = df_kek['OrderedDate'].dt.hour
    df['dow'] = df_kek['OrderedDate'].dt.dayofweek
    df = pd.concat([pd.get_dummies(df['dow'], prefix='dow'), pd.get_dummies(df['hour'], prefix='hour')], axis=1)
    return df


def add_distance_features(df_kek):
    """
    Extract distance related features.
    :param df_kek: init dataframe
    :return: dataframe with distance features
    """
    df = pd.DataFrame([])
    df['distance'] = get_distance(df_kek['latitude'], df_kek['longitude'], df_kek['del_latitude'],
                                  df_kek['del_longitude'])
    df['distance_start_from_center'] = get_distance(df_kek['latitude'], df_kek['longitude'], df_kek['center_latitude'],
                                                    df_kek['center_longitude'])
    df['distance_dest_from_center'] = get_distance(df_kek['center_latitude'], df_kek['center_longitude'],
                                                   df_kek['del_latitude'], df_kek['del_longitude'])
    return df


def add_crossroads(df_kek, name):
    """
    Read crossroads file and merge it to given df.
    :param df_kek: init dataframe
    :param name: name of dataset.
    :return: dataframe with distance features
    """

    crossroads = pd.read_csv(f'data/{name}_crossroads.csv')
    df = df_kek.merge(crossroads, on='Id', how='left')

    df['p200'].fillna(value=df['p200'].mean(), inplace=True)
    df['p500'].fillna(value=df['p500'].mean(), inplace=True)
    df['p1000'].fillna(value=df['p1000'].mean(), inplace=True)
    return df


def preprocess(df_kek, crossroads=False, name=None):
    """
    Extract features from initial dataframe.
    :param df_kek: init dataframe.
    :param crossroads: True if use crossroads feature
    :param name: name of dataset.
    :return: preprocessed dataframe.
    """
    if crossroads and not name:
        raise AttributeError('Cannot find dataset name!')
    df = pd.DataFrame([])
    df['ETA'] = df_kek['ETA']
    df['EDA'] = df_kek['EDA']
    df['ESP'] = df['EDA'] / df['ETA']
    df['city_id'] = df_kek['main_id_locality']
    df = pd.concat([df, add_time_features(set_time_by_timezone(df_kek))], axis=1)
    df = pd.concat([df, add_distance_features(df_kek)], axis=1)

    if crossroads:
        # Добавляем временно `Id`, чтобы правильно смержить перекрестки
        df['Id'] = df_kek['Id']
        df = add_crossroads(df, name=name)
        df.drop(columns=['Id'], inplace=True)
    return df


def mean_absolute_percentage_error(y_true, y_pred):
    """
    MAPE metric eval.
    :param y_true:
    :param y_pred:
    :return: MAPE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class WeightedRegressor(BaseEstimator, RegressorMixin):
    """
    Weighted arithmetic mean regressor
    """

    def __init__(self, weights=None):
        self.weights = weights

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        return self

    def predict(self, X):
        # Input validation
        X = check_array(X)

        y = np.mean(X, axis=1)
        return y


def get_data(target='RTA', crossroads=False):
    """
    Read data and return preprocessed dataframe
    :param crossroads: True if use crossroads feature
    :return: X_train, y_train, X_val, y_val, X_test, Test_ID
    """
    logger.info('start reading...')

    df_train = pd.read_csv('data/train_with_arrived_error_q80.csv', parse_dates=['OrderedDate'])
    df_val = pd.read_csv('data/validation.csv', parse_dates=['OrderedDate'])
    df_test = pd.read_csv('data/test.csv', parse_dates=['OrderedDate'])

    df_train = pd.concat([df_train, df_val], axis=0)  # remove it if you want true validation score

    logger.info('end reading')
    logger.info('start preprocessing...')

    X_train = preprocess(df_train, crossroads, name='train')
    df_train['RTA_over_ETA'] = df_train['RTA'] / df_train['ETA']
    y_train = df_train[target]

    X_val = preprocess(df_val, crossroads, name='validation')
    df_val['RTA_over_ETA'] = df_val['RTA'] / df_val['ETA']
    y_val = df_val['RTA']

    X_test = preprocess(df_test, crossroads, name='test')

    logger.info('end preprocessing.')

    return X_train, y_train, X_val, y_val, X_test, df_test['Id'], df_train['ETA'], df_val['ETA'], df_test['ETA']


def get_city_idxs():
    """
    Get city indexes mapping.
    :return: dict with indexes
    """
    df_train = pd.read_csv('data/train_with_arrived_error_q80.csv', parse_dates=['OrderedDate'])
    df_val = pd.read_csv('data/validation.csv', parse_dates=['OrderedDate'])
    df_test = pd.read_csv('data/test.csv', parse_dates=['OrderedDate'])

    df_train = pd.concat([df_train, df_val], axis=0)  # remove it if you want true validation score

    idxs = {}

    for city in cities:
        idxs[city] = {
            'train': df_train['main_id_locality'] == city,
            'val': df_val['main_id_locality'] == city,
            'test': df_test['main_id_locality'] == city
        }
    return idxs


xgb_params = {
    'colsample_bytree': 0.5301797410622617, 'gamma': 0.14668495067793016,
    'learning_rate': 0.20536349468140228, 'max_depth': 16, 'min_child_weight': 2.7432782857217655,
    'n_estimators': 230, 'objective': 'reg:tweedie', 'reg_alpha': 0.22112710555603482, 'seed': 1337,
    'subsample': 0.9992716172305873
}

lgb_params = {
    'colsample_bytree': 0.770851806404094, 'learning_rate': 0.1293171365711842, 'max_depth': 8,
    'min_child_weight': 3.694288074910049, 'n_estimators': 230, 'objective': 'mae',
    'reg_alpha': 0.21311120642722892, 'seed': 1337, 'subsample': 0.46341695004627526, 'num_leaves': 256
}

cat_params = {
    'l2_leaf_reg': 0.8989329839485665, 'learning_rate': 0.06323887930914782, 'max_depth': 15,
    'n_estimators': 230, 'objective': 'MAE', 'random_state': 1337, 'random_strength': 0.017040865775220557,
    'silent': True, 'subsample': 0.7045725254275295
}
