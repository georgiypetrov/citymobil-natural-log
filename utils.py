from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array


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
    df['night'] = (df['hour'] == 23) | (df['hour'] <= 6)
    df['morning'] = (df['hour'] >= 7) & (df['hour'] <= 10)
    df['day'] = (df['hour'] >= 11) & (df['hour'] <= 16)
    df['evening'] = (df['hour'] >= 17) & (df['hour'] <= 22)
    df = pd.concat([pd.get_dummies(df['dow'], prefix='dow'), df], axis=1)
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
    :return: dataframe with distance features
    """

    crossroads = pd.read_csv(f'{name}_crossroads.csv')
    df = df_kek.merge(crossroads, on='Id')

    df['p200'].fillna(value=df['p200'].mean(), inplace=True)
    df['p500'].fillna(value=df['p500'].mean(), inplace=True)
    df['p1000'].fillna(value=df['p1000'].mean(), inplace=True)
    return df


def preprocess(df_kek, name=None):
    """
    Extract features from initial dataframe.
    :param df_kek: init dataframe
    :return: preprocessed dataframe
    """
    if not name:
        raise AttributeError('Cannot find dataset name!')
    df = pd.DataFrame([])
    df['ETA'] = df_kek['ETA']
    df['EDA'] = df_kek['EDA']
    df['ESP'] = df['EDA'] / df['ETA']
    df = pd.concat([df, add_time_features(set_time_by_timezone(df_kek))], axis=1)
    df = pd.concat([df, add_distance_features(df_kek)], axis=1)

    df = add_crossroads(df, name=name)
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


xgb_params = {
    'colsample_bytree': 0.8352388561415909, 'gamma': 0.2043721253945482, 'learning_rate': 0.06857932105137683,
    'max_depth': 16, 'min_child_weight': 2.766048148395735, 'n_estimators': 90, 'objective': 'reg:tweedie',
    'reg_alpha': 0.224422036680493, 'seed': 1337, 'subsample': 0.6769895125085235
}

lgb_params = {
    'colsample_bytree': 0.7091469058999612, 'learning_rate': 0.07251773698080877, 'max_depth': 8,
    'min_child_weight': 0.7681687439259843, 'n_estimators': 260, 'objective': 'mae',
    'reg_alpha': 0.10756807176613815, 'seed': 1337, 'subsample': 0.30374315411765795, 'num_leaves': 256
}

cat_params = {
    'l2_leaf_reg': 4.405217231589936, 'learning_rate': 0.18677256599482014, 'max_depth': 10, 'n_estimators': 80,
    'objective': 'MAE', 'random_state': 1337, 'random_strength': 0.011276399896596003, 'silent': True,
    'subsample': 0.653407636532965
}
