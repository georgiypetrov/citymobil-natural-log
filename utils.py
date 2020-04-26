from datetime import timedelta

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array

PROCESSED_DATA = 'processed_data'


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
    df = set_city_time_by_timezone(df, 338, 3)
    df = set_city_time_by_timezone(df, 22402, 5)
    df = set_city_time_by_timezone(df, 22406, 5)
    df = set_city_time_by_timezone(df, 22394, 4)
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
    df['weekend'] = (df['dow'] >= 6) | (df_kek['OrderedDate'] == '2020-02-22') | (
            df_kek['OrderedDate'] == '2020-02-24') | (df_kek['OrderedDate'] == '2020-03-09') | (
                            df_kek['OrderedDate'] >= '2020-03-30') | (df_kek['OrderedDate'] == '2020-03-07')
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


def preprocess(df_kek):
    """
    Extract features from initial dataframe.
    :param df_kek: init dataframe.
    :return: preprocessed dataframe.
    """
    df = pd.DataFrame([])
    df['ETA'] = df_kek['ETA']
    df['EDA'] = df_kek['EDA']
    df['ESP'] = df['EDA'] / df['ETA']
    df['EXZ'] = df['ETA'] / df['EDA']
    df['city_id'] = df_kek['main_id_locality']
    df['p200'] = df_kek['p200']
    df['p500'] = df_kek['p500']
    df['p1000'] = df_kek['p1000']
    df = pd.concat([df, pd.get_dummies(df['city_id'], prefix='city')], axis=1)
    df = pd.concat([df, add_time_features(set_time_by_timezone(df_kek))], axis=1)
    df = pd.concat([df, add_distance_features(df_kek)], axis=1)

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
    'colsample_bytree': 0.7890864150842599, 'gamma': 0.05404663287513706, 'learning_rate': 0.02934427899673602,
    'max_depth': 12, 'min_child_weight': 1.76376145852182, 'n_estimators': 180, 'objective': 'reg:tweedie',
    'reg_alpha': 0.039693131533982774, 'seed': 1337, 'subsample': 0.8783295983009739
}

lgb_params = {
    'colsample_bytree': 0.8449687436723844, 'learning_rate': 0.0744244665657087, 'max_depth': 15,
    'min_child_weight': 1.921203193035535, 'n_estimators': 160, 'objective': 'tweedie',
    'reg_alpha': 0.062349993873795945, 'seed': 1337, 'subsample': 0.7928740108261731, 'num_leaves': 32768
}

cat_params = {
    'l2_leaf_reg': 1.3539377384230196, 'learning_rate': 0.2520401525832272, 'max_depth': 14, 'n_estimators': 90,
    'objective': 'MAE', 'random_state': 1337, 'random_strength': 0.08231902261750074, 'silent': True,
    'subsample': 0.7575674870857737
}
