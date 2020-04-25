import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from geopy import distance
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y, check_array
from vecstack import StackingTransformer
from xgboost import XGBRegressor


def get_distance(latitude, longitude, del_latitude, del_longitude):
    """
    Get distance from start and destination coordinates.
    :param latitude: latitude coord
    :param longitude: longitude coord
    :param del_latitude: destination latitude coord
    :param del_longitude: destination longitude coord
    :return: distance in km
    """
    coord = (latitude, longitude)
    del_coord = (del_latitude, del_longitude)
    return distance.geodesic(coord, del_coord).km


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
    df['distance'] = df_kek.apply(
        lambda x: get_distance(x['latitude'], x['longitude'], x['del_latitude'], x['del_longitude']), axis=1)
    df['distance_dest_from_center'] = df_kek.apply(
        lambda x: get_distance(x['center_latitude'], x['center_longitude'], x['del_latitude'], x['del_longitude']),
        axis=1)
    df['distance_start_from_center'] = df_kek.apply(
        lambda x: get_distance(x['center_latitude'], x['center_longitude'], x['latitude'], x['longitude']), axis=1)
    df = pd.concat([df, pd.get_dummies(df_kek['main_id_locality'], prefix='City')], axis=1)
    return df


def preprocess(df_kek):
    """
    Extract features from initial dataframe.
    :param df_kek: init dataframe
    :return: preprocessed dataframe
    """
    df = pd.DataFrame([])
    df['ETA'] = df_kek['ETA']
    df['EDA'] = df_kek['EDA']
    df['ESP'] = df['EDA'] / df['ETA']
    df = pd.concat([df, add_time_features(df_kek)], axis=1)
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
    'colsample_bytree': 0.6821852734352154,
    'gamma': 0.4553674559967951,
    'learning_rate': 0.06310339310159835,
    'max_depth': 12,
    'min_child_weight': 2.6876210444837954,
    'n_estimators': 200,
    'objective': 'reg:tweedie',
    'reg_alpha': 0.05950096345795112,
    'seed': 1337,
    'subsample': 0.8358148287882489
}

lgb_params = {
    'colsample_bytree': 0.8422252658916799,
    'learning_rate': 0.0445869385219306,
    'max_depth': 12,
    'min_child_weight': 0.4223370679383952,
    'n_estimators': 280,
    'objective': 'tweedie',
    'reg_alpha': 0.11493392438491513,
    'seed': 1337,
    'subsample': 0.752202953524499
}

cat_params = {
    'l2_leaf_reg': 4.002663589178608,
    'learning_rate': 0.09816790972650707,
    'max_depth': 12,
    'n_estimators': 260,
    'objective': 'MAE',
    'random_state': 1337,
    'random_strength': 0.03565561790725354,
    'silent': True,
    'subsample': 0.8274371361968814
}


def main():
    logger.info('start reading...')
    df_train = pd.read_csv('data/train.csv', parse_dates=['OrderedDate'])
    df_val = pd.read_csv('data/validation.csv', parse_dates=['OrderedDate'])
    df_test_old = pd.read_csv('data/test_additional.csv', parse_dates=['OrderedDate', 'GoodArrived', 'ClientCollected'])
    df_test = pd.read_csv('data/test.csv', parse_dates=['OrderedDate'])

    df_test_old['RTA'] = (df_test_old['GoodArrived'] - df_test_old['ClientCollected']).dt.seconds

    logger.info('end reading')
    logger.info('start preprocessing...')

    X_train = preprocess(df_train)
    y_train = df_train['RTA']

    X_val = preprocess(df_val)
    y_val = df_val['RTA']

    X_test_old = preprocess(df_test_old)
    y_test_old = df_test_old['RTA']

    X_test = preprocess(df_test)

    logger.info('end preprocessing.')

    estimators = [
        ('xgb', XGBRegressor(**xgb_params)),
        ('lgb', LGBMRegressor(**lgb_params)),
        ('cat', CatBoostRegressor(**cat_params))
    ]

    final_estimator = WeightedRegressor()

    stack = StackingTransformer(estimators=estimators, variant='A', regression=True, n_folds=5, shuffle=False,
                                random_state=None)
    steps = [('stack', stack),
             ('final_estimator', final_estimator)]
    pipe = Pipeline(steps)

    logger.info('start training...')

    pipe.fit(X_train, y_train)

    logger.info('end training.')

    y_pred = pipe.predict(X_val)
    logger.info(f'MAPE on valid: {mean_absolute_percentage_error(y_pred, y_val)}')

    y_pred = pipe.predict(X_test_old)
    logger.info(f'MAPE on old test: {mean_absolute_percentage_error(y_pred, y_test_old)}')

    y_test = pipe.predict(X_test)
    df_test['Prediction'] = y_test
    df_test = df_test[['Id', 'Prediction']]
    df_test.to_csv('data/submission.csv', index=None)

    logger.info('the end!')


if __name__ == '__main__':
    main()
