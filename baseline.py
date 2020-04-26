import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y, check_array
from vecstack import StackingTransformer
from xgboost import XGBRegressor


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


def main():
    logger.info('start reading...')
    df_train = pd.read_csv('data/train_with_arrived_error_q80.csv', parse_dates=['OrderedDate'])
    df_val = pd.read_csv('data/validation.csv', parse_dates=['OrderedDate'])
    # df_train = pd.concat([df_train, df_val], axis=0)
    df_test = pd.read_csv('data/test.csv', parse_dates=['OrderedDate'])

    logger.info('end reading')
    logger.info('start preprocessing...')

    X_train = preprocess(df_train)
    y_train = df_train['RTA']

    X_val = preprocess(df_val)
    y_val = df_val['RTA']

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
    logger.info(f'MAPE on valid: {mean_absolute_percentage_error(y_val, y_pred)}')

    y_test = pipe.predict(X_test)
    df_test['Prediction'] = y_test
    df_test = df_test[['Id', 'Prediction']]
    df_test.to_csv('data/submission.csv', index=None)

    logger.info('the end!')


if __name__ == '__main__':
    main()
