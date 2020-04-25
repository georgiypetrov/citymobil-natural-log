import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from geopy import distance
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y, check_array
from vecstack import StackingTransformer
from xgboost import XGBRegressor


def get_distance(latitude, longitude, del_latitude, del_longitude):
    coord = (latitude, longitude)
    del_coord = (del_latitude, del_longitude)
    return distance.geodesic(coord, del_coord).km


def preprocess(df_kek):
    df = df_kek.copy()
    df['distance'] = df.apply(
        lambda x: get_distance(x['latitude'], x['longitude'], x['del_latitude'], x['del_longitude']), axis=1)
    df = df[['ETA', 'distance']]
    return df


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class WeightedRegressor(BaseEstimator, RegressorMixin):

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
    'seed': 1337,
    'colsample_bytree': 1,
    'silent': 1,
    'subsample': 0.75,
    'eta': 0.06,
    'objective': 'reg:tweedie',
    'max_depth': 5,
    'gamma': 0,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'tree_methods': 'gpu_hist',
    'n_estimators': 200,
    'reg_alpha': 0.1,
}

lgb_params = {
    'seed': 1337,
    'colsample_bytree': 1,
    'silent': 1,
    'subsample': 0.75,
    'eta': 0.06,
    'objective': 'tweedie',
    'max_depth': 5,
    'gamma': 0,
    'min_child_weight': 0.19,
    'n_estimators': 200,
    'reg_alpha': 0.1,
    'n_jobs': -1
}

cat_params = {
    'silent': True,
    'subsample': 0.8,
    'eta': 0.035,
    'objective': 'MAE',
    'depth': 6,
    'n_estimators': 200,
    'random_state': 1337,
    'l2_leaf_reg': 13,
    'random_strength': 0.0001,
    'thread_count': -1
}

df_train = pd.read_csv('data/train.csv', date_parser=['OrderedDate'])
df_val = pd.read_csv('data/validation.csv', date_parser=['OrderedDate'])
df_test = pd.read_csv('data/test.csv', date_parser=['OrderedDate'])

X_train = preprocess(df_train)
y_train = df_train['RTA']

X_val = preprocess(df_val)
y_val = df_val['RTA']

X_test = preprocess(df_test)

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

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_val)
print(mean_absolute_percentage_error(y_pred, y_val))

y_test = pipe.predict(X_test)
df_test['Prediction'] = y_test
df_test = df_test[['Id', 'Prediction']]
df_test.to_csv('data/submission.csv', index=None)
