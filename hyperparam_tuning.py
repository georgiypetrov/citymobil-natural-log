from functools import partial

import fire
import pandas as pd
from catboost import CatBoostRegressor
from hyperopt import hp, fmin, tpe
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.pipeline import Pipeline
from vecstack import StackingTransformer
from xgboost import XGBRegressor

from baseline import mean_absolute_percentage_error, WeightedRegressor, preprocess

df_train = pd.read_csv('data/train.csv', parse_dates=['OrderedDate'])
df_val = pd.read_csv('data/validation.csv', parse_dates=['OrderedDate'])

df_train = df_train.sample(250000, random_state=1337)

X_train = preprocess(df_train)
y_train = df_train['RTA']

X_val = preprocess(df_val)
y_val = df_val['RTA']


def objective(params, keker):
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])

    if keker is LGBMRegressor:
        params['num_leaves'] = 2 ** params['max_depth']

    clf = keker(**params)

    estimators = [
        ('clf', clf),
    ]

    final_estimator = WeightedRegressor()

    stack = StackingTransformer(estimators=estimators, variant='A', regression=True, n_folds=3, shuffle=False,
                                random_state=None)
    steps = [('stack', stack),
             ('final_estimator', final_estimator)]
    pipe = Pipeline(steps)

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    score = mean_absolute_percentage_error(y_val, y_pred)
    logger.info(f'MAPE on valid: {score}, params: {params}')
    return score


xgb_space = {
    'colsample_bytree': 0.9957950833423739,
    'gamma': 0.9890937928764717,
    'learning_rate': 0.02661201974949031,
    'max_depth': 13,
    'min_child_weight': 1.2145058849488062,
    'n_estimators': 230,
    'objective': 'reg:tweedie',
    'reg_alpha': 0.1377712027161633,
    'seed': 1337,
    'subsample': 0.9298354406857103
}

lgb_space = {
    'max_depth': hp.quniform('max_depth', 2, 16, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'subsample': hp.uniform('subsample', 0.3, 1.0),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'seed': hp.choice('seed', [1337]),
    'objective': hp.choice('objective', ['tweedie']),
    'n_estimators': hp.quniform('n_estimators', 60, 300, 10),
    'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.3),
    'min_child_weight': hp.uniform('min_child_weight', 0.2, 4),
}

cat_space = {
    'max_depth': hp.quniform('max_depth', 2, 16, 1),
    'subsample': hp.uniform('subsample', 0.3, 1.0),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'random_state': hp.choice('random_state', [1337]),
    'objective': hp.choice('objective', ['MAE']),
    'silent': hp.choice('silent', [True]),
    'n_estimators': hp.quniform('n_estimators', 60, 300, 10),
    'random_strength': hp.uniform('random_strength', 0.00001, 0.1),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0.1, 10),
}

xgb_objective = partial(objective, keker=XGBRegressor)
lgb_objective = partial(objective, keker=LGBMRegressor)
cat_objective = partial(objective, keker=CatBoostRegressor)


def main(regressor, max_evals=100):
    if regressor not in ['cat', 'xgb', 'lgb']:
        raise Exception('undefined regressor')
    best = None

    logger.info(f'start optimizing {regressor}')

    if regressor == 'xgb':
        best = fmin(fn=xgb_objective,
                    space=xgb_space,
                    algo=tpe.suggest,
                    max_evals=max_evals)

    if regressor == 'lgb':
        best = fmin(fn=lgb_objective,
                    space=lgb_space,
                    algo=tpe.suggest,
                    max_evals=max_evals)

    if regressor == 'cat':
        best = fmin(fn=cat_objective,
                    space=cat_space,
                    algo=tpe.suggest,
                    max_evals=max_evals)

    logger.info(best)


if __name__ == '__main__':
    fire.Fire(main)
