import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.pipeline import Pipeline
from vecstack import StackingTransformer
from xgboost import XGBRegressor

from utils import preprocess, xgb_params, lgb_params, cat_params, WeightedRegressor, mean_absolute_percentage_error


def main():
    logger.info('start reading...')
    df_train = pd.read_csv('data/train_with_arrived_error_q80.csv', parse_dates=['OrderedDate'])
    df_val = pd.read_csv('data/validation.csv', parse_dates=['OrderedDate'])
    df_test = pd.read_csv('data/test.csv', parse_dates=['OrderedDate'])

    df_train = df_train.sample(250000, random_state=1337)

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
                                random_state=1337)
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
