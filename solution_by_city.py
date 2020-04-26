import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.pipeline import Pipeline
from vecstack import StackingTransformer
from xgboost import XGBRegressor

from utils import xgb_params, lgb_params, cat_params, WeightedRegressor, mean_absolute_percentage_error, \
    get_data, cities, get_city_idxs


def main():
    df = pd.DataFrame([], columns=['Id', 'Prediction'])

    X_train, y_train, X_val, y_val, X_test, Test_ID = get_data()

    idxs = get_city_idxs()

    for city in cities:
        X_train_city = X_train[idxs[city]['train']]
        y_train_city = y_train[idxs[city]['train']]

        X_val_city = X_val[idxs[city]['val']]
        y_val_city = y_val[idxs[city]['val']]

        X_test_city = X_test[idxs[city]['test']]

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

        pipe.fit(X_train_city, y_train_city)

        logger.info('end training.')

        y_pred = pipe.predict(X_val_city)
        logger.info(f'MAPE on valid: {mean_absolute_percentage_error(y_val_city, y_pred)}')

        y_test = pipe.predict(X_test_city)
        df_city = pd.DataFrame([])
        df_city['Id'] = Test_ID[idxs[city]['test']]
        df_city['Prediction'] = y_test
        df = pd.concat([df, df_city], axis=0)

    df.to_csv('data/submission_city.csv', index=None)

    logger.info('the end!')


if __name__ == '__main__':
    main()
