from typing import Union, Any

from sklearn.base import BaseEstimator
import lightgbm as lgb
import xgboost as xgb
import catboost as cat

LGBM = Union[lgb.LGBMClassifier, lgb.LGBMRegressor]
XGB = Union[xgb.XGBClassifier, xgb.XGBRegressor]
CAT = Union[cat.CatBoostClassifier, cat.CatBoostRegressor]

Estimators = Union[BaseEstimator, LGBM, XGB, CAT]

def estimator_type(model) -> Any:
    if type(model) is lgb.LGBMClassifier or type(model) is lgb.LGBMRegressor:
        return LGBM

    if type(model) is xgb.XGBClassifier or type(model) is xgb.XGBRegressor:
        return XGB

    if type(model) is cat.CatBoostClassifier or type(model) is cat.CatBoostRegressor:
        return CAT

    return type(model)