# ======
# implemantation inspired by nyaggle
# (https://github.com/nyanp/nyaggle/blob/master/nyaggle/validation/cross_validate.py)
# ======

import warnings
from copy import deepcopy
from typing import Dict, List, Tuple, Iterable, Optional, Union, Callable
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import multiclass
from sklearn.exceptions import NotFittedError

from datasets import Dataset

from ..estimator.type import estimator_type, LGBM, CAT, XGB, Estimators
from ..dataset import TableDataset

@dataclass
class EvalResult:
    oof:np.ndarray
    scores:Optional[np.ndarray]=None
    test_probas:Optional[np.ndarray]=None
    model_name:Optional[str]=None

    # TODO
    # model自体を保存するべきかどうか(その場合はfitのparameter等の保存も)

    def __repr__(self) -> str:
        if self.scores is not None:
            score_mean = np.mean(self.scores)
            score_std = np.std(self.scores)
            scores = f'Avg. scores: {score_mean: .4f}/{score_std:.4f} {self.scores}'
        else:
            scores = 'scores: Not Available Metric Function'

        return f"model: {self.model_name}\n"\
            f"oof: {self.oof.shape}\n"\
            f"{scores}\n"\
            f"test_probas: {None if self.test_probas is None else self.test_probas.shape}"

@dataclass
class ModelSet:
    model: Estimators
    fit_params:Optional[Dict]=None
    name: Optional[str]=None

    def __post_init__(self):
        if self.name is None:
            self.name = self.model.__class__.__name__


class BaseRunner(BaseEstimator):
    def __init__(
        self,
        table_dataset:TableDataset,
        metric_func=None,
        ):

        self.table_dataset = table_dataset
        self.metric_func = metric_func

    def fit(self, model_set:ModelSet) -> List[Estimators]:
        trained_models = []

        for (trn_set, val_set) in self.table_dataset.iter_fold():
            _model = fitter(model_set.model, trn_set=trn_set, val_set=val_set, params=model_set.fit_params)
            trained_models.append(_model)

        return trained_models

    def evaluate(self, trained_models: List[Estimators]) -> EvalResult:
        model_name = trained_models[0].__class__.__name__

        class_size = self.table_dataset.class_size
        type_of_target = self.table_dataset.type_of_target
        train = self.table_dataset.train
        test = self.table_dataset.test
        cv = self.table_dataset.cv

        assert len(trained_models) == len(cv)

        if self.table_dataset.class_size > 1:
            oof = np.zeros((len(train), class_size), dtype=np.float32)
        else:
            oof = np.zeros(len(train), dtype=np.float32)

        test_probas = None
        if test is not None:
            if class_size > 1:
                test_probas = np.zeros((len(test), class_size), dtype=np.float32)
            else:
                test_probas = np.zeros(len(test), dtype=np.float32)

        scores = None
        if self.metric_func is not None:
            scores = np.zeros(len(cv), dtype=np.float32)

        for i, (_model, (_, val_idx)) in enumerate(zip(trained_models, cv)):

            proba = predictor(train['data'][val_idx], _model, type_of_target)
            if test_probas is not None:
                test_proba = predictor(test['data'], _model, type_of_target)
                test_probas += test_proba / len(cv)

            oof[val_idx] = proba
            if self.metric_func is not None:
                # calculate each fold score
                score = self.metric_func(train['label'][val_idx], oof[val_idx])
                scores[i] = score
                print(f'[Fold {i} Score]: {score:.4f}')

        if self.metric_func is not None:
            print(f"[Overall Score]: {self.metric_func(train['label'], oof):.4f}")

        return EvalResult(
            oof = oof,
            scores=scores,
            test_probas=test_probas,
            model_name=model_name
        )

    def fit_eval(self, model_set:ModelSet) -> Tuple[EvalResult, List[Estimators]]:
        trained_models = self.fit(model_set)
        eval_result = self.evaluate(trained_models)

        return eval_result, trained_models


@dataclass
class RunnerSet:
    model_set:ModelSet
    runner:BaseRunner

class StackedRunner:
    def __init__(self, runner:BaseRunner):
        self.runner = runner
        self.train_models:List[List[Estimators]] = []
        self.eval_result:List[EvalResult] = []

    def train_eval(self, models:Iterable[ModelSet]):
        for model_set in models:
            print(f'===[Train]===: {model_set.name}')
            trained_models = self.runner.fit(model_set)
            self.train_models.append(trained_models)

            eval_result = self.runner.evaluate(trained_models)
            self.eval_result.append(eval_result)

    def emsamble(self, methods:Union[str, Callable]):
        if len(self.eval_result) <= 1:
            raise ValueError("Required at least two models to ensemble")
        


def trainer(
    features:np.ndarray,
    targets:np.ndarray,
    model:Estimators,
    cv:Iterable[Tuple[np.ndarray, np.ndarray]],
    fit_params:Optional[Dict]=None,
    ) -> Iterable[Estimators]:

    models = []

    for fold, (trn_idx, val_idx) in enumerate(cv):

        trn_set = (features[trn_idx], targets[trn_idx])
        val_set = (features[val_idx], targets[val_idx])

        _model = fitter(model, trn_set=trn_set, val_set=val_set, params=fit_params)

        models.append(_model)

    return models

def fitter(
    model:Estimators,
    trn_set:Iterable[Tuple[np.ndarray, np.ndarray]],
    val_set:Optional[Iterable[Tuple[np.ndarray, np.ndarray]]]=None,
    params:Optional[Dict]=None,
    ) -> Estimators:

    if params is None:
        params = {}

    trn_xs, trn_ys = trn_set[0], trn_set[1]

    if estimator_type(model) is LGBM:
        _model: LGBM = deepcopy(model)
        _model.fit(
            X=trn_xs,
            y=trn_ys,
            eval_set=[val_set],
            **params,
        )
        return _model
    elif estimator_type(model) is XGB:
        _model: XGB = deepcopy(model)
        _model.fit(
            X=trn_xs,
            y=trn_ys,
            eval_set=[val_set],
            **params,
        )
        return _model
    elif estimator_type(model) is CAT:
        _model: CAT = deepcopy(model)
        _model.fit(
            X=trn_xs,
            y=trn_ys,
            eval_set=[val_set],
            **params,
        )
        return _model
    else:
        # other type
        _model = deepcopy(model)
        _model.fit(
            X=trn_xs,
            y=trn_ys,
            **params,
            )
        return _model

# 使ってない
def evaluator(
    features:np.ndarray,
    targets:np.ndarray,
    models:Iterable[Estimators],
    cv: List[Tuple[np.ndarray, np.ndarray]],
    test_features:Optional[np.ndarray]=None,
    metric_func=None,
    type_of_target: str = 'auto',
    ) -> EvalResult:

    cls_size = 1
    if type_of_target == 'auto':
        type_of_target = multiclass.type_of_target(targets)
    if type_of_target == 'multiclass':
        cls_size = np.unique(targets, return_counts=True)[0].size

    if not isinstance(models, list):
        models = list(models)

    assert len(models) == len(cv)

    oof = np.zeros((len(features), cls_size), dtype=np.float32) if cls_size > 1 else np.zeros(len(features), dtype=np.float32)

    test_probas = None
    if test_features is not None:
        test_probas = np.zeros((len(test_features), cls_size), dtype=np.float32) if cls_size > 1 else np.zeros(len(test_features), dtype=np.float32)

    scores = None
    if metric_func is not None:
        scores = np.zeros(len(cv), dtype=np.float32)

    for i, (_model, (trn_idx, val_idx)) in enumerate(zip(models, cv)):

        proba = predictor(features[val_idx], _model, type_of_target)

        if test_probas is not None:
            test_proba = predictor(test_features, _model, type_of_target)
            test_probas += test_proba / len(cv)

        oof[val_idx] = proba

        if metric_func is not None:
            # calculate each fold score
            score = metric_func(targets[val_idx], oof[val_idx])
            scores[i] = score
            print(f'[Fold {i} Score]: {score:.4f}')

    if metric_func is not None:
        print(f'[Overall Score]: {metric_func(targets, oof):.4f}')

    return EvalResult(
        oof = oof,
        scores=scores,
        test_probas=test_probas
    )


def predictor(
    features:np.ndarray,
    model:Estimators,
    type_of_target:str,
    ):

    if type_of_target in ('binary', 'multiclass'):
        # classification
        if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)

        elif hasattr(model, "decision_function"):
                warnings.warn('Since {} does not have predict_proba method, '
                              'decision_function is used for the prediction instead.'.format(type(model)))
                proba = model.decision_function(features)
        else:
            raise RuntimeError('Estimator in classification problem should have '
                                   'either predict_proba or decision_function')

        if proba.ndim != 1:
            return proba[:, 1] if proba.shape[1] == 2 else proba
        else:
            return proba
    else:
        # regression
        return model.predict(features)