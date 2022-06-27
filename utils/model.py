
import pandas as pd
import numpy as np

from typing import List, Dict, Optional, Tuple, Union, Any

# from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import inspect


def get_all_args_without_self(func):
    return set(inspect.signature(func).parameters)-set(["self"])


class RegressionModel():
    def __init__(
        self,
        model_type: str,
        model_kwargs: Optional[Dict] = {}
    ) -> None:
        # self._init_possible_models()
        self._init_possible_models_original()

        self.model_type = model_type
        self.model_kwargs = model_kwargs

        self.model = self._init_model()
        self.is_fitted = False

    def _init_possible_models_original(self):
        self._sklearn_models = {
            "Ridge": Ridge,
            "RandomForestRegressor": RandomForestRegressor,
            "MLPRegressor": MLPRegressor,
            "Lasso": Lasso,
            "ElasticNet": ElasticNet
        }
        self._other_models = {"CatBoostRegressor": CatBoostRegressor}

        self._models_kwargs_available = {model_name: get_all_args_without_self(
            model_class.__init__) for model_name, model_class in self._sklearn_models.items()}
        self._models_kwargs_available["CatBoostRegressor"] = get_all_args_without_self(
            CatBoostRegressor.__init__)

    def _init_possible_models(self):
        """Требует доработки, если будет использоваться, на данный момент не импортятся нормально объекты."""
        self._sklearn_models = [
            "Ridge", "RandomForestRegressor", "MLPRegressor", "Lasso", "ElasticNet"]
        self._other_models = ["CatBoostRegressor"]

        all_variables = set(
            filter(lambda x: not x.startswith('_'), vars().keys()))
        print(vars())
        self._sklearn_models = dict(map(lambda x: (x, eval(x)), filter(
            lambda x: x in all_variables, self._sklearn_models)))
        self._other_models = dict(map(lambda x: (x, eval(x)), filter(
            lambda x: x in all_variables, self._other_models)))

        self._models_kwargs_available = {model_name: get_all_args_without_self(
            model_class.__init__) for model_name, model_class in self._sklearn_models.items()}
        self._models_kwargs_available.update({model_name: get_all_args_without_self(
            model_class.__init__) for model_name, model_class in self._other_models.items()})

    def _init_model(self):
        all_model_types = {**self._sklearn_models, **self._other_models}
        if self.model_type not in all_model_types.keys():
            raise ValueError('Unknown model_type, must be one of %r.' %
                             list(all_model_types.keys()))

        model_kwargs_filtered = filter(
            lambda x: x in self._models_kwargs_available[self.model_type],
            self.model_kwargs.keys()
        )

        model_kwargs_filtered = {
            key: self.model_kwargs[key] for key in model_kwargs_filtered}

        model = all_model_types[self.model_type](**model_kwargs_filtered)
        return model

    def fit(
        self,
        X_train,
        y_train,
        **kwargs
    ):
        # fit_kwargs = ('eval_set', 'cat_features')
        # fit_kwargs = {key: value for key in fit_kwargs if (value:=kwargs.get(key)) is not None}

        allowed_parameters = get_all_args_without_self(self.model.fit)
        fit_kwargs = {key: value for key,
                      value in kwargs.items() if key in allowed_parameters}

        self.model.fit(
            X_train, y_train,
            **fit_kwargs
        )

        self.is_fitted = True

        return self

    def predict(
        self,
        X
    ):
        if self.is_fitted:
            return self.model.predict(X)
        else:
            raise Exception("Not Trained")
