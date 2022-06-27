from utils.preprocessing import columns_for_deletion, split_train_test
from utils.model import RegressionModel
from utils.storage import ClearMLStorage
import clearml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid
import pickle
import os
from itertools import product
from clearml import Dataset, Task
import time

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Optional, Tuple, Union, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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


def init_scaler_original(scaling, **scaler_kwargs):
    if scaling == "MinMax":
        if scaler_kwargs.__len__() == 0:
            scaler_kwargs = dict(feature_range=(0.5, 1.5))
        scaler = MinMaxScaler(**scaler_kwargs)
    elif scaling == "Standard":
        scaler = StandardScaler()
    else:
        pass
    return scaling, scaler


def init_scaler(scaling, valid_scaling=None, **scaler_kwargs):
    """Требует доработки, если будет использоваться, на данный момент не импортятся нормально объекты."""
    scaler = None
    if valid_scaling is None:
        valid_scaling = ['MinMaxScaler', 'StandardScaler']
    if scaling in valid_scaling:
        try:
            scaler = eval(scaling)(**scaler_kwargs)
        except TypeError:
            print('Incorrect arguments for scaler. Turning `scaling` to None')
            scaling = None
        except Exception as e:
            print(f'Unknown exception {type(e).__name__.__repr__()}: {e}')
            print("Turning `scaling` to None.")
            scaling = None
    return scaling, scaler


def columns_for_deletion(df: pd.DataFrame, startswith: Union[str, List[str]] = 'predict') -> List[str]:
    if isinstance(startswith, str):
        startswith = [startswith]
    columns_to_drop = [col for col in df.columns if any(
        list(map(lambda x: col.startswith(x), startswith)))]
    return columns_to_drop


def split_train_test(
    data_df: pd.DataFrame,
    business_unit: str,
    analog_group: str,
    window_size: int,
    split_date: datetime.date,
    n_predict: int,
    target_column_name_formatter: str,
    log_target: bool = False,
    scaling: Optional[str] = None,
    scaler_kwargs: Optional[Dict[str, Any]] = None,
    drop_not_scalable: bool = True
):
    """_summary_

    Args:
        data_df (pd.DataFrame): Preprocessed dataframe
        analog_group (Union[str, int]): Specific group of analogs.
        window_size (int): number of days used to create the dataset.
        split_date (datetime.date): day relative to which the train and test are divided.
        n_predict (int): What month is the prediction for.
        target_column_name_formatter (str): What value is predicted.
        log_target (bool, optional): whether to logarithm target or not. Defaults to False.
        scaling (Optional[str], optional): whether use specific scaling or not. Defaults to None.
        scaler_kwargs (Optional[Dict[str, Any]], optional): Scaler kwargs if needed. Defaults to None.
        drop_not_scalable (bool, optional): _description_. Defaults to True.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if isinstance(split_date, str):
        # split_date = datetime.datetime.strptime(split_date, "%d.%m.%Y").date()
        split_date = pd.to_datetime(split_date)

    valid_scaling = ['MinMaxScaler', 'StandardScaler']
    valid_scaling += ['MinMax', 'Standard']

    target_column_name = target_column_name_formatter.format(n_predict)
    # if target_column_name not in data_df.columns:
    #     raise ValueError(f'Missed columns: {target_column_name}')

    # check scaling
    if scaling and scaling not in valid_scaling:
        raise ValueError("scaling_type must be one of %r." % valid_scaling)

    tdf = data_df[(data_df['business_unit'] == business_unit) &
                  (data_df['material_cd'] == analog_group) &
                  (data_df['window_size'] == window_size)
                  ]

    # split to train and test
    date_to_split = split_date - relativedelta(days=30) * (n_predict + 1)
    date_split_query = tdf['cut_date'].dt.date < date_to_split

    # train_slice = tdf[date_split_query]
    # test_slice = tdf[~date_split_query]

    train_slice = tdf[tdf['cut_date'].dt.date < date_to_split]
    test_slice = tdf[tdf['cut_date'].dt.date >= split_date]

    # work around target
    y_train = train_slice[target_column_name]
    y_test = test_slice[target_column_name]

    if log_target:
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)

    if scaler_kwargs is None:
        scaler_kwargs = {}
    # scaling, scaler = self.init_scaler(scaling, **scaler_kwargs)
    scaling, scaler = init_scaler_original(scaling, **scaler_kwargs)

    # delete cut_date, mtr_cd, business_unit, w_size, predicts
    columns_to_drop = ['cut_date', 'material_cd',
                       'business_unit', 'window_size']
    columns_to_drop += columns_for_deletion(tdf, startswith='predict')

    # X_train = train_slice.drop(columns_to_drop, axis=1)
    # X_test = test_slice.drop(columns_to_drop, axis=1)
    X_train = train_slice[:]
    X_test = test_slice[:]

    # # delete stock_bmu_count in sum_df dataset
    # if mtr_group == "all":
    #     X_train = X_train.drop(['stock_bmu_count'], axis=1)
    #     X_test = X_test.drop(['stock_bmu_count'], axis=1)
    print(f"Train Shape : {train_slice.shape}, Test Shape {test_slice.shape}")
    # applying scaling
    if scaling:
        columns_list = X_train.columns
        columns_list_ = list(set(X_train.columns) - set(columns_to_drop))

        X_train_ = scaler.fit_transform(X_train[columns_list_])
        X_test_ = scaler.transform(X_test[columns_list_])

        X_train.loc[:, columns_list_] = X_train_
        X_test.loc[:, columns_list_] = X_test_

        # X_train = pd.DataFrame(
        #     X_train, columns=columns_list, index=train_slice.index)
        # X_test = pd.DataFrame(X_test, columns=columns_list,
        #                     index=test_slice.index)

        if not drop_not_scalable:
            for col in columns_to_drop:
                X_train[col] = train_slice[col]
                X_test[col] = test_slice[col]

    return X_train, X_test, y_train, y_test


class ClearMLStorage():
    def __init__(self, project_name, task_name,
                 dataset_id, dataset_file_name,
                 **kwargs) -> None:
        self._set_task_params(project_name, task_name,
                              kwargs.get('task_type', 'training'))
        self._set_dataset(dataset_id, dataset_file_name)
        self._load_params(**kwargs)

    def _set_task_params(self, project_name, task_name, task_type='training', **kwargs):
        self._project_name = project_name
        self._task_name = task_name
        self._task_type = task_type
        self.task_init_params = {'project_name': self._project_name,
                                 "task_name": self._task_name,
                                 "task_type": self._task_type,
                                 **kwargs
                                 }

    def _set_dataset(self, dataset_id, dataset_file_name):
        self._dataset_id = dataset_id
        self._dataset_file_name = dataset_file_name
        self.dataset = {"dataset_id": self._dataset_id,
                        "dataset_file_name": self._dataset_file_name}

    def _load_params(self, **params):
        self.dataset_params = {
            "business_unit": "3990",
            "analog_group": '150-40',
            "window_size": 60,
            "split_date": "01.01.2021",
            # "n_predict" : 1,
            "target_column_name_formatter": "predict_{}",
            "log_target": False,
            "scaling": "Standard",
            "scaler_kwargs": None,
            # "drop_not_scalable": True,
        }
        self.dataset_params.update(
            params.get('dataset_params', {}))

        self.model_type_params = {
            "model_type": "Ridge",
            "use_kfold": True,
            "save_kfold_predicts": True,
            "save_model": False,
        }
        self.model_type_params.update(params.get('model_type_params', {}))

        self.model_kwargs_params = {
            "alpha": 1.0
        }
        self.model_kwargs_params.update(params.get('model_kwargs_params', {}))

        self.kflod_kwargs_params = {
            "n_splits": 5,
            "shuffle": False
        }
        self.kflod_kwargs_params.update(params.get('kflod_kwargs_params',  {}))


def make_task(storage: ClearMLStorage):
    import warnings
    warnings.filterwarnings('ignore')
    clearml.Task.add_requirements("pyarrow")
    task = Task.init(**storage.task_init_params)
    task.connect(storage.dataset, name="dataset")

    # params
    task.connect(storage.dataset_params,
                 'dataset_params')
    task.connect(storage.model_type_params, 'model_type_params')
    task.connect(storage.model_kwargs_params, 'model_kwargs_params')
    task.connect(storage.kflod_kwargs_params, 'kflod_kwargs_params')
    return task


def load_dataset(storage: ClearMLStorage):
    """Выглядит трудозатратным, весь датасет сейчас занимает Гб ОЗУ. 
    Мб разделить на отдельные датасеты и мерджить при необходимости?
    """

    dataset_id = storage.dataset["dataset_id"]
    dataset_file_name = storage.dataset["dataset_file_name"]

    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_file_path = dataset.get_local_copy()

    dataset_df = pd.read_parquet(os.path.join(
        dataset_file_path, dataset_file_name))
    return dataset_df

# def clearml_task_iteration(task: Task, n_predict_max=12):


def clearml_task_iteration(storage: ClearMLStorage, n_predict_max=12):
    task = make_task(storage)
    dataset_df = load_dataset(storage)

    logger = task.get_logger()
    full_dict_to_save = {}

    for n_predict in range(n_predict_max):
        print(f"training : {n_predict}")
        X_train, X_test, y_train, y_test = split_train_test(
            dataset_df,
            **storage.dataset_params,
            n_predict=n_predict,
            drop_not_scalable=False
        )

        # columns_to_drop = ['cut_date', 'material_cd',
        #                     'business_unit', 'window_size']
        # columns_to_drop += columns_for_deletion(dataset_df, startswith='predict')

        if storage.model_type_params["use_kfold"]:
            model, dict_to_save = train_with_cv(
                storage, X_train, X_test, y_train, y_test, logger)
        else:
            model, dict_to_save = train_wo_cv(
                storage, X_train, X_test, y_train, y_test)
        full_dict_to_save[n_predict] = dict_to_save

    pickle.dump(
        full_dict_to_save,
        open("model_results.pkl", "wb")
    )
    task.upload_artifact("model_results", artifact_object='model_results.pkl')


def train_with_cv(storage: ClearMLStorage, X_train, X_test, y_train, y_test, logger):
    cv_kfold = KFold(**storage.kflod_kwargs_params)

    kfold_model_num = 0
    kfold_model_dict = {}

    columns_train_on_dict = {}

    if storage.model_type_params["save_kfold_predicts"]:
        X_kfold_with_predict = X_train.copy()

    X_test_with_predict = X_test.copy()

    dates_range = pd.Series(pd.date_range(
        *X_train['cut_date'].agg([min, max]).values.tolist()))

    for train_index, valid_index in cv_kfold.split(dates_range):
        print("training kfold")
        kfold_model_num += 1

        train_slice = X_train["cut_date"].isin(
            dates_range[train_index].values)
        valid_slice = X_train["cut_date"].isin(
            dates_range[valid_index].values)

        X_kfold_train, y_kfold_train = X_train[train_slice], y_train[train_slice]
        X_kfold_valid, y_kfold_valid = X_train[valid_slice], y_train[valid_slice]

        if y_train.nunique() > 1:
            model = RegressionModel(
                model_type=storage.model_type_params["model_type"],
                model_kwargs=storage.model_kwargs_params
            )

            columns_to_drop = ['cut_date', 'material_cd',
                               'business_unit', 'window_size']
            columns_to_drop += columns_for_deletion(
                X_train, startswith='predict')
            columns_train_on = list(
                set(X_train.columns) - set(columns_to_drop))

            model.fit(
                X_kfold_train[columns_train_on],
                y_kfold_train,
                eval_set=(X_kfold_valid[columns_train_on], y_kfold_valid)
            )

            predict = model.predict(X_test[columns_train_on])

            X_test_with_predict[f"model_predict_{kfold_model_num}"] = predict

            columns_train_on_dict[kfold_model_num] = columns_train_on

            valid_predict = model.predict(X_kfold_valid[columns_train_on])

            if storage.model_type_params["save_kfold_predicts"]:
                X_kfold_with_predict.loc[valid_slice,
                                         "model_predict"] = valid_predict

            if storage.model_type_params["save_model"]:
                kfold_model_dict[kfold_model_num] = kfold_model_dict

            logger.report_scalar(
                "kfold error",
                "rmse",
                iteration=kfold_model_num,
                value=mean_squared_error(y_kfold_valid, valid_predict)
            )
            logger.report_scalar(
                "kfold error",
                "mae",
                iteration=kfold_model_num,
                value=mean_absolute_error(y_kfold_valid, valid_predict)
            )

    dict_to_save = {
        "X_test_with_predict": X_test_with_predict,
        "columns_train_on_dict": columns_train_on_dict,
    }

    if storage.model_type_params["save_model"]:
        dict_to_save["kfold_model_dict"] = kfold_model_dict

    if storage.model_type_params["save_kfold_predicts"]:
        dict_to_save["X_kfold_with_predict"] = X_kfold_with_predict

    return model, dict_to_save


def train_wo_cv(storage: ClearMLStorage, X_train, X_test, y_train, y_test):
    model = RegressionModel(
        model_type=storage.model_type_params["model_type"],
        model_kwargs=storage.model_kwargs_params
    )

    columns_to_drop = ['cut_date', 'material_cd',
                       'business_unit', 'window_size']
    columns_to_drop += columns_for_deletion(X_train, startswith='predict')
    columns_train_on = list(set(X_train.columns) - set(columns_to_drop))

    model.fit(X_train[columns_train_on], y_train)

    predict = model.predict(X_test[columns_train_on])
    X_test_with_predict = X_test.copy()
    X_test_with_predict["model_predict"] = predict

    dict_to_save = {
        "columns_train_on": columns_train_on,
        "X_test_with_predict": X_test_with_predict
    }

    if storage.model_type_params["save_model"]:
        dict_to_save["model"] = model
    return model, dict_to_save


def clone_template(template_task_id, dataset_hyper_params_dict, model_type_hyper_params_dict, kflod_kwargs_hyper_params_dict,  queue_name='cpu'):
    template_task = Task.get_task(task_id=template_task_id)

    s = 0
    param_grid = product(ParameterGrid(dataset_hyper_params_dict),
                         ParameterGrid(model_type_hyper_params_dict),
                         ParameterGrid(kflod_kwargs_hyper_params_dict))
    for dataset_param_grid, model_type_param_grid, kfold_param_grid in param_grid:
        pair = (dataset_param_grid['business_unit'],
                dataset_param_grid['analog_group'])
        window_size = dataset_param_grid['window_size']
        model_name = model_type_param_grid['model_type']
        print("pair: {}, window_size: {}, model_name: {}".format(
            *map(repr, pair, window_size, model_name)))

        cloned_task = Task.clone(source_task=template_task)
        cloned_task.add_tags(
            ["grid_search", model_name, pair[0], f"bu={pair[0]}", f"group={pair[1]}"])

        cloned_task.set_parameter(f"model_type/model_type", value=model_name)
        for key, value in dataset_param_grid.items():
            cloned_task.set_parameter(f"dataset_params/{key}", value=value)

        for key, value in model_type_param_grid.items():
            cloned_task.set_parameter(f"model_type_params/{key}", value=value)

        for key, value in kfold_param_grid.items():
            cloned_task.set_parameter(
                f"kflod_kwargs_params/{key}", value=value)

        Task.enqueue(task=cloned_task, queue_name=queue_name)


storage = ClearMLStorage(project_name='test/test0',
                         task_name='template_full',
                         dataset_id='1f08b7ac2f43421a81ebb957cac81997',
                         dataset_file_name='test_data.parquet')

task = make_task(storage)
clearml_task_iteration(storage)
time.sleep(5)
task.close()