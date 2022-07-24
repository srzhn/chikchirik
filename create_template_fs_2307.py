# class ClearMLStorage():
#     def __init__(self, project_name, task_name,
#                  dataset_id, dataset_file_name=None,
                 
#                 use_fs=True,
#                  method_name='pearson',
#                 n_features = 25,
#                 use_rfe = False,
#                 rfe_n_features=None,
#                  **kwargs) -> None:
#         self._set_task_params(project_name, task_name,
#                               kwargs.get('task_type', 'training'))
#         self._set_dataset(dataset_id, dataset_file_name)
#         self._set_fs(use_fs=use_fs, method_name=method_name, n_features=n_features, use_rfe=use_rfe, rfe_n_features=rfe_n_features, **kwargs)

#         self._load_params(**kwargs)
        
#         # 2806 change
#         self._change_dataset_by_window_size(dataset_file_name)
        
        
#     # FIXME: Костыль. Храним для каждого размера окна свой паркет, на данном этапе [60, 120, 360].
#     def _change_dataset_by_window_size(self, dataset_file_name=None):
#         if dataset_file_name is not None:
#             return
#         if self.dataset_params['window_size']==60:
#             self.dataset['dataset_file_name'] = "dataset_ws60.parquet"
#         elif self.dataset_params['window_size']==120:
#             self.dataset['dataset_file_name'] = "dataset_ws120.parquet"
#         elif self.dataset_params['window_size']==360:
#             self.dataset['dataset_file_name'] = "dataset_ws360.parquet"
#         return 

#     def _set_task_params(self, project_name, task_name, task_type='training', **kwargs):
#         self._project_name = project_name
#         self._task_name = task_name
#         self._task_type = task_type
#         self.task_init_params = {'project_name': self._project_name,
#                                  "task_name": self._task_name,
#                                  "task_type": self._task_type,
#                                  **kwargs
#                                  }

#     def _set_dataset(self, dataset_id, dataset_file_name):
#         self._dataset_id = dataset_id
#         self._dataset_file_name = dataset_file_name
#         self.dataset = {"dataset_id": self._dataset_id,
#                         "dataset_file_name": self._dataset_file_name}

#     def _set_fs(self, use_fs, method_name, n_features=100, use_rfe=False, rfe_n_features=None, **params):
#         self._use_fs = use_fs

#         self._method_name = method_name
#         self._n_features = n_features

#         self._use_rfe = use_rfe
#         self._rfe_n_features = rfe_n_features

#         self.fs_kwargs_params = {'use_fs': use_fs,
#                                 'method': self._method_name,
#                                 'n_features': self._n_features,
#                                 'rfe': self._use_rfe,
#                                 'rfe_n_features': self._rfe_n_features
#         }

#     def _load_params(self, **params):
#         self.dataset_params = {
#             "business_unit": "all",
#             # "analog_group": '150',
#             # "analog_group": '150-40', # 0707
#             "window_size": 60,
#             "split_date": "01.01.2021",
#             # "n_predict" : 1,
#             "target_column_name_formatter": "predict_{}",
#             "log_target": False,
#             "scaling": "Standard",
#             "scaler_kwargs": None,
#             # "drop_not_scalable": True,
#         }
#         self.dataset_params.update(
#             params.get('dataset_params', {}))

#         self.model_type_params = {
#             # "model_type": "Ridge",
#             "model_type": "CatBoostRegressor",
#             "use_kfold": False,
#             "save_kfold_predicts": True,
#             "save_model": False,
#         }
#         self.model_type_params.update(params.get('model_type_params', {}))

#         self.model_kwargs_params = {
#             "alpha": 1.0,
            
#             "max_depth": 5,
#             "n_estimators": 500,

#             # "depth": 5,
#             # "iterations": 200,
#             "l2_leaf_reg": 0.01
#         }
#         self.model_kwargs_params.update(params.get('model_kwargs_params', {}))

#         self.kflod_kwargs_params = {
#             "n_splits": 5,
#             "shuffle": False
#         }
#         self.kflod_kwargs_params.update(params.get('kflod_kwargs_params',  {}))

#     def print_params(self):
#         print("Dataset:")
#         print(self.dataset)

#         print("Dataset Preprocessing Params:")
#         print(self.dataset_params)

#         print("Model type:")
#         print(self.model_type_params)

#         print('Model Parameters:')
#         print(self.model_kwargs_params)

#         print('K-Fold Parameters:')
#         print(self.kflod_kwargs_params)




############


import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Optional, Tuple, Union, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def init_scaler_original(scaling, **scaler_kwargs):
    if scaling == "MinMax":
        if scaler_kwargs.__len__() == 0:
            scaler_kwargs = dict(feature_range=(0.5, 1.5))
        scaler = MinMaxScaler(**scaler_kwargs)
    elif scaling == "Standard":
        scaler = StandardScaler()
    else:
        scaling = None
        scaler = None
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
    # business_unit: str,
    # analog_group: str,
    # window_size: int,
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

    valid_scaling = ['None', 'MinMaxScaler', 'StandardScaler']
    valid_scaling += ['MinMax', 'Standard']

    target_column_name = target_column_name_formatter.format(n_predict)
    # if target_column_name not in data_df.columns:
    #     raise ValueError(f'Missed columns: {target_column_name}')

    # check scaling
    if scaling and scaling not in valid_scaling:
        raise ValueError("scaling_type must be one of %r." % valid_scaling)

    # tdf = data_df[(data_df['business_unit'] == business_unit) &
    #               (data_df['material_cd'] == analog_group) &
    #               (data_df['window_size'] == window_size)
    #               ]

    # print(f"Unique window_sizes: {data_df['window_size'].unique()}")

    tdf = data_df.copy()
    # tdf = data_df[(data_df['business_unit'] == business_unit)]
    # print(tdf.shape[0])
    # tdf = tdf[tdf['material_cd'] == analog_group]
    # print(tdf.shape[0])
    # tdf = tdf[tdf['window_size'] == window_size]
    # print(tdf.shape[0])

    tdf = tdf.dropna()

    # split to train and test
    train_date_to_split = split_date - relativedelta(days=30) * (n_predict + 1)

    # date_split_query = tdf['cut_date'].dt.date < train_date_to_split
    # train_slice = tdf[date_split_query]
    # test_slice = tdf[~date_split_query]

    train_slice = tdf[tdf['cut_date'].dt.date < train_date_to_split]
    test_slice = tdf[tdf['cut_date'].dt.date >= split_date]

    # work around target
    y_train = train_slice[target_column_name]
    y_test = test_slice[target_column_name]

    if str(log_target)=='True':
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)

    if scaler_kwargs is None or len(scaler_kwargs)==0:
        scaler_kwargs = {}
    # scaling, scaler = self.init_scaler(scaling, **scaler_kwargs)
    scaling, scaler = init_scaler_original(scaling, **scaler_kwargs)

    # delete cut_date, mtr_cd, business_unit, w_size, predicts
    columns_to_drop = ['cut_date', 'analog_group']
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

    if X_train.shape[0]==0:
        print('[INFO] Empty X_train.')
        return X_train, X_test, y_train, y_test

    # applying scaling
    if scaling:        
        columns_list_available = list(set(X_train.columns) - set(columns_to_drop))

        X_train_ = scaler.fit_transform(X_train[columns_list_available])
        X_test_ = scaler.transform(X_test[columns_list_available])

        X_train.loc[:, columns_list_available] = X_train_
        X_test.loc[:, columns_list_available] = X_test_

        # X_train = pd.DataFrame(
        #     X_train, columns=columns_list, index=train_slice.index)
        # X_test = pd.DataFrame(X_test, columns=columns_list,
        #                     index=test_slice.index)

        if not drop_not_scalable:
            for col in columns_to_drop:
                X_train[col] = train_slice[col]
                X_test[col] = test_slice[col]

    return X_train, X_test, y_train, y_test


############################

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



########################
import datetime
from itertools import product
import os
import pickle

import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# import clearml
from clearml import Dataset, Task

# from utils.storage import ClearMLStorage
# from utils.model import RegressionModel
# from utils.preprocessing import columns_for_deletion, split_train_test

# from .storage import ClearMLStorage
# from .model import RegressionModel
# from .preprocessing import columns_for_deletion, split_train_test

import warnings
warnings.filterwarnings('ignore')



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostRegressor
# from boruta import BorutaPy
# from BorutaShap import BorutaShap
from sklearn.feature_selection import RFE

class FeatureSelection:
    def __init__(self, method='pearson', rfe=False, **kwargs) -> None:
        self.rfe = rfe
        self.method = method

    # Функция для отбора признаков на основе корреляции Пирсона:
    def _pearson_feature_selection(self, X: pd.DataFrame, y: pd.Series, n_features=100) -> pd.DataFrame:
        
        # Инициализация селектора признаков:
        selector = SelectKBest(score_func=f_regression, k=n_features)
        
        # Применение селектора, оценка результата:
        # X_selected = selector.fit_transform(X, y)
        selector.fit(X, y)
        
        # Собираем обратно датасет в уже отфильтрованном виде:
        # filtered_df = pd.DataFrame(columns=selector.get_feature_names_out(), index = X.index, data=X_selected)
        filtered_df = X.iloc[:, selector.get_support()]
        
        return filtered_df

    # Функция для фильтрации с помощью RF и Lasso с выбором только совпадающих признаков:
    def _rf_lasso_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        
        # Инициализируем и обучаем случайный лес:
        rf_selector = RandomForestRegressor(n_estimators=500, random_state=1)
        rf_selector.fit(X, y)
        
        # Вычисляем значения важности признаков:
        feature_importances = rf_selector.feature_importances_
        
        # Собираем вспомогательный датафрейм с исходными колонками и значениями важности:
        temp_df_rf = pd.DataFrame(columns=X.columns, data=[feature_importances])
        
        # Обучаем линейную регрессию с L1-регуляризацией:
        l1_selector = Lasso(alpha=0.2, random_state=42, max_iter=10000).fit(X, y)
        
        # Собираем вспомогательный датафрейм с исходными колонками и значениями важности:
        temp_df_l1 = pd.DataFrame(columns=X.columns, data=[l1_selector.coef_])
        
        # Найдём признаки, которые оставили и RF, и Lasso:
        temp_df = pd.concat([temp_df_rf, temp_df_l1])
        
        # Оставляем из исходного датафрейма только важные колонки (признаки):
        filtered_df = X.loc[:, (temp_df != 0).all(axis=0)]
        
        return filtered_df

    # Функция для фильтрации с помощью Feature Importance из Catboost:
    def _catboost_feature_selection(self, X: pd.DataFrame, y: pd.Series, n_features=100) -> pd.DataFrame:
        
        # Инициализируем и обучаем регрессор:
        selector = CatBoostRegressor()
        selector.fit(X = X, y = y, verbose=False)
        
        # Собираем вспомогательный датафрейм с исходными колонками и значениями важности:
        feature_importances = pd.Series(selector.get_feature_importance(), X.columns)
        
        # Отбираем 100 признаков с самым большим значением важности:
        selected_features = feature_importances.sort_values(ascending=False)[:n_features]
        
        # Оставляем из исходного датафрейма только важные колонки (признаки):
        filtered_df = X.loc[:, selected_features.index]
        
        return filtered_df

    # Функция для фильтрации с помощью Boruta:
    def _boruta_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        
        # Инициализация случайного леса в качестве эстиматора:
        rf = RandomForestRegressor(n_jobs = -1, max_depth = 5)
        
        # Инициализация Boruta для отбора признаков:
        selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=1)

        # Поиск всех релевантных признаков:
        selector.fit(X.values, y.values)
        
        # Фильтруем датасет, чтобы получить итоговый вариант:
        filtered_df = X.loc[:,selector.support_]
        
        return filtered_df

    # Функция для фильтрации с помощью Boruta:
    def _borutashap_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        
        # # Инициализация Boruta для отбора признаков:
        selector = BorutaShap(importance_measure='shap', classification=False)

        # Поиск всех релевантных признаков:
        selector.fit(X=X, y=y, n_trials=100, sample=False, verbose=False, random_state=0)
        
        # Получаем подвыборку в качестве итогового варианта:
        filtered_df = selector.Subset()
        
        return filtered_df

    def _rfe_feature_selection(self, X: pd.DataFrame, y: pd.Series, n_features=100) -> pd.DataFrame:
        # Инициализация и обучение RFE в качестве селектора признаков:
        selector = RFE(RandomForestRegressor(n_estimators=500, random_state=1),
                    n_features_to_select=n_features,
                    verbose=0)
        features = selector.fit(X, y)
        
        # Фильтруем датасет, чтобы получить итоговый вариант:
        filtered_df = X.loc[:, features.support_]
        
        return filtered_df

    def filter_X(self, X, y, n_features=100, rfe_n_features=None):
        # FIXME
        if self.rfe and rfe_n_features is None:
            rfe_n_features = n_features
            n_features *= 2
        # print(self.method)
        if self.method == 'pearson':
            new_X = self._pearson_feature_selection(X, y, n_features=n_features)
        elif self.method == 'catboost':
            new_X = self._catboost_feature_selection(X, y, n_features=n_features)
        else:
            raise ValueError("Unknown method")

        if not self.rfe:
            return new_X
        
        new_X = self._rfe_feature_selection(new_X, y, int(rfe_n_features))
        # new_X = self._rfe_feature_selection(new_X, y, rfe_n_features)
        return new_X


# def make_task(storage: ClearMLStorage):
    
#     Task.add_requirements("pyarrow")
#     task = Task.init(**storage.task_init_params)
#     task.connect(storage.dataset, name="dataset")

#     # params
#     task.connect(storage.dataset_params,
#                  'dataset_params')
#     task.connect(storage.model_type_params, 'model_type_params')
#     task.connect(storage.model_kwargs_params, 'model_kwargs_params')
#     task.connect(storage.kflod_kwargs_params, 'kflod_kwargs_params')
#     task.connect(storage.fs_kwargs_params, 'fs_kwargs_params')
#     return task


# def load_dataset(storage: ClearMLStorage):
#     """Выглядит трудозатратным, весь датасет сейчас занимает Гб ОЗУ. 
#     Мб разделить на отдельные датасеты и мерджить при необходимости?
#     """

#     dataset_id = storage.dataset["dataset_id"]
#     dataset_file_name = storage.dataset["dataset_file_name"]

#     dataset = Dataset.get(dataset_id=dataset_id)
#     dataset_file_path = dataset.get_local_copy()

#     dataset_df = pd.read_parquet(os.path.join(
#         dataset_file_path, dataset_file_name))
#     return dataset_df

# def load_dataset(task):
#     """Выглядит трудозатратным, весь датасет сейчас занимает Гб ОЗУ. 
#     Мб разделить на отдельные датасеты и мерджить при необходимости?
#     """

#     dataset_id = task.get_parameters()['dataset/dataset_id']
#     dataset_file_name = task.get_parameters()['dataset/dataset_file_name']

#     dataset = Dataset.get(dataset_id=dataset_id)
#     dataset_file_path = dataset.get_local_copy()

#     dataset_df = pd.read_parquet(os.path.join(
#         dataset_file_path, dataset_file_name))
#     return dataset_df

# def clearml_task_iteration(task: Task, n_predict_max=12):


def clearml_task_iteration(task, dataset_df: pd.DataFrame,  
            fs_kwargs_params,
            preprocessing_kwargs_params,
            model_type_params,
            model_kwargs_params,
            kflod_kwargs_params,
            n_predict_max=12,
        ):    

    logger = task.get_logger()
    full_dict_to_save = []
    models_dict_to_save = []

    # task_params = task.get_parameters_as_dict()
    # print(task.get_parameters())

    # task_params = task.get_parameters_as_dict()
    # print(task_params)

    for n_predict in range(n_predict_max):
        X_train, X_test, y_train, y_test = split_train_test(
            dataset_df,            
            # **task_params['preprocessing'],
            **preprocessing_kwargs_params,
            n_predict=n_predict,
            drop_not_scalable=False
        )
        if X_train.shape[0]==0:
            print('empty train dataset!')
            break

        # if task.get_parameter("kfold/use_kfold"):
        if kflod_kwargs_params['use_kfold']:
            model, dict_to_save = train_with_cv(
                # storage, X_train, X_test, y_train, y_test, logger, n_predict)
                task_params, X_train, X_test, y_train, y_test, logger, n_predict)
        else:
            model, dict_to_save = train_wo_cv(
                # storage, X_train, X_test, y_train, y_test, logger, n_predict)
                X_train, X_test, y_train, y_test, logger, n_predict, 
                    fs_kwargs_params,
                    preprocessing_kwargs_params,
                    model_type_params,
                    model_kwargs_params,
                    kflod_kwargs_params)

        full_dict_to_save.append(dict_to_save)
        if task.get_parameter('model/save_model'):
            models_dict_to_save.append(model)

    else:    
        pickle.dump(
            full_dict_to_save,
            open("results.pkl", "wb")
        )
        if task.get_parameter('model/save_model'):  
            pickle.dump(
                models_dict_to_save,
                open("models.pkl", "wb")
            )

        task.upload_artifact("results", artifact_object='results.pkl')
        # TODO: сохранение всех моделей, а не только последней, если их необходимо сохранять
        # if str(task_params['model_type_params']['save_model'])=='True':
        if model_type_params['save_model']==True:
            task.upload_artifact("models", artifact_object='models.pkl')
    return task


def to_numeric(x):
    if not isinstance(x, str):
        return x
    try:
        y = eval(x)
        return y
    except:
        return x

# def train_with_cv(storage: ClearMLStorage, X_train, X_test, y_train, y_test, logger, n_predict):
def train_with_cv(task_params, X_train, X_test, y_train, y_test, logger, n_predict):
    # cv_kfold = KFold(**storage.kflod_kwargs_params)
    # cv_kfold = KFold(**task_params['kflod_kwargs_params'])
    cv_kfold = KFold(n_splits=int(task_params['kfold']['n_splits']))

    kfold_model_num = 0
    kfold_model_dict = {}

    columns_train_on_dict = {}

    # if storage.model_type_params["save_kfold_predicts"]:
    if task_params['model_type_params']["save_kfold_predicts"]:
        X_kfold_with_predict = X_train.copy()

    X_test_with_predict = X_test.copy()

    dates_range = pd.Series(pd.date_range(
        *X_train['cut_date'].agg([min, max]).values.tolist()))

    for train_index, valid_index in cv_kfold.split(dates_range):
        model_type = task_params['model_type_params']["model_type"]
        model_kwargs = task_params['model_kwargs_params']
        # model_kwargs.update({key: int(value) for key, value in model_kwargs.items() if (type(value)==str and value.isnumeric())})
        
        # print('clearml_module')
        # print(model_kwargs)
        model_kwargs.update({key: to_numeric(value) for key, value in model_kwargs.items()})
        print(model_kwargs)

        print(f"[INFO] model_type: {model_type}")
        print(f"[INFO] model_kwargs: {model_kwargs}")
        model = RegressionModel(
                # model_type=storage.model_type_params["model_type"],
                # model_kwargs=storage.model_kwargs_params
                
                # model_type=task_params['model_type_params']["model_type"],
                # model_kwargs=task_params['model_kwargs_params']

                model_type=model_type,
                model_kwargs=model_kwargs
            )

        # print("training kfold")
        kfold_model_num += 1

        train_slice = X_train["cut_date"].isin(
            dates_range[train_index].values)
        valid_slice = X_train["cut_date"].isin(
            dates_range[valid_index].values)

        X_kfold_train, y_kfold_train = X_train[train_slice], y_train[train_slice]
        X_kfold_valid, y_kfold_valid = X_train[valid_slice], y_train[valid_slice]

        if y_train.nunique() <= 1:
            print(f"[INFO]: y_train unique values = {y_train.nunique()}. Continue...")
            continue
            # model = RegressionModel(
            #     model_type=storage.model_type_params["model_type"],
            #     model_kwargs=storage.model_kwargs_params
            # )

        columns_to_drop = ['cut_date', 'material_cd',
                            'business_unit', 'window_size']
        columns_to_drop += columns_for_deletion(
            X_train, startswith='predict')
        columns_train_on = list(
            set(X_train.columns) - set(columns_to_drop))

        model.fit(
            X_kfold_train[columns_train_on],
            y_kfold_train,
            eval_set=(X_kfold_valid[columns_train_on], y_kfold_valid),
            verbose=False
        )

        predict = model.predict(X_test[columns_train_on])

        X_test_with_predict[f"model_predict_{kfold_model_num}"] = predict

        columns_train_on_dict[kfold_model_num] = columns_train_on

        valid_predict = model.predict(X_kfold_valid[columns_train_on])

        # if storage.model_type_params["save_kfold_predicts"]:
        save_kfold_predicts = task_params['model_type_params']["save_kfold_predicts"]
        if str(save_kfold_predicts)=='True':
            X_kfold_with_predict.loc[valid_slice,
                                        "model_predict"] = valid_predict

        if str(task_params['model_type_params']["save_model"])=='True':
            kfold_model_dict[kfold_model_num] = kfold_model_dict

        logger.report_scalar(
            "kfold error",
            "rmse",
            iteration=kfold_model_num,
            value=mean_squared_error(y_kfold_valid, valid_predict, squared=False)
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

    # TODO: сохранение всех моделей, а не только последней, если их необходимо сохранять
    if str(task_params['model_type_params']["save_model"])=='True':
        dict_to_save["kfold_model_dict"] = kfold_model_dict

    if str(task_params['model_type_params']["save_kfold_predicts"])=='True':
        dict_to_save["X_kfold_with_predict"] = X_kfold_with_predict

    return model, dict_to_save


# def train_wo_cv(storage: ClearMLStorage, X_train, X_test, y_train, y_test, logger, n_predict):
def train_wo_cv(X_train, X_test, y_train, y_test, logger, n_predict,
                fs_kwargs_params,
                preprocessing_kwargs_params,
                model_type_params,
                model_kwargs_params,
                kflod_kwargs_params,):
    dict_to_save = {}

    model_type = model_type_params["model_type"]
    model_kwargs = model_kwargs_params

    # model_kwargs.update({key: to_numeric(value) for key, value in model_kwargs.items()})
    # print(model_kwargs)
    # TODO: loguru
    print(f"[INFO] model_type: {model_type}")
    print(f"[INFO] model_kwargs: {model_kwargs}")
    model = RegressionModel(
            model_type=model_type,
            model_kwargs=model_kwargs
        )

    columns_to_drop = ['cut_date', 'material_cd', 'analog_group']
    columns_to_drop += columns_for_deletion(X_train, startswith='predict')
    columns_train_on = list(set(X_train.columns) - set(columns_to_drop))
    dict_to_save['columns_train_on'] = columns_train_on
    X_train = X_train[columns_train_on]

    # FS 
    # if str(task_params['fs_kwargs_params']['use_fs'])=='True':
    if fs_kwargs_params['use_fs']==True:
        # rfe = False if str(task_params['fs_kwargs_params']['rfe'])=='False' else True
        rfe = fs_kwargs_params['rfe']
        method = fs_kwargs_params['method']
        n_features = int(fs_kwargs_params['n_features'])
        rfe_n_features = fs_kwargs_params.get('rfe_n_features')
        try:
            rfe_n_features = int(rfe_n_features)
        except:
            rfe_n_features = None

        print(f"[Feature Selection] method={method}, rfe={rfe}.")

        fs = FeatureSelection(method, rfe)
        X_train = fs.filter_X(X_train, y_train, n_features=n_features, rfe_n_features=rfe_n_features)

        columns_train_on = X_train.columns.tolist()

        dict_to_save['columns_train_on_after_fs'] = columns_train_on



    if not model_type_params['train']:
        return None, dict_to_save

    model.fit(X_train, y_train, verbose=False)

    predict = model.predict(X_test[columns_train_on])

    predict = predict[:180]
    y_test = y_test[:180]

    X_test_with_predict = X_test.copy()
    X_test_with_predict["model_predict"] = predict

    dict_to_save['X_test_with_predict'] = X_test_with_predict

    # dict_to_save = {
    #     "columns_train_on": columns_train_on,
    #     "X_test_with_predict": X_test_with_predict
    # }

    # if str(model_type_params["save_model"])=='True':
    if model_type_params["save_model"]==True:
        dict_to_save["model"] = model

    logger.report_scalar(
        "month",
        "rmse",
        iteration=n_predict,
        value=mean_squared_error(y_test, predict, squared=False)
    )
    logger.report_scalar(
        "month",
        "mae",
        iteration=n_predict,
        value=mean_absolute_error(y_test, predict)
    )
    
    return model, dict_to_save

def clone_template_fs(template_task_id, fs_kwargs_params, preprocessing_kwargs_params, model_type_params, model_kwargs_params=None, kflod_kwargs_params=None, queue_name='default'):
    template_task = Task.get_task(task_id=template_task_id)
    param_grid = list(product(ParameterGrid(fs_kwargs_params),
                        ParameterGrid(preprocessing_kwargs_params),
                        ParameterGrid(model_type_params), 
                        ParameterGrid(model_kwargs_params), 
                        ParameterGrid(kflod_kwargs_params)
    ))

    i = 0
    dataset_project = 'zra/datasets2307'
    datasets = list(Task.get_tasks(project_name=dataset_project))
    total_len = len(list(param_grid)) * len(datasets)
    print(f"Total amount of clones = {total_len}")

    for dataset in datasets:
        # dataset_project = 'zra/datasets2307'
        dataset_name = dataset.name
        dataset_file_name = dataset_name + '.parquet'

        for fs, preprocessing, model, model_hyper_params, kfold in param_grid:
            i += 1
            print(f"pair {i}/{total_len}: " , end='')
            # pair = (dataset_param_grid['business_unit'],
            #         dataset_param_grid['analog_group'])
            # window_size = dataset_param_grid['window_size']
            # model_name = model_type_param_grid['model_type']
            print("analog_group: {}, feature_selection: {}, rfe: {}".format(
                *map(repr, (dataset_name, fs['method'], fs['rfe']))))

            cloned_task = Task.clone(source_task=template_task)
            cloned_task.add_tags(
                [dataset_name, f"fs1={fs['method']}", f"rfe={fs['rfe']}"])

            cloned_task.set_parameter(f"dataset/dataset_name", value=dataset_name)
            cloned_task.set_parameter(f"dataset/dataset_file_name", value=dataset_file_name)
            
            for key, value in fs.items():
                cloned_task.set_parameter(f"feature_selection/{key}", value=value)

            for key, value in preprocessing.items():
                cloned_task.set_parameter(f"preprocessing/{key}", value=value)

            for key, value in model.items():
                cloned_task.set_parameter(f"model/{key}", value=value)

            for key, value in model_hyper_params.items():
                cloned_task.set_parameter(
                    f"model_hyper_params/{key}", value=value)
            
            for key, value in kfold.items():
                cloned_task.set_parameter(
                    f"kfold/{key}", value=value)

            Task.enqueue(task=cloned_task, queue_name=queue_name)


# def main(project_name, task_name, dataset_id):
#     storage = ClearMLStorage(project_name=project_name,
#                             task_name=task_name,
#                             dataset_id=dataset_id
#                             )

#     storage.print_params()

#     task = clearml_task_iteration(storage)
#     task.close()

if __name__=="__main__":
    # project_name = 'zra/test'
    project_name = 'zra/fs_2307'
    task_name = 'FeatureSelection'
    # dataset_id = '5dc94de095014553acbe4f011a579241' # 0407

    task_init_params = {'project_name': project_name,
                        "task_name": task_name,
                        "task_type": 'training',
                        }

    Task.add_requirements("pyarrow")
    task = Task.init(**task_init_params)
    # dataset = {"dataset_id": dataset_id,
    #             "dataset_file_name": dataset_file_name}

    dataset_project = 'zra/datasets2307'
    dataset_name = 'ЗРА_1_100'
    dataset_file_name = dataset_name + '.parquet'

    dataset_params = dict(dataset_project=dataset_project,
                        dataset_name=dataset_name,
                        dataset_file_name=dataset_file_name
                        )
    task.connect(dataset_params, name="dataset")

    dataset = Dataset.get(dataset_name=dataset_name, dataset_project=dataset_project)
    dataset_file_path = dataset.get_local_copy()

    dataset_df = pd.read_parquet(os.path.join(
        dataset_file_path, dataset_file_name))

    

    fs_kwargs_params = {'use_fs': True,
                        'method': 'pearson',
                        'n_features': 500,
                        'rfe': False,
                        'rfe_n_features': 300
    }
    task.connect(fs_kwargs_params, 'feature_selection')

    preprocessing_kwargs_params = {
            # "business_unit": "all",
            # "analog_group": '150',
            # "analog_group": '150-40', # 0707
            # "window_size": 60,
            "split_date": "01.01.2021",
            # "n_predict" : 1,
            "target_column_name_formatter": "predict_normalized_{}",
            "log_target": False,
            "scaling": "Standard",
            "scaler_kwargs": None,
            # "drop_not_scalable": True,
        }
    task.connect(preprocessing_kwargs_params, 'preprocessing')


    model_type_params = {
            'train': False,
            # "model_type": "Ridge",
            "model_type": "CatBoostRegressor",
            
            "save_model": False,
        }
    task.connect(model_type_params, 'model')

    model_kwargs_params = {
            "alpha": 1.0,
            
            "max_depth": 5,
            "n_estimators": 500,

            # "depth": 5,
            # "iterations": 200,
            "l2_leaf_reg": 0.01
        }
    task.connect(model_kwargs_params, 'model_hyper_params')

    kflod_kwargs_params = {
            "use_kfold": False,
            "save_kfold_predicts": True,
            "n_splits": 5,
            "shuffle": False
        }
    task.connect(kflod_kwargs_params, 'kfold')


    # renamer = 
    # dataset_df = dataset_df.rename

    n_predict_max = 6
    task = clearml_task_iteration(task, dataset_df,
        fs_kwargs_params,
        preprocessing_kwargs_params,
        model_type_params,
        model_kwargs_params,
        kflod_kwargs_params,
         n_predict_max)

    task.close()    
