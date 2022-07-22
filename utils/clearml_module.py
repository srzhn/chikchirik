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

from .storage import ClearMLStorage
from .model import RegressionModel
from .preprocessing import columns_for_deletion, split_train_test

import warnings
warnings.filterwarnings('ignore')



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostRegressor
from boruta import BorutaPy
from BorutaShap import BorutaShap
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

    def _rfe_feature_selection(X: pd.DataFrame, y: pd.Series, n_features=100) -> pd.DataFrame:
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

        if self.method == 'pearson':
            new_X = self._pearson_feature_selection(X, y, n_features=n_features)
        elif self.method == 'catboost':
            new_X = self._catboost_feature_selection(X, y, n_features=n_features)
        else:
            raise ValueError("Unknown method")

        if not self.rfe:
            return new_X
        
        new_X = self._rfe_feature_selection(new_X, y, int(rfe_n_features))
        return new_X


def make_task(storage: ClearMLStorage):
    
    Task.add_requirements("pyarrow")
    task = Task.init(**storage.task_init_params)
    task.connect(storage.dataset, name="dataset")

    # params
    task.connect(storage.dataset_params,
                 'dataset_params')
    task.connect(storage.model_type_params, 'model_type_params')
    task.connect(storage.model_kwargs_params, 'model_kwargs_params')
    task.connect(storage.kflod_kwargs_params, 'kflod_kwargs_params')
    task.connect(storage.fs_kwargs_params, 'fs_kwargs_params')
    return task


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

def load_dataset(task):
    """Выглядит трудозатратным, весь датасет сейчас занимает Гб ОЗУ. 
    Мб разделить на отдельные датасеты и мерджить при необходимости?
    """

    dataset_id = task.get_parameters()['dataset/dataset_id']
    dataset_file_name = task.get_parameters()['dataset/dataset_file_name']

    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_file_path = dataset.get_local_copy()

    dataset_df = pd.read_parquet(os.path.join(
        dataset_file_path, dataset_file_name))
    return dataset_df

# def clearml_task_iteration(task: Task, n_predict_max=12):


def clearml_task_iteration(storage: ClearMLStorage, n_predict_max=12):
    task = make_task(storage)

    # FIXME: Костыль.
    get_ws = int(task.get_parameters()["dataset_params/window_size"])
    task.set_parameter(
                "dataset/dataset_file_name", value=f"dataset_ws{get_ws}.parquet")
    

    # dataset_df = load_dataset(storage)
    dataset_df = load_dataset(task)

    logger = task.get_logger()
    full_dict_to_save = {}
    models_dict_to_save = {}

    
    print(task.get_parameters_as_dict())
    storage.print_params()

    task_params = task.get_parameters_as_dict()
    print(task_params)

    for n_predict in range(n_predict_max):
        # print(f"training : {n_predict}")
        print(dataset_df['window_size'].unique())

        X_train, X_test, y_train, y_test = split_train_test(
            dataset_df,
            # **storage.dataset_params,
            # **task.get_parameters_as_dict()['dataset_params'],
            **task_params['dataset_params'],
            n_predict=n_predict,
            drop_not_scalable=False
        )
        if X_train.shape[0]==0:
            print('empty train dataset!')
            break
        # columns_to_drop = ['cut_date', 'material_cd',
        #                     'business_unit', 'window_size']
        # columns_to_drop += columns_for_deletion(dataset_df, startswith='predict')

        if storage.model_type_params["use_kfold"]:
            model, dict_to_save = train_with_cv(
                # storage, X_train, X_test, y_train, y_test, logger, n_predict)
                task_params, X_train, X_test, y_train, y_test, logger, n_predict)
        else:
            model, dict_to_save = train_wo_cv(
                # storage, X_train, X_test, y_train, y_test, logger, n_predict)
                task_params, X_train, X_test, y_train, y_test, logger, n_predict)
        full_dict_to_save[n_predict] = dict_to_save
        models_dict_to_save[n_predict] = model

    else:    
        pickle.dump(
            full_dict_to_save,
            open("results.pkl", "wb")
        )

        pickle.dump(
            models_dict_to_save,
            open("models.pkl", "wb")
        )

        # print(task.get_parameters_as_dict())
        # storage.print_params()

        task.upload_artifact("results", artifact_object='results.pkl')
        # TODO: сохранение всех моделей, а не только последней, если их необходимо сохранять
        if str(task_params['model_type_params']['save_model'])=='True':
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
    cv_kfold = KFold(n_splits=int(task_params['kflod_kwargs_params']['n_splits']))

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
def train_wo_cv(task_params, X_train, X_test, y_train, y_test, logger, n_predict):
    dict_to_save = {}

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

    columns_to_drop = ['cut_date', 'material_cd',
                       'business_unit', 'window_size']
    columns_to_drop += columns_for_deletion(X_train, startswith='predict')
    columns_train_on = list(set(X_train.columns) - set(columns_to_drop))
    dict_to_save['columns_train_on'] = columns_train_on
    X_train = X_train[columns_train_on]

    # FS 
    if str(task_params['fs_kwargs_params']['use_fs'])=='True':
        rfe = False if str(task_params['fs_kwargs_params']['rfe'])=='False' else True
        method = task_params['fs_kwargs_params']['method']
        n_features = int(task_params['fs_kwargs_params']['n_features'])
        rfe_n_features = task_params['fs_kwargs_params'].get('rfe_n_features')

        fs = FeatureSelection(method, rfe)
        X_train = fs.filter_X(X_train, y_train, n_features=n_features, rfe_n_features=rfe_n_features)

        columns_train_on = X_train.columns.tolist()

        dict_to_save['columns_train_on_after_fs'] = columns_train_on



    # model.fit(X_train[columns_train_on], y_train, verbose=False)
    model.fit(X_train, y_train, verbose=False)

    predict = model.predict(X_test[columns_train_on])
    X_test_with_predict = X_test.copy()
    X_test_with_predict["model_predict"] = predict

    dict_to_save['X_test_with_predict'] = X_test_with_predict

    # dict_to_save = {
    #     "columns_train_on": columns_train_on,
    #     "X_test_with_predict": X_test_with_predict
    # }

    if str(task_params['model_type_params']["save_model"])=='True':
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

def clone_template(template_task_id, dataset_hyper_params_dict, model_type_hyper_params_dict, kflod_kwargs_hyper_params_dict, model_kwargs_params_dict=None,  queue_name='default'):
    template_task = Task.get_task(task_id=template_task_id)
    if model_kwargs_params_dict is None:
        param_grid = list(product(ParameterGrid(dataset_hyper_params_dict),
                            ParameterGrid(model_type_hyper_params_dict),
                            ParameterGrid(kflod_kwargs_hyper_params_dict), 
                            [None]))
    else:
        param_grid = list(product(ParameterGrid(dataset_hyper_params_dict),
                            ParameterGrid(model_type_hyper_params_dict),
                            ParameterGrid(kflod_kwargs_hyper_params_dict),
                            ParameterGrid(model_kwargs_params_dict)))

    total_len = len(list(param_grid))
    print(f"Total amount of clones = {total_len}")
    i = 0

    for dataset_param_grid, model_type_param_grid, kfold_param_grid, model_kwargs_params_grid in param_grid:
        i += 1
        print(f"pair {i}/{total_len}: " , end='')
        pair = (dataset_param_grid['business_unit'],
                dataset_param_grid['analog_group'])
        window_size = dataset_param_grid['window_size']
        model_name = model_type_param_grid['model_type']
        print("{}, window_size: {}, model_name: {}".format(
            *map(repr, (pair, window_size, model_name))))

        cloned_task = Task.clone(source_task=template_task)
        cloned_task.add_tags(
            ["grid_search", model_name, f"bu={pair[0]}", f"group={pair[1]}", f"window_size={window_size}"])

        # cloned_task.set_parameter(f"model_type/model_type", value=model_name)
        for key, value in dataset_param_grid.items():
            cloned_task.set_parameter(f"dataset_params/{key}", value=value)

        for key, value in model_type_param_grid.items():
            cloned_task.set_parameter(f"model_type_params/{key}", value=value)

        for key, value in kfold_param_grid.items():
            cloned_task.set_parameter(
                f"kflod_kwargs_params/{key}", value=value)

        if model_kwargs_params_dict is not None:
            for key, value in model_kwargs_params_grid.items():
                cloned_task.set_parameter(
                    f"model_kwargs_params/{key}", value=value)

        Task.enqueue(task=cloned_task, queue_name=queue_name)
