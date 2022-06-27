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

from utils.storage import ClearMLStorage
from utils.model import RegressionModel
from utils.preprocessing import columns_for_deletion, split_train_test

import warnings
warnings.filterwarnings('ignore')


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


def clone_template(template_task_id, dataset_hyper_params_dict, model_type_hyper_params_dict, kflod_kwargs_hyper_params_dict,  queue_name='default'):
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
            *map(repr, (pair, window_size, model_name))))

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
