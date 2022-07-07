import datetime
from itertools import product
import os
import pickle

import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error
# import numpy as np

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
    else:    
        pickle.dump(
            full_dict_to_save,
            open("model_results.pkl", "wb")
        )

        pickle.dump(
            model,
            open("model.pkl", "wb")
        )

        # print(task.get_parameters_as_dict())
        # storage.print_params()

        task.upload_artifact("model_results", artifact_object='model_results.pkl')
        # TODO: сохранение всех моделей, а не только последней, если их необходимо сохранять
        if task_params['model_type_params']['save_model']:
            task.upload_artifact("model", artifact_object='model.pkl')
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
        model_kwargs.update({key: to_numeric(value) for key, value in model_kwargs.items()})

        print('clearml_module')
        print(model_kwargs)
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
            eval_set=(X_kfold_valid[columns_train_on], y_kfold_valid)
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

        if task_params['model_type_params']["save_model"]:
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

    if task_params['model_type_params']["save_model"]:
        dict_to_save["kfold_model_dict"] = kfold_model_dict

    if task_params['model_type_params']["save_kfold_predicts"]:
        dict_to_save["X_kfold_with_predict"] = X_kfold_with_predict

    return model, dict_to_save


# def train_wo_cv(storage: ClearMLStorage, X_train, X_test, y_train, y_test, logger, n_predict):
def train_wo_cv(task_params, X_train, X_test, y_train, y_test, logger, n_predict):
    model_type = task_params['model_type_params']["model_type"]
    model_kwargs = task_params['model_kwargs_params']
    # model_kwargs.update({key: int(value) for key, value in model_kwargs.items() if (type(value)==str and value.isnumeric())})

    model_kwargs.update({key: to_numeric(value) for key, value in model_kwargs.items()})
    
    print('clearml_module')
    print(model_kwargs)
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

    model.fit(X_train[columns_train_on], y_train)

    predict = model.predict(X_test[columns_train_on])
    X_test_with_predict = X_test.copy()
    X_test_with_predict["model_predict"] = predict

    dict_to_save = {
        "columns_train_on": columns_train_on,
        "X_test_with_predict": X_test_with_predict
    }

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
