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
