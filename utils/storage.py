
class ClearMLStorage():
    def __init__(self, project_name, task_name,
                 dataset_id, dataset_file_name=None,

                use_fs=True,
                 method_name='pearson',
                n_features = 100,
                use_rfe = False,
                rfe_n_features=None,
                 **kwargs) -> None:
        self._set_task_params(project_name, task_name,
                              kwargs.get('task_type', 'training'))
        self._set_dataset(dataset_id, dataset_file_name)
        self._set_fs(use_fs=use_fs, method_name=method_name, n_features=n_features, use_rfe=use_rfe, rfe_n_features=rfe_n_features, **kwargs)

        self._load_params(**kwargs)
        
        # 2806 change
        self._change_dataset_by_window_size(dataset_file_name)
        
        
    # FIXME: Костыль. Храним для каждого размера окна свой паркет, на данном этапе [60, 120, 360].
    def _change_dataset_by_window_size(self, dataset_file_name=None):
        if dataset_file_name is not None:
            return
        if self.dataset_params['window_size']==60:
            self.dataset['dataset_file_name'] = "dataset_ws60.parquet"
        elif self.dataset_params['window_size']==120:
            self.dataset['dataset_file_name'] = "dataset_ws120.parquet"
        elif self.dataset_params['window_size']==360:
            self.dataset['dataset_file_name'] = "dataset_ws360.parquet"
        return 

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

    def _set_fs(self, use_fs, method_name, n_features=100, use_rfe=False, rfe_n_features=None, **params):
        self._use_fs = use_fs

        self._method_name = method_name
        self._n_features = n_features

        self._use_rfe = use_rfe
        self._rfe_n_features = rfe_n_features

        self.fs_kwargs_params = {'use_fs': use_fs,
                                'method': self._method_name,
                                'n_features': self._n_features,
                                'rfe': self._use_rfe,
                                'rfe_n_features': self._rfe_n_features
        }

    def _load_params(self, **params):
        self.dataset_params = {
            "business_unit": "3990",
            # "analog_group": '150',
            "analog_group": '150-40', # 0707
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
            # "model_type": "Ridge",
            "model_type": "CatBoostRegressor",
            "use_kfold": False,
            "save_kfold_predicts": True,
            "save_model": False,
        }
        self.model_type_params.update(params.get('model_type_params', {}))

        self.model_kwargs_params = {
            "alpha": 1.0,
            
            "max_depth": 5,
            "n_estimators": 500,

            # "depth": 5,
            # "iterations": 200,
            "l2_leaf_reg": 0.01
        }
        self.model_kwargs_params.update(params.get('model_kwargs_params', {}))

        self.kflod_kwargs_params = {
            "n_splits": 5,
            "shuffle": False
        }
        self.kflod_kwargs_params.update(params.get('kflod_kwargs_params',  {}))

    def print_params(self):
        print("Dataset:")
        print(self.dataset)

        print("Dataset Preprocessing Params:")
        print(self.dataset_params)

        print("Model type:")
        print(self.model_type_params)

        print('Model Parameters:')
        print(self.model_kwargs_params)

        print('K-Fold Parameters:')
        print(self.kflod_kwargs_params)


