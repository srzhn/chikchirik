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
