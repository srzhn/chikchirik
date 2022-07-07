from utils.storage import ClearMLStorage
from utils.clearml_module import clearml_task_iteration

def main(project_name, task_name, dataset_id):
    storage = ClearMLStorage(project_name=project_name,
                            task_name=task_name,
                            dataset_id=dataset_id
                            )

    storage.print_params()

    task = clearml_task_iteration(storage)
    task.close()

if __name__=="__main__":
    project_name = 'zra/test'
    task_name = 'all'
    dataset_id = '5dc94de095014553acbe4f011a579241'
    main(project_name, task_name, dataset_id)