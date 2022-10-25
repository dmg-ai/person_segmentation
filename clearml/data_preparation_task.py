from config import AppConfig
from src.data_preparation import load_dataset, run_preparation

from clearml import Dataset, Task, TaskTypes


def main():

    config: AppConfig = AppConfig.parse_raw()
    task: Task = Task.init(
        project_name=config.project_name,
        task_name="data preparation",
        task_type=TaskTypes.data_processing,
    )
    clearml_config = {
        "dataset_path": config.dataset_path,
        "dataset_name": config.raw_dataset_name,
        "dataset_id": config.raw_dataset_id,
    }
    task.connect(clearml_config)
    dataset_path = load_dataset(clearml_config)

    run_preparation(config)

    dataset = Dataset.create(
        dataset_project=config.project_name,
        dataset_name=config.prepared_dataset_name,
    )
    dataset.add_files(config.dataset_path)
    task.set_parameter("prepared_dataset_id", dataset.id)
    dataset.upload()
    dataset.finalize()


if __name__ == "__main__":
    main()
