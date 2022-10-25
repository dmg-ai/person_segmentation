import sys
from imp import load_compiled
from pathlib import Path

from config import AppConfig
from src.data_preparation import load_dataset
from src.data_validation import run_validation

from clearml import Task, TaskTypes


def main():

    config: AppConfig = AppConfig.parse_raw()

    task: Task = Task.init(
        project_name=config.project_name,
        task_name="data validation",
        task_type=TaskTypes.data_processing,
    )
    clearml_config = {
        "dataset_path": config.dataset_path,
        "dataset_id": config.prepared_dataset_id,
        "dataset_name": config.prepared_dataset_name,
    }
    task.connect(clearml_config)

    if config.load_prepared_data:
        dataset_path = load_dataset(clearml_config)
        config.dataset_path = Path(dataset_path)
    run_validation(config)


if __name__ == "__main__":
    main()
