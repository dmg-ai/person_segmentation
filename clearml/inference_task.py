from config import AppConfig
from src.inference import run_inference

from clearml import Task, TaskTypes


def main():

    config: AppConfig = AppConfig.parse_raw()

    task: Task = Task.init(
        project_name=config.project_name,
        task_name="inference task",
        task_type=TaskTypes.inference,
    )

    clearml_config = {"model_path": config.checkpoint_weights}

    task.connect(clearml_config)

    run_inference(config)


if __name__ == "__main__":
    main()
