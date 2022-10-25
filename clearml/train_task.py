from config import AppConfig
from src.train import run_training

from clearml import Task, TaskTypes


def main():

    config: AppConfig = AppConfig.parse_raw()

    task: Task = Task.init(
        project_name=config.project_name,
        task_name="training task",
        task_type=TaskTypes.training,
    )

    clearml_config = {
        "dataset_path": config.dataset_path,
        "dataset_id": config.prepared_dataset_id,
        "dataset_name": config.prepared_dataset_name,
        "project_name": config.project_name,
        "batch_size": config.batch_size,
        "num_epochs": config.num_epochs,
        "image_size": config.resize_shape,
        "model_name": config.model_name,
        "encoder_name": config.model_params["encoder_name"],
        "encoder_weights": config.model_params["encoder_weights"],
        "num_classes": config.model_params["classes"],
        "loss_name": config.loss_name,
    }

    task.connect(clearml_config)

    best_onnx_model_path = run_training(config)

    task.upload_artifact(name="best_onnx", artifact_object=best_onnx_model_path)


if __name__ == "__main__":
    main()
