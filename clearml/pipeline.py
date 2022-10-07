from config import AppConfig

from clearml import PipelineController


def run_pipeline(config: AppConfig):
    pipe = PipelineController(
        name=config.pipe_name,
        project=config.project_name,
        version=config.pipe_version,
    )
    pipe.set_default_execution_queue("default")
    pipe.add_step(
        name="preparation_step",
        base_task_project=config.project_name,
        base_task_name="data preparation",
        # parameter_override={"General/dataset_id": "88b74eebddcd4fe3ac28b41e78971d26"},
        # parameter_override={"dataset_id": "88b74eebddcd4fe3ac28b41e78971d26"},
    )
    pipe.add_step(
        name="validation_step",
        parents=["preparation_step"],
        base_task_project=config.project_name,
        base_task_name="data validation",
        parameter_override={
            "General/dataset_id": "${preparation_step.parameters.General/prepared_dataset_id}"
        },
    )
    pipe.add_step(
        name="training_step",
        parents=["validation_step"],
        base_task_project=config.project_name,
        base_task_name="training task",
        parameter_override={
            "General/dataset_id": "${validation_step.parameters.General/dataset_id}"
        },
    )
    pipe.add_step(
        name="inference_step",
        parents=["training_step"],
        base_task_project=config.project_name,
        base_task_name="inference task",
    )
    if config.pipe_local:
        pipe.start_locally(run_pipeline_steps_locally=True)
    else:
        pipe.start(queue="default")


if __name__ == "__main__":
    config = AppConfig.parse_raw()
    run_pipeline(config)
