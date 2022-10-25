from pathlib import Path
from typing import List, Union

from pydantic import HttpUrl
from pydantic_yaml import YamlModel


class AppConfig(YamlModel):
    # trainer type
    name: str

    # train,test, inderence
    setup: dict

    # data
    dataset_name_default: str
    dataset_path: Path
    raw_images_path: Path
    raw_masks_path: Path
    use_augmentations: bool
    resize_shape: list

    # clearml info
    project_name: str
    raw_dataset_name: str
    raw_dataset_id: str
    prepared_dataset_name: str
    prepared_dataset_id: str
    load_prepared_data: bool

    # training setup
    num_epochs: int
    batch_size: int
    device: str

    # score function
    score_name: str
    iou_treshold: float

    # model config
    model_name: str
    model_params: dict
    class_probability: float

    # criterion
    loss_name: str
    loss_params: dict

    # optimizer
    optimizer_name: str
    optimizer_params: dict

    # scheduler
    scheduler_name: str
    scheduler_params: dict

    # model formats
    save_onnx: bool

    # inference
    checkpoint_weights: str

    # tensorboard logging
    logging: bool

    # pipeline run
    pipe_name: str
    pipe_version: str
    pipe_local: bool

    # trigger
    scheduler_task_id: str
    scheduler_queue: str
    scheduler_minute: int
    scheduler_hour: int
    scheduler_day: int
    scheduler_weekdays: list

    @classmethod
    def parse_raw(cls, filename: Union[str, Path] = "config.yaml", *args, **kwargs):
        with open(filename) as f:
            data = f.read()
        return super().parse_raw(data, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
