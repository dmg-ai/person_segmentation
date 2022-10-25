import os
import shutil
from pathlib import Path

import numpy as np
from clearml import Dataset
from config import AppConfig
from tqdm import tqdm


def load_dataset(config):
    if len(os.listdir(config["dataset_path"])) > 0:
        shutil.rmtree(config["dataset_path"])
    os.makedirs(config["dataset_path"], exist_ok=True)

    # deleting train, val, test to check correct loading data from clearml
    # if "train" in os.listdir(config["dataset_path"]):
    #     shutil.rmtree(config["dataset_path"] / "train")
    #     shutil.rmtree(config["dataset_path"] / "val")
    #     shutil.rmtree(config["dataset_path"] / "test")
    #     print("Train, val, test data removed.")

    dataset = Dataset.get(dataset_id=config["dataset_id"])
    dataset_path = dataset.get_mutable_local_copy(
        target_folder=config["dataset_path"], overwrite=False
    )
    return dataset_path


def run_preparation(config):
    image_list = []
    raw_img_dir = config.dataset_path / config.raw_images_path
    raw_mask_dir = config.dataset_path / config.raw_masks_path
    image_list = list(raw_img_dir.rglob("*.jpg"))

    img_folder = config.dataset_path / Path("images")
    mask_folder = config.dataset_path / Path("masks")
    Path.mkdir(img_folder, exist_ok=True)
    Path.mkdir(mask_folder, exist_ok=True)

    # move all images to single img folder
    for img_path in tqdm(image_list):
        shutil.move(img_path, img_folder)

    # move related images to single mask folder
    for img_name in tqdm(os.listdir(img_folder)):
        mask_name = img_name.split(".")[0] + ".png"
        mask_path = list(raw_mask_dir.rglob(mask_name))[0]
        shutil.move(mask_path, mask_folder)

    # delete raw folders
    shutil.rmtree(raw_img_dir)
    shutil.rmtree(raw_mask_dir)

    # train val test split
    image_list = Path("data/images").rglob("*.jpg")
    names = [file.name.split(".")[0] for file in image_list]
    train_names, val_names, test_names = np.split(
        np.array(names), [int(len(names) * 0.7), int(len(names) * 0.8)]
    )

    # creating folders for train val test
    for mode in ["train", "val", "test"]:
        os.makedirs(f"data/{mode}/images", exist_ok=True)
        os.makedirs(f"data/{mode}/masks", exist_ok=True)

    # move images, masks to train, val, test folders
    for names, mode in zip(
        [train_names, val_names, test_names], ["train", "val", "test"]
    ):
        print(f"\nProcessing {mode}")
        for name in tqdm(names):
            shutil.move(f"data/images/{name}.jpg", f"data/{mode}/images/")
            shutil.move(f"data/masks/{name}.png", f"data/{mode}/masks/")

    # delete tmp folders
    shutil.rmtree(img_folder)
    shutil.rmtree(mask_folder)


if __name__ == "__main__":
    config: AppConfig = AppConfig.parse_raw()
    run_preparation(config)
