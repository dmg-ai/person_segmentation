import os
import shutil
from pathlib import Path

from tqdm import tqdm

from config import AppConfig
from src.data_preparation import load_dataset


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
    for img_path in tqdm(image_list, desc="Images moving"):
        shutil.move(img_path, img_folder)

    # move related masks to single mask folder
    for img_name in tqdm(os.listdir(img_folder), desc="Masks moving"):
        mask_name = img_name.split(".")[0] + ".png"
        mask_path = list(raw_mask_dir.rglob(mask_name))[0]
        shutil.move(mask_path, mask_folder)

    # delete raw folders
    shutil.rmtree(raw_img_dir)
    shutil.rmtree(raw_mask_dir)


def main(config):
    clearml_config = {
        "dataset_path": config.dataset_path,
        "dataset_name": config.raw_dataset_name,
        "dataset_id": config.raw_dataset_id,
    }

    dataset_path = load_dataset(clearml_config)


def short():
    # print(len(os.listdir("./data/clip_img/1803191139/clip_00000000")))
    # print(len(os.listdir("./data/clip_img/1803191139/clip_00000001")))
    images = list(Path("data/clip_img").rglob("*.jpg"))
    masks = list(Path("data/matting").rglob("*.png"))
    print(len(images))
    print(len(masks))


if __name__ == "__main__":
    config = AppConfig.parse_raw()
    # main(config)
    # run_preparation(config)
    short()
