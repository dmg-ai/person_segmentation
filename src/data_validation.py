from config import AppConfig
from PIL import Image
from tqdm import tqdm


def run_validation(config: AppConfig):
    images_list = [x for x in config.dataset_path.rglob("*.jpg")]
    for img in tqdm(images_list, desc="Images validation"):
        Image.open(img)
    print("Dataset validation is DONE!")


if __name__ == "__main__":
    config = AppConfig.parse_raw()
    run_validation(config)
