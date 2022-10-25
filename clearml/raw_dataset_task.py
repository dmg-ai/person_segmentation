from config import AppConfig

from clearml import Dataset


def main():
    config = AppConfig.parse_raw()
    dataset = Dataset.create(
        dataset_name=config.dataset_name_default, dataset_project=config.project_name
    )
    dataset.add_files(config.dataset_path)
    dataset.upload()
    dataset.finalize()


if __name__ == "__main__":
    main()
