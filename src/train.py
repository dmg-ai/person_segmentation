from config import AppConfig

from src.model.trainer import SMPTrainer


def run_training(config):
    trainer = SMPTrainer(config)
    best_onnx_model_path = trainer.run()
    return best_onnx_model_path


if __name__ == "__main__":
    config = AppConfig.parse_raw()
    run_training(config)
