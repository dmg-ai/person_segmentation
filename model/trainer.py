import copy
import os

import numpy as np
import segmentation_models_pytorch as smp
import torch
from invoke.config import Config
from model.utils.augmentations import (
    test_augmentations,
    train_augmentations,
)
from model.utils.dataset import CustomImageDataset
from model.utils.plots import plot_loss_and_score, visualize
from torch.utils.data import DataLoader
import wandb

import onnxruntime
import cv2

class SMPTrainer():
    def __init__(self, config: Config):
        self.config = config

        if config.weights:
            model_type = config.weights.split('.')[-1]
            if model_type == 'onnx':
                self.model = onnxruntime.InferenceSession(config.weights)
            elif model_type == 'pt':
                self.model.to(config.device)
                state = torch.load(config.weights)
                self.model.load_state_dict(state)
                self.best_model = self.model
            else:
                pass
        
        if config.setup.inference:
            pass
        else:
            self.model = smp.__getattribute__(self.config.model.name)(**self.config.model.params)

            self.optimizer = torch.optim.__getattribute__(self.config.optimizer.name)(
                self.model.parameters(), **self.config.optimizer.params
            )

            self.criterion = smp.losses.__getattribute__(self.config.loss.name)(
                **self.config.loss.params
            )
            self.criterion.__name__ = self.config.loss.name

            self.scheduler = torch.optim.lr_scheduler.__getattribute__(self.config.scheduler.name)(
                self.optimizer, **self.config.scheduler.params
            )
            
            self.train_transform = None
            self.test_transform = None
            if self.config.use_augmentations:
                self.train_transform = train_augmentations
                self.test_transform = test_augmentations
            
            
            self.best_model = self.model
            self.best_epoch = 1

            self.best_loss = float("inf")
            self.best_score = 0
            
            
            self.train_dataset = CustomImageDataset(
                self.config.dataset_path + "train/images/",
                self.config.dataset_path + "train/masks/",
                self.config.resize_shape,
                transform=self.train_transform,
                num_classes=self.config.model.params.classes,
            )
            self.val_dataset = CustomImageDataset(
                self.config.dataset_path + "val/images/",
                self.config.dataset_path + "val/masks/",
                self.config.resize_shape,
                transform=self.test_transform,
                num_classes=self.config.model.params.classes,
            )
            self.test_dataset = CustomImageDataset(
                self.config.dataset_path + "test/images/",
                self.config.dataset_path + "test/masks/",
                self.config.resize_shape,
                transform=self.test_transform,
                num_classes=self.config.model.params.classes,
            )

            self.train_data = DataLoader(
                self.train_dataset, batch_size=self.config.batch_size, shuffle=True
            )
            self.val_data = DataLoader(
                self.val_dataset, batch_size=self.config.batch_size, shuffle=True
            )
            self.test_data = DataLoader(
                self.test_dataset, batch_size=self.config.batch_size, shuffle=True
            )

            self.device = self.config.device

            self.metrics = [
                smp.utils.metrics.IoU(threshold=self.config.tresholds.iou),
            ]
            self.train_losses = []
            self.val_losses = []
            self.test_losses = []

            self.train_scores = []
            self.val_scores = []
            self.test_scores = []

            self.train_epoch = smp.utils.train.TrainEpoch(
                self.model,
                loss=self.criterion,
                metrics=self.metrics,
                optimizer=self.optimizer,
                device=self.config.device,
                verbose=True,
            )

            self.valid_epoch = smp.utils.train.ValidEpoch(
                self.model,
                loss=self.criterion,
                metrics=self.metrics,
                device=self.config.device,
                verbose=True,
            )
            self.test_epoch = smp.utils.train.ValidEpoch(
                model=self.best_model,
                loss=self.criterion,
                metrics=self.metrics,
                device=self.config.device,
            )
            self.log_path = "experiments/exp{}/".format(
                str(len([i for i in os.listdir("experiments/") if i.startswith("exp")]) + 1)
            )
            os.makedirs(self.log_path)

            if self.config.wandb.logging:
                wandb.init(
                    project=self.config.wandb.project_name,
                    name=self.config.wandb.run_name,
                    entity=self.config.wandb.team_name,
                    config={
                        "epochs": self.config.num_epochs,
                        "batch_size": self.config.batch_size,
                        "experiment": self.log_path.split("/")[1],
                        "model name": self.config.model.name,
                        "encoder name": self.config.model.params.encoder_name,
                    },
                )

    def make_dirs(self) -> None:
        dirs = ["pics", "plots", "weights"]
        for directory in dirs:
            if not os.path.exists(self.log_path + directory):
                os.makedirs(self.log_path + directory)
        print("All files are in " + self.log_path)


    def run(self):
        self.model.to(self.device)
        if self.config.setup.train:
            self.make_dirs()
            for epoch in range(self.config.num_epochs):
                print("Epoch: {}".format(epoch + 1))

                train_logs = self.train_epoch.run(self.train_data)

                self.train_losses.append(train_logs[self.criterion.__name__])
                self.train_scores.append(train_logs[self.metrics[0].__name__])

                valid_logs = self.valid_epoch.run(self.val_data)

                self.val_losses.append(valid_logs[self.criterion.__name__])
                self.val_scores.append(valid_logs[self.metrics[0].__name__])

                if self.config.scheduler.name == "ReduceLROnPlateau":
                    self.scheduler.step(self.val_losses[-1])
                else:
                    self.scheduler.step()

                if epoch == 0 or epoch % 5 == 0:
                    X_val, Y_val = next(iter(self.val_data))
                    Y_pred = self.model(X_val.to(self.device)).detach().to("cpu")
                    if self.config.model.params.classes == 1:
                        Y_pred = torch.where(
                            torch.sigmoid(Y_pred) > self.config.tresholds.class_probability, 1, 0
                        )

                    pic_save_path = self.log_path + f"pics/visualization_{str(epoch+1)}.png"
                    visualize(
                        X_val.type(torch.uint8),
                        Y_val,
                        Y_pred,
                        epoch,
                        tr_loss=train_logs[self.criterion.__name__],
                        save_path=pic_save_path,
                    )

                if valid_logs[self.criterion.__name__] < self.best_loss:
                    self.best_model = copy.deepcopy(self.model)
                    self.best_loss = valid_logs[self.criterion.__name__]
                    pred_best_epoch = self.best_epoch
                    self.best_epoch = epoch
                    torch.save(
                        self.best_model.state_dict(),
                        self.log_path + f"weights/best_model_{self.best_epoch+1}ep.pt",
                    )
                    print(
                        "SAVED: ", self.log_path + f"weights/best_model_{self.best_epoch+1}ep.pt"
                    )
                    if os.path.exists(
                        self.log_path + f"weights/best_model_{pred_best_epoch+1}ep.pt"
                    ):
                        os.remove(self.log_path + f"weights/best_model_{pred_best_epoch+1}ep.pt")
                        print(
                            "DELETED: ",
                            self.log_path + f"weights/best_model_{pred_best_epoch+1}ep.pt",
                        )

                torch.save(
                    self.model.state_dict(), self.log_path + f"weights/last_model_{epoch+1}ep.pt"
                )
                print("SAVED: ", self.log_path + f"weights/last_model_{epoch+1}ep.pt")

                if os.path.exists(self.log_path + f"weights/last_model_{epoch}ep.pt"):
                    os.remove(self.log_path + f"weights/last_model_{epoch}ep.pt")
                    print("DELETED: ", self.log_path + f"weights/last_model_{epoch}ep.pt")

                if self.config.wandb.logging:
                    wandb.log(
                    {
                        f"train/{self.criterion.__name__}": np.float64(self.train_losses[-1]),
                        f"val/{self.criterion.__name__}": np.float64(self.val_losses[-1]),
                        f"train/{self.config.score_name}": np.float64(self.train_scores[-1]),
                        f"val/{self.config.score_name}": np.float64(self.val_scores[-1]),
                    }
                    )
                    print("Logging Wandb.")
                    
                    if epoch == 0 or epoch % 5 == 0:
                        wandb.log({f"pics/": wandb.Image(self.log_path + f"pics/visualization_{str(epoch+1)}.png")})

            save_pic_path = self.log_path + "plots/" + "Curve_Of_Loss_Score.png"
            plot_loss_and_score(
                self.train_losses,
                self.val_losses,
                self.train_scores,
                self.val_scores,
                save_path=save_pic_path,
            )
            if self.config.save_torchscript:
                model_scripted = torch.jit.trace(
                    self.best_model,
                    torch.Tensor(
                        1, 3, self.config.resize_shape[0], self.config.resize_shape[1]
                    ).to(self.device),
                )
                model_scripted.save(self.log_path + "weights/best_model_scripted.pt")
            if self.config.save_onnx:
                x = torch.randn(
                    1,
                    3,
                    self.config.resize_shape[0],
                    self.config.resize_shape[1],
                    requires_grad=True,
                )
                try:
                    self.best_model.encoder.set_swish(memory_efficient=False)
                except:
                    pass
                torch.onnx.export(
                    self.best_model.to(self.device),  # model being run
                    x.to(self.device),  # model input (or a tuple for multiple inputs)
                    self.log_path
                    + "weights/best_model.onnx",  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=11,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=["input"],  # the model's input names
                    output_names=["output"],  # the model's output names
                    dynamic_axes={
                        "input": {0: "batch_size"},  # variable length axes
                        "output": {0: "batch_size"},
                    },
                )

        print("TRAINING OVER")

        if self.config.setup.test:
            print("Testing started.")

            test_logs = self.test_epoch.run(self.test_data)
            self.test_losses.append(test_logs[self.criterion.__name__])
            self.test_scores.append(test_logs[self.metrics[0].__name__])

            if self.config.wandb.logging:
                for metric in test_logs.keys():
                    wandb.summary["test/{}".format(metric)] = np.float64(test_logs[metric])
        if self.config.wandb.logging:
            wandb.finish()
            
    def __detect__(self, ort_session, image):
        preprocessed_image = cv2.resize(image, (256,256))
        preprocessed_image = np.rollaxis(np.array(preprocessed_image, np.float32), 2, 0)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        ort_inputs = {ort_session.get_inputs()[0].name: preprocessed_image}
        mask = ort_session.run(None, ort_inputs)
        mask = torch.sigmoid(torch.Tensor(mask[0])).detach().numpy()
        #mask = mask

        mask = np.rollaxis(mask[0], 0, 3)
        mask = cv2.resize(mask, (image.shape[1],image.shape[0]))
        mask = np.where(mask > 0.7, 1, 0)
        return mask
    def inference(self):
        video_capture = cv2.VideoCapture(0)

        while True:
            # считываем кадр за кадром
            ret, img = video_capture.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = self.__detect__(self.model, img)
            blur = cv2.blur(img,(40,40),0)
            out = img.copy()
            out[mask==0] = blur[mask==0]

            cv2.imshow('Result', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break