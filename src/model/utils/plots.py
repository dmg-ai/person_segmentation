from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_loss_and_score(
    train_loss: List[float],
    val_loss: List[float],
    train_score: List[float],
    val_score: List[float],
    save_path: str = "",
) -> None:

    fg, ax = plt.subplots(1, 2, figsize=(19, 5))
    ax[0].plot(train_loss, label="train_loss")
    ax[0].plot(val_loss, label="val_loss")
    ax[0].set_title("Loss Curve")
    ax[0].legend(loc="best")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].plot(train_score, label="train_score")
    ax[1].plot(val_score, label="val_score")
    ax[1].set_title("Score Curve")
    ax[1].legend(loc="best")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("score")
    plt.savefig(save_path)


def visualize(
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    Y_pred: torch.Tensor,
    epoch: int,
    tr_loss: float = None,
    save_path: str = "./",
) -> None:
    if Y_val.shape[1] > 1:
        image = np.rollaxis(X_val.type(torch.uint8).numpy(), 1, 4)
        target = np.rollaxis(Y_val.numpy(), 1, 4)
        prediction = np.rollaxis(Y_pred.numpy(), 1, 4)
    else:
        image = np.rollaxis(X_val.numpy(), 1, 4)
        target = Y_val.squeeze(1)
        prediction = Y_pred.squeeze(1)

    plt.figure(figsize=(10, 7))
    if X_val.shape[0] < 6:
        n_images = X_val.shape[0]
    else:
        n_images = 6
    for k in range(n_images):
        plt.subplot(3, n_images, k + 1)
        plt.imshow(image[k], cmap="gray")
        plt.title("Real")
        plt.axis("off")

        plt.subplot(3, n_images, k + 1 + n_images)
        plt.imshow(target[k], cmap="gray")
        plt.title("Mask")
        plt.axis("off")

        plt.subplot(3, n_images, k + 1 + 2 * n_images)
        plt.imshow(prediction[k], cmap="gray")
        plt.title("Output")
        plt.axis("off")
    if tr_loss:
        plt.suptitle("%d epoch - loss: %f" % (epoch + 1, tr_loss))

    plt.savefig(save_path)
