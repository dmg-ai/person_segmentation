import os
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(
        self,
        path_to_img: str,
        path_to_mask: str,
        resize_shape: list = (256, 256),
        transform=None,
        num_classes: int = 3,
    ) -> None:
        self.img_dir = path_to_img
        self.mask_dir = path_to_mask

        self.images = [i for i in sorted(os.listdir(path_to_img)) if i.endswith(".jpg")]
        self.masks = [i for i in sorted(os.listdir(path_to_mask)) if i.endswith(".png")]

        self.transform = transform
        self.resize_shape = tuple(resize_shape)
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        if self.num_classes < 3:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)[:,:,3]
        else:
            mask = cv2.cvtColor(
                cv2.imread(mask_path),
                cv2.COLOR_BGR2RGB,
            )
        image = cv2.resize(image, self.resize_shape)
        mask = cv2.resize(mask, self.resize_shape)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            mask = mask.unsqueeze(0)
            mask = mask / 255.0
        else:
            image = np.rollaxis(image, 2, 0)
            mask = mask / 255.0

            if self.num_classes == 1:
                mask = np.expand_dims(mask, axis=0)
            else:
                mask = np.rollaxis(mask, 2, 0)

        return torch.Tensor(image), torch.Tensor(mask)
