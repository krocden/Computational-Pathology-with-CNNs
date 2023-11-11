import os
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import _init


class Loader:
    def __init__(self, config: _init.Config, directory, batch_size=60, shuffle=True, transform=None, ):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=False),
            # maybe more image manipulations
        ])
        self.glb = config

    def get_dataloader(self, train_ratio: float):
        torch.manual_seed(self.glb.seed)
        random.seed(self.glb.seed)
        np.random.seed(self.glb.seed)

        dataset = datasets.ImageFolder(
            root=self.directory,
            transform=self.transform
        )

        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # 创建 DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=os.cpu_count()
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count()
        )

        return train_loader, test_loader

        # self.data = DataLoader(
        #     dataset,
        #     batch_size=self.batch_size,
        #     shuffle=self.shuffle,
        #     num_workers=os.cpu_count()
        # )
