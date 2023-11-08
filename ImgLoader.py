import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

class Loader:
    def __init__(self, directory, batch_size=60, shuffle=True, transform=None):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=False),
            # maybe more image manipulations
        ])

    def get_dataloader(self):

        dataset = datasets.ImageFolder(
            root=self.directory,
            transform=self.transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=os.cpu_count()  # 根据你的机器配置调整线程数
        )
        return dataloader
