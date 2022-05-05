# Image datasets
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import glob

from PIL import Image

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class ImgDataset(Dataset):
    def __init__(self, img_foldpath:str, img_size=256):
        super(ImgDataset, self).__init__()
        self.img_foldpath = img_foldpath
        self.img_pathlist = glob.glob(f"{img_foldpath}/*.jpeg")
        self.len = len(self.img_pathlist)
        self.img_size = img_size
        self.std = 1
    
    def __getitem__(self, i):
        raw_img = self.load_img(self.img_pathlist[i])
        img = self.transform_img(raw_img)
        item = {
            "img": img,
        }
        if i == 0:
            self.std = max(0, self.std - (self.std / 100))
        return item
    
    def __len__(self):
        return self.len
    
    def load_img(self, img_path:str):
        img = Image.open(img_path)
        return img
    
    def transform_img(self, img):
        # Apply all transform to img
        trfs = transforms.Compose([
            # Randomize
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            AddGaussianNoise(0, self.std)
        ])
        return trfs(img)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = ImgDataset("data/small")
    for i in range(5):
        transform_img = dataset.transform_img(dataset.load_img(dataset.img_pathlist[i]))
        plt.imshow(transform_img.permute(1, 2, 0))
        plt.show()