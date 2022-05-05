import cv2 as cv

import torch
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms

import os



class VideoDataset(IterableDataset):
    def _check(self):
        # Check every var
        ## video_path
        assert os.path.exists(self.video_path), "File does not exist"

    def __init__(self, video_path:str, image_size:str):
        super(VideoDataset, self).__init__()
        self.video_path = video_path
        self.video = cv.VideoCapture(self.video_path)
        self.len = int(self.video.get(cv.CAP_PROP_FRAME_COUNT)) - 100
        self.transforms = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
        self._check()

    def __iter__(self):
        i = 0
        while self.video.isOpened() and i < 10:
            i += 1
            _, raw_img = self.video.read()
            if raw_img is None: break
            img = self.transform_img(raw_img)
            yield {"img": img}

    def __len__(self):
        return self.len

    def load_img(self, img_path:str):
        #Load 
        return None

    def transform_img(self, img):
        # Apply all transform to img
        return self.transforms(img)


if __name__ == "__main__":
    video = VideoDataset("data/s01ep01.mkv", 128)
    print(f"shape of pic: {next(iter(video))['img'].shape}")