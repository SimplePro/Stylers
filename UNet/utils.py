import torch
from torch.utils.data import Dataset

import torchvision.transforms.functional as F

from PIL import Image

import os

import pickle


class CelebADataset(Dataset):

    def __init__(self, male_number):
        super().__init__()

        self.male_number = male_number
        self.root_dir = "../data/CelebAMask-HQ"
    
    def __len__(self):
        return len(self.male_number)
    
    def __getitem__(self, index):
        number = self.male_number[index]

        img_file = f"{number}.jpg"
        mask_file = "{:05d}_hair.png".format(number)

        img = Image.open(os.path.join(self.root_dir, "CelebA-HQ-img", img_file)).convert("RGB").resize((256, 256))
        mask = Image.open(os.path.join(self.root_dir, "CelebAMask-HQ-mask-anno", mask_file)).convert("L").resize((256, 256))

        img = F.to_tensor(img)
        mask = F.to_tensor(mask)

        return (img, mask)


if __name__ == '__main__':
    
    with open("../data/celeba_male_number.pickle", "rb") as f:
        male_number = pickle.load(f)

    celeba_dataset = CelebADataset(male_number)

    img, mask = celeba_dataset.__getitem__(0)

    print(img.shape, mask.shape)