import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as TF
from torchvision import models
from torchvision.models import ResNet18_Weights

import pickle

import os

from time import time
from datetime import timedelta

import sys
sys.path.append("/home/kdhsimplepro/kdhsimplepro/AI/Stylers/UNet")

from models import UNet

class ImageGenerator:

    def __init__(self, model_path, device):
        self.device = device

        with open(model_path, "rb") as f:
            self.gen = pickle.load(f)["G_ema"].to(self.device)


    def generate_image(self, n_size):
        z = torch.randn([n_size, self.gen.z_dim]).to(self.device)
        img = self.gen(z, None)
        img = (img * 127.5 + 128).clamp(0, 255) / 255
        return img


class Augmentation:

    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        self.image_generator = ImageGenerator(model_path="pretrained/ffhq.pkl", device=self.device)

        self.gender_classification = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.gender_classification.fc = nn.Linear(self.gender_classification.fc.in_features, 2)
        self.gender_classification.load_state_dict(torch.load("../data/face_gender_classification_transfer_learning_with_ResNet18.pth"))
        self.gender_classification = self.gender_classification.to(self.device)
        self.gender_classification.eval()

        self.transforms = TF.Compose([
            TF.Resize((224, 224)),
            TF.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.unet = UNet().to(self.device)
        self.unet.eval()
        self.unet.load_state_dict(torch.load("../UNet/best_params.pth", map_location=self.device))

        self.forehead0_proto = torch.load("../UNet/forehead0_proto.pth").to(self.device)
        self.forehead1_proto = torch.load("../UNet/forehead1_proto.pth").to(self.device)

        with open("../UNet/threshold.pickle", "rb") as f:
            self.threshold = pickle.load(f)
            
        self.to_pil_image = TF.ToPILImage()


    @torch.no_grad()
    def pred_gender(self, img):
        if img.ndim == 3:
            img = img.unsqueeze(0)

        pred = self.gender_classification(self.transforms(img))
        pred = (F.softmax(pred, dim=1)[:, 1] > 0.95).type(torch.float32)

        return pred


    @torch.no_grad()
    def img2latent(self, img: torch.tensor):
        
        if img.ndim == 3:
            img = img.unsqueeze(0)

        latent = F.interpolate(img, size=256)

        for i in range(len(self.unet.encoder)):
            latent = self.unet.encoder[i](latent)
        
        latent = self.unet.center[0](latent)

        return latent


    @torch.no_grad()
    def run(self, save_dir, n_size):
        cnt = 0
        class_cnt = torch.tensor([0, 0], dtype=torch.long) # class_cnt[0] = forehead0 count, class_cnt[1] = forehead1 count
        start_time = time()

        while cnt < n_size:
            batch_size = 32
            img = self.image_generator.generate_image(n_size=batch_size) # img shape: (batch_size, 3, 1024, 1024)
            is_male = self.pred_gender(img) # (batch_size, 1)

            latents = self.img2latent(img).view(batch_size, -1) # (batch_size, 8192)
            
            dis0 = ((latents - self.forehead0_proto)**2).sum(dim=1) # (batch_size)
            dis1 = ((latents - self.forehead1_proto)**2).sum(dim=1) # (batch_size)

            force_cla = torch.argmin(class_cnt).item()
            
            cla = (dis0 > dis1).type(torch.long)

            difference_dis = torch.abs(dis0 - dis1)

            idx = torch.argmax(((is_male == 1) * (difference_dis > self.threshold) * (cla == force_cla)).type(torch.long)).item()
            
            if is_male[idx] == 1 and difference_dis[idx] > self.threshold and cla[idx] == force_cla:

                pil_img = self.to_pil_image(img[idx].squeeze(0).cpu().detach())
                pil_img = pil_img.resize((256, 256))
                pil_img.save(os.path.join(save_dir, str(cla[idx].item()), f"{cnt}.jpg"))

                cnt += 1
                class_cnt[cla[idx]] += 1

                time_str = str(timedelta(seconds=round(time() - start_time)))

                print("progress: {:6d}/{:6d}, class_ratio: [{:d}, {:d}], time: {:s}".format(cnt, n_size, *class_cnt.tolist(), time_str), end="\r")


if __name__ == '__main__':

    augmentation = Augmentation()

    augmentation.run(save_dir="../data/augmentation_data/forehead/", n_size=100000)
