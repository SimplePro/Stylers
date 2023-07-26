import torch
from torchvision.transforms import functional as TF

import cv2
import mediapipe as mp

import numpy as np

from PIL import Image

from copy import deepcopy

import matplotlib.pyplot as plt

import sys

sys.path.append("/home/kdhsimplepro/kdhsimplepro/AI/Stylers/UNet")

from models import UNet

import warnings
warnings.filterwarnings("ignore")


class Preprocessing:

    def __init__(self, device):

        self.device = device

        self.unet = UNet().to(self.device)
        self.unet.load_state_dict(torch.load("/home/kdhsimplepro/kdhsimplepro/AI/Stylers/UNet/best_params.pth", map_location=device))

        for p in self.unet.parameters():
            p.requires_grad = False

        self.unet.eval()

        self.mpFaceMesh = mp.solutions.face_mesh
        self.face_mesh = self.mpFaceMesh.FaceMesh(max_num_faces=1)

    def detect_landmarks(self, image: np.array):

        result = self.face_mesh.process(image).multi_face_landmarks

        if result != None:
            result = result[0].landmark

            landmarks_x, landmarks_y = [], []

            for lm in result:
                landmarks_x.append(lm.x)
                landmarks_y.append(lm.y)

            return [landmarks_x, landmarks_y]

        return None
    

    def get_segmentation_mask(self, img: np.array):
        img = TF.to_tensor(Image.fromarray(img)).to(self.device).unsqueeze(0)
        mask = self.unet(img).squeeze(0).cpu().detach()

        mask_img = np.expand_dims(np.array(TF.to_pil_image(mask)), axis=2) / 255.

        return mask_img

    
    def get_shift_n(self, landmark_src, landmark_ref):
        landmark_src = np.array(landmark_src)
        landmark_ref = np.array(landmark_ref)

        x_shift, y_shift = np.mean(landmark_src - landmark_ref, axis=1)

        return x_shift, y_shift
    
        
    def shift_img(self, ref_img, x_shift, y_shift):
        ref = deepcopy(ref_img)

        size = ref.shape[0]

        x_shift = int(x_shift * size)
        y_shift = int(y_shift * size)

        img_ = np.zeros_like(ref)

        if x_shift >= 0:
            if y_shift >= 0:
                img_[y_shift:, x_shift:, :] = ref[:size-y_shift, :size-x_shift, :]

            elif y_shift < 0:
                img_[:y_shift, x_shift:, :] = ref[-y_shift:, :size-x_shift, :]

        elif x_shift < 0:
            if y_shift >= 0:
                img_[y_shift:, :x_shift, :] = ref[:size-y_shift, -x_shift:, :]

            elif y_shift < 0:
                img_[:y_shift, :x_shift, :] = ref[-y_shift:, -x_shift:, :]

        return img_
    
    def preprocess_(self, src_path, ref_path):
        src_img = cv2.resize(cv2.imread(src_path), dsize=((256, 256)))
        ref_img = cv2.resize(cv2.imread(ref_path), dsize=((256, 256)))

        landmark_src = self.detect_landmarks(src_img)
        landmark_ref = self.detect_landmarks(ref_img)

        src_mask = self.get_segmentation_mask(src_img)
        ref_mask = self.get_segmentation_mask(ref_img)

        x_shift, y_shift = 0, 0

        if landmark_src != None and landmark_ref != None:
            x_shift, y_shift = self.get_shift_n(landmark_src, landmark_ref)
            shifted_ref_img = self.shift_img(ref_img, x_shift, y_shift)
        
        shifted_ref_mask = self.shift_img(ref_mask, x_shift, y_shift)
        
        src_face = src_img * (1 - src_mask)
        ref_hair_src_face = src_face * (1 - shifted_ref_mask) + shifted_ref_img * shifted_ref_mask
        preprocessed_ref = src_img * (1 - src_mask)
        # shifted_ref_img * shifted_ref_mask
        preprocessed_ref = np.clip(ref_hair_src_face, 0, 255).astype(np.uint8)
        preprocessed_ref = cv2.cvtColor(preprocessed_ref, cv2.COLOR_BGR2RGB)

        return preprocessed_ref


if __name__ == '__main__':

    src_img = cv2.resize(cv2.imread("data/male/0.jpg"), dsize=((256, 256)))

    preprocessing = Preprocessing(device="cuda")
    shifted_ref_img = preprocessing.preprocess_(src_path="data/male/0.jpg", ref_path="data/male/1000.jpg")

    plt.imshow(shifted_ref_img)
    plt.show()