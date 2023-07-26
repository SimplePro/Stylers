import cv2
import torch
import mediapipe as mp

import numpy as np

from copy import deepcopy

import matplotlib.pyplot as plt


class Preprocessing:

    def __init__(self):

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
        
    def transform_img(self, ref_img, landmark_src, landmark_ref):
        ref = deepcopy(ref_img)

        size = ref.shape[0]

        landmark_src = np.array(landmark_src)
        landmark_ref = np.array(landmark_ref)

        x_shift, y_shift = np.mean(landmark_src - landmark_ref, axis=1)

        x_shift = int(x_shift * size)
        y_shift = int(y_shift * size)

        print(x_shift, y_shift)

        img_ = np.zeros_like(src_img)

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

        print(x_shift, y_shift)

        return img_

    
    def preprocess_(self, src_path, ref_path):
        src_img = cv2.resize(cv2.imread(src_path), dsize=((256, 256)))
        ref_img = cv2.resize(cv2.imread(ref_path), dsize=((256, 256)))

        print(src_img, ref_img)

        landmark_src = self.detect_landmarks(src_img)
        landmark_ref = self.detect_landmarks(ref_img)

        if landmark_src != None and landmark_ref != None:
            img_ = self.transform_img(ref_img, landmark_src, landmark_ref)

            return img_
    
        return None


if __name__ == '__main__':

    src_img = cv2.resize(cv2.imread("data/male/0.jpg"), dsize=((256, 256)))

    preprocessing = Preprocessing()
    transform_img = preprocessing.preprocess_(src_path="elonmusk_left.png", ref_path="stargan-v2/assets/representative/custom/src/1/Screenshot from 2023-07-26 18-34-55.png")

    plt.imshow(transform_img)
    plt.show()