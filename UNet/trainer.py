import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader

from models import UNet
from utils import CelebADataset

import pickle

from time import time


class Trainer:

    def __init__(
            self,
            device,
            model,
            trainloader,
            validloader,
            lr,
        ):

        self.device = device

        self.model = model.to(self.device)

        self.trainloader = trainloader
        self.validloader = validloader

        self.criterion = nn.BCELoss()

        self.optim = Adam(self.model.parameters(), lr=lr)

        self.train_history = {
            "iter": [],
            "avg": []
        }

        self.valid_history = {
            "iter": [],
            "avg": []
        }

        self.model_params = []

    @torch.no_grad()
    def valid_epoch(self):
        self.model.eval()

        avg_loss = 0

        for (x, y) in self.validloader:
            x, y = x.to(self.device), y.to(self.device)

            pred = self.model(x)

            loss = self.criterion(pred, y)
            avg_loss += loss.item()
            self.valid_history["iter"].append(loss.item())

        avg_loss /= len(self.validloader)
        self.valid_history["avg"].append(avg_loss)

        return avg_loss
    
    def train_epoch(self):
        self.model.train()

        avg_loss = 0

        for (x, y) in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)

            pred = self.model(x)

            loss = self.criterion(pred, y)
            avg_loss += loss.item()
            self.train_history["iter"].append(loss.item())

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        avg_loss /= len(self.trainloader)
        self.train_history["avg"].append(avg_loss)

        return avg_loss


    def run(self, epochs):

        for epoch in range(epochs):
            start_time = time()
            log = "EPOCH: {:d}/{:d}, train_loss: {:.5f}, valid_loss: {:.5f}, time: {:.3f}s".format(epoch+1, epochs, self.train_epoch(), self.valid_epoch(), time() - start_time)
            print(log)
            self.model_params.append(self.model.state_dict())


if __name__ == '__main__':

    with open("../data/celeba_male_number.pickle", "rb") as f:
        male_number = pickle.load(f)
    
    dataset = CelebADataset(male_number=male_number)
    trainset, validset = random_split(dataset, lengths=[6000, len(dataset) - 6000])

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    validloader = DataLoader(validset, batch_size=64)

    model = UNet()

    device = "cuda"

    trainer = Trainer(
        device=device,
        model=model,
        trainloader=trainloader,
        validloader=validloader,
        lr=0.002
    )

    trainer.run(epochs=2)