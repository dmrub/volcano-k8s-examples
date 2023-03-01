#!/usr/bin/env python3
import mlflow
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchmetrics.functional import accuracy

mlflow.set_tracking_uri('http://mlflow-server-service.mlflow-system:5000')  # set up connection
mlflow.set_experiment('mnist-experiment')          # set the experiment
mlflow.pytorch.autolog()

class MNISTModel(pl.LightningModule):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.nll_loss(self(x), y)
        acc = accuracy(loss, y, task='binary')
        self.log("train_loss", loss, on_epoch=True)
        self.log("acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.5)

mnist_model = MNISTModel()
train_ds = MNIST(os.getcwd(), train=True,
    download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=32)
trainer = pl.Trainer(max_epochs=10)

with mlflow.start_run() as run:
    trainer.fit(mnist_model, train_loader)
