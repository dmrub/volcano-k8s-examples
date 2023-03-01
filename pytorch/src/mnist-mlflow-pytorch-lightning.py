#!/usr/bin/env python3
# Based on https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html
import os

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

try:
    from torchmetrics.functional import accuracy
except ImportError:
    from pytorch_lightning.metrics.functional import accuracy

import mlflow.pytorch
from mlflow import MlflowClient

# For brevity, here is the simplest most minimal example with just a training
# loop step, (no validation, no testing). It illustrates how you can use MLflow
# to auto log parameters, metrics, and models.

class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        acc = accuracy(pred, y, task='multiclass', num_classes=10)

        # Use the current of PyTorch logger
        self.log("train_loss", loss, on_epoch=True)
        self.log("acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))

def main():
    # Initialize our model
    mnist_model = MNISTModel()

    # Initialize DataLoader from MNIST Dataset
    train_ds = MNIST(os.getcwd(), train=True,
        download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=32)

    # Initialize a trainer
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=20, log_every_n_steps=10) # , progress_bar_refresh_rate=20

    # Auto log all MLflow entities
    mlflow.pytorch.autolog()

    # Train the model
    with mlflow.start_run() as run:
        trainer.fit(mnist_model, train_loader)

    # fetch the auto logged parameters and metrics
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

if __name__ == '__main__':
    main()
