#!/usr/bin/env python3
import sys
import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio.mlflow-system.svc:9000"

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m", "--use-mlflow", action="store_true", default=False, help="use mlflow"
    )
    parser.add_argument(
        "-t",
        "--mlflow-uri",
        metavar="URI",
        default=None,
        help="mlflow tracking URI (alternatively use environment variable MLFLOW_TRACKING_URI)",
    )
    parser.add_argument(
        "-e",
        "--mlflow-experiment",
        metavar="NAME",
        default="mnist",
        help="mlflow experiment name",
    )
    parser.add_argument(
        "-n",
        "--nodes",
        default=1,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "-g", "--gpus", default=1, type=int, help="number of gpus per node"
    )
    parser.add_argument(
        "-nr", "--nr", default=0, type=int, help="ranking within the nodes"
    )
    parser.add_argument(
        "--epochs",
        default=2,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    args = parser.parse_args()
    train(0, args)


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(gpu, args):
    mlflow_enabled = False
    if args.use_mlflow:
        import mlflow

        print("Use mlflow", file=sys.stderr, flush=True)
        if args.mlflow_uri:
            print(
                "Use mlflow tracking URI:", args.mlflow_uri, file=sys.stderr, flush=True
            )
            mlflow.set_tracking_uri(args.mlflow_uri)
        print(
            "Mlflow experiment name:",
            args.mlflow_experiment,
            file=sys.stderr,
            flush=True,
        )
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow_enabled = True

    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    if mlflow_enabled:
        mlflow.start_run()
        mlflow.log_artifact(os.path.abspath(__file__), 'source code')

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, args.epochs, i + 1, total_step, loss.item()
                    )
                )
            if mlflow_enabled:
                mlflow.log_metric("train_loss", loss.data.item(), i + 1)
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))

    model_fn = "./mnist.model"
    print("Save model to", model_fn, flush=True)
    torch.save(model.state_dict(), model_fn)
    if mlflow_enabled:
        mlflow.log_artifact(os.path.abspath(model_fn), 'model')
        mlflow.end_run()

if __name__ == "__main__":
    main()
