#!/usr/bin/env python3
# Based on https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
import os
import sys
import socket
import time
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


def main():
    parser = argparse.ArgumentParser()
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
    args.world_size = args.gpus * args.nodes
    # os.environ['MASTER_ADDR'] = '10.57.23.164'
    # os.environ['MASTER_PORT'] = '8888'
    print("MASTER_ADDR:", os.environ["MASTER_ADDR"])
    print("MASTER_PORT:", os.environ["MASTER_PORT"])
    
    # check MASTER_ADDR
    master_addr = os.environ["MASTER_ADDR"]
    master_ip = None
    attempts = 0
    max_attempts = 10
    while attempts < max_attempts:
        try:
            master_ip = socket.gethostbyname(master_addr)
            break
        except socket.gaierror as e:
            if e.args[0] == socket.EAI_NONAME:
                print(
                    "[{}/{}] Could not resolve name {}, error code {}, message {}".format(
                        attempts + 1, max_attempts, master_addr, e.args[0], e.args[1]
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                attempts += 1
                time.sleep(10)
            else:
                raise e

    if master_ip is None:
        print("Could not get MASTER_ADDR IP from address", master_addr, file=sys.stderr)
        sys.exit(1)
    print("MASTER_ADDR IP", master_ip)

    os.environ["MASTER_ADDR"] = master_ip

    if args.nr == 0:
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, transform=transforms.ToTensor(), download=True
        )
    mp.spawn(train, nprocs=args.gpus, args=(args,))


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
    print("Start time", datetime.now(), file=sys.stderr)
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=args.world_size, rank=rank
    )
    dist_rank = dist.get_rank()
    print(f"DDP rank {dist_rank}.")

    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    model_without_ddp = model.module
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=False
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
    )

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
                    ),
                    flush=True
                )
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start), flush=True)

    if args.nr == 0:
        print("Save model", flush=True)
        # Print model's state_dict
        print("Model's state_dict:", flush=True)
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        # Print optimizer's state_dict
        print("Optimizer's state_dict:", flush=True)
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])
        model_fn = "./mnist.model"
        print("Save model to", model_fn, flush=True)
        torch.save(model_without_ddp.state_dict(), model_fn)


if __name__ == "__main__":
    main()
