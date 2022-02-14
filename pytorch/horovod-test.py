"""Modified code from PyTorch MNIST example: https://github.com/pytorch/examples/blob/master/mnist/main.py"""
from __future__ import print_function

import os
import logging

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.utils.data.distributed
import horovod.torch as hvd


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('--num-epochs', type=int, help='Set the number of epochs', default=12)
    add_arg('--batch-size', type=int, help='Set the global batch size', default=32)
    add_arg('--output', help='Set the output directory')
    add_arg('--data', help='Specify the directory with input')

    return parser.parse_args()


def config_logging(filename):
    """Configures logging"""
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=log_format, filename=filename)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch, log_interval, train_sampler):
    model.train()
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler), 100. * batch_idx / len(train_loader), loss.item())    
            )


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test(model, device, test_loader, train_sampler):
    model.eval()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            test_accuracy += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))


if __name__ == '__main__':
    args = parse_args()

    output = args.output
    data = args.data

    gamma = 0.7 # Decay factor of learning rate
    log_interval = 10

    logfile = os.path.join(output, 'logfile')
    tensorboard_dir = os.path.join(output, 'tensorboard')

    config_logging(filename=logfile)

    # Call to horovod init
    hvd.init()

    # Set 1 GPU visible per process
    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())

    # Scale the learning rate
    lr = 0.01
    lr_scaler = hvd.size()
    lr = lr * lr_scaler

    # Setup device
    device = torch.device("cuda")

    kwargs = {
        'num_workers': 1,
        'pin_memory': True,
        'batch_size': args.batch_size
    }

    # Setup transform
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Load data
    train_dataset = datasets.MNIST(data, train=True, download=True,
                       transform=transform)
    test_dataset = datasets.MNIST(data, train=False,
                       transform=transform)

    # Use DistributedSampler to partition the training data
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )

    # Partition dataset among workers
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, **kwargs)

    # Move model to device
    model = Net().to(device)

    # Wrap optimizer in hvd.DistributedOptimizer
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        op=hvd.Average
    )

    # Broadcast parameters from rank 0 to all other processes
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, args.num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval, train_sampler)
        test(model, device, test_loader, train_sampler)
        scheduler.step()
