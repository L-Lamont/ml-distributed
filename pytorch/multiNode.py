"""Modified code from PyTorch MNIST example: https://github.com/pytorch/examples/blob/master/mnist/main.py"""
from __future__ import print_function

import os
import re
import logging

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


def get_slurm_nodelist():
    """Gets SLURM_NODELIST"""
    try:
        return os.environ['SLURM_NODELIST']
    except KeyError:
        raise RuntimeError('SLURM_NODELIST not found in environment')


def expand_hostlist(hostlist):
    """Create a list of hosts from hostlist"""

    def split_hostlist(hostlist):
        """Split hostlist as commas outside of range expressions ('[3-5]')"""
        in_brackets = False
        cur_host = ''
        for c in hostlist:
            if in_brackets:
                assert c != '['
                if c == ']':
                    in_brackets = False
            elif c == '[':
                in_brackets = True
            elif c == ',':
                assert cur_host != ''
                yield cur_host
                cur_host = ''
                continue
            cur_host += c
        if cur_host:
            yield cur_host

    def expand_range_expression(range_exp):
        """Expand a range expression like '3-5' to 3,4,5"""
        for part in range_exp.split(','):
            sub_range = part.split('-')
            if len(sub_range) == 1:
                sub_range = sub_range * 2
            else:
                assert len(sub_range) == 2
            for i in range(int(sub_range[0]), int(sub_range[1]) + 1):
                yield i

    hosts = []
    try:
        for part in split_hostlist(hostlist):
            m = re.match(r'([^,[\]]*)(\[([^\]]+)\])?$', part)
            if m is None:
                raise ValueError('Invalid part: {}'.format(part))
            prefix = m.group(1) or ''
            if m.group(3) is None:
                hosts.append(prefix)
            else:
                hosts.extend(
                    prefix + str(i) for i in expand_range_expression(m.group(3))
                )
    except Exception as e:
        raise ValueError('Invalid hostlist format "{}": {}'.format(hostlist, e))
    return hosts


def setup(rank, world_size):
    """Setup process group"""
    nodelist = get_slurm_nodelist()
    hosts = expand_hostlist(nodelist)
    master = hosts[0]

    os.environ['MASTER_ADDR'] = master
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def cleanup():
    """Destroys process group"""
    dist.destroy_process_group()


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('--num-epochs', type=int, help='Set the number of epochs', default=12)
    add_arg('--batch-size', type=int, help='Set the global batch size', default=32)
    add_arg('--output', help='Set the output directory')
    add_arg('--data', help='Specify the directory with input')

    return parser.parse_args()


def config_logging():
    """Configures logging"""
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=log_format)


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


def train(model, device, train_loader, optimizer, epoch, log_interval):
    train_loader.sampler.set_epoch(epoch)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, epoch):
    test_loader.sampler.set_epoch(epoch)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f})\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def main():
    args = parse_args()

    lr = 1.0
    gamma = 0.7
    log_interval = 10
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    local_rank = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']

    config_logging()
    logging.info('rank: {}\tlocal_rank: {}\tworld_size: {}'.format(rank, local_rank, world_size))

    setup(rank, world_size)

    device = torch.device('cuda:{}'.format(local_rank))

    # Setup arguments
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}

    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True}

    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    # Setup transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load data
    dataset1 = datasets.MNIST(args.data, train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(args.data, train=False, download=True, transform=transform)

    sampler1 = torch.utils.data.distributed.DistributedSampler(dataset1, shuffle=True, drop_last=True)
    sampler2 = torch.utils.data.distributed.DistributedSampler(dataset2, shuffle=True)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs, sampler=sampler1)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs, sampler=sampler2)

    # Move model to device
    model = Net()
    logging.info('\ndevice: {}\n'.format(device))
    model = DDP(model.to(device), device_ids=[device])

    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, args.num_epochs+1):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        test(model, device, test_loader, epoch)
        scheduler.step()

    cleanup()


if __name__ == '__main__':
    main()
