import os
import logging

import argparse
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('--num-epochs', type=int, help='Set the number of epochs', default=12)
    add_arg('--batch-size', type=int, help='Set the global batch size', default=32)
    add_arg('--output', help='Set the output directory')
    add_arg('--data', help='Specify the directory with input')
    add_arg('--num-nodes', type=int, help='Dummy argument not used in horovod')
    add_arg('--strategy', help='Dummy argument not used in horovod')

    return parser.parse_args()


def config_logging(filename):
    """Configures logging"""
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=log_format, filename=filename)


class LitMNIST(LightningModule):
    def __init__(self, data_dir=None, hidden_size=64, learning_rate=2e-4, batch_size=32):
        super().__init__()

        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        # Define model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


    ######################
    # Data related hooks #
    ######################
    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)


    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)


    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


def main():
    args = parse_args()

    data_dir = args.data
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    output = args.output
    num_nodes = args.num_nodes
    strategy = args.strategy

    model = LitMNIST(data_dir=data_dir, batch_size=batch_size)
    trainer = Trainer(
        max_epochs=num_epochs,
        strategy=strategy,
        default_root_dir=output
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()
