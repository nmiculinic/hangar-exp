import os
import sys
import torch
from pathlib import Path
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from pytorch_lightning.metrics.functional import accuracy
from hangar import Repository
from hangar import make_torch_dataset

class LitModel(pl.LightningModule):
    def __init__(self, lr:float = 0.0001, batch_size:int = 32):
        super().__init__()
        self.save_hyperparameters()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        self.log('val_acc', accuracy(y_hat, y))
        return loss

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--max_elems', type=int, default=60000)
    parser.add_argument('--hangar', action='store_true')
    args = parser.parse_args()

    repo = Repository(path=Path(__file__).parent / "hangar")
    co = repo.checkout()

    if args.hangar:
        dataset = make_torch_dataset([co.columns['digits'], co.columns['label']], index_range=slice(0, args.max_elems))
    else:
        dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    print(len(dataset))
    datapoint, label = dataset[0]
    print(type(datapoint), type(label))
    print("making a loader!")
    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16, shuffle=False)

    # init model
    model = LitModel(lr=args.lr)

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.max_epochs)
    trainer.fit(model, train_loader)
