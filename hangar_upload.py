import os
import numpy as np
from torchvision.datasets import MNIST
from pathlib import Path
import os
import torch.nn.functional as F
from torchvision.datasets import MNIST
import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from pytorch_lightning.metrics.functional import accuracy
from torch import Tensor
from hangar import Repository

dataset = MNIST(Path(__file__).parent, download=True, transform=transforms.ToTensor())

datapoint, label = dataset[0]
datapoint: Tensor
print(type(datapoint), type(label))
print(datapoint)

repo = Repository(path=Path(__file__).parent / "hangar")
repo.init(
    user_name="Neven Miculinic",
    user_email="neven.miculinic@gmail.com",
    remove_old=True,
)

master_checkout = repo.checkout(write=True)
digits = master_checkout.add_ndarray_column(name="digits", dtype=np.float32, shape=(28, 28))
labels = master_checkout.add_ndarray_column(name="label", dtype=np.int, shape=None, )

print("loading data!")
for i, (digit, label) in tqdm.tqdm(enumerate(dataset)):
    digit: Tensor
    digits[i] = digit.squeeze(dim=0).numpy()
    labels[i] = np.array(label)

master_checkout.commit("Added MNIST to the hangar dataset")
master_checkout.log()
master_checkout.close()
print(repo.summary())