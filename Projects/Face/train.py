import imath as im
from imath.Model import Model
import torch
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from trainers import Trainer
from dataloaders import DataLoader


if __name__ == "__main__":
    dataloader = DataLoader(
        data_root     = '/home/dj/Downloads/lfw-deepfunneled',
        list_filename = '/home/dj/Downloads/lfw-deepfunneled/train_list.txt',
        shuffle       = True,
        batch_size    = 64,
        #  test_data_root     = '/home/dj/Downloads/lfw-deepfunneled',
        #  test_list_filename = '/home/dj/Downloads/lfw-deepfunneled/test_list.txt',
    )

    model = Model('./models/little_net.yaml')
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        params       = model.parameters(),
        lr           = 0.1,
        momentum     = 0.9,
        weight_decay = 5e-4,
    )

    trainer = Trainer(model, criterion, optimizer, dataloader=dataloader)

    trainer.fit(10, 'checkpoints/')
