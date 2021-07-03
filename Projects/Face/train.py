import imath as im
from imath.Model import Model
import torch
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim

class DataLoader(im.DataLoader.DataLoader):

    def __init__(self, **kwargs):
        super(DataLoader, self).__init__(**kwargs)

    def get_data(self, data_root, line, **kwargs):
        name, label = line.split()
        img = im.imread(os.path.join(data_root, name), 1) / 255
        label = int(label)

        return img.view(3, 250, 250), label

class Trainer(im.Trainer.Trainer):

    def __init__(self, model, criterion, optimizer, **kwargs):
        super(Trainer, self).__init__(**kwargs)

        self.model     = model
        self.criterion = criterion
        self.optimizer = optimizer

    def forward(self, data, forward_mode, **kwargs):
        img, label = data

        predict = self.model(img)
        acc = (torch.argmax(predict, dim=1) == label).float().mean()

        if forward_mode == 'train':
            loss = self.criterion(predict, label)
            self.update(self.optimizer, loss)

            return {'loss': loss, 'acc': acc}
        else:
            return {'acc': acc}

if __name__ == "__main__":
    dataloader = DataLoader(
        data_root          = '/home/imath/Downloads/lfw-deepfunneled',
        list_filename      = '/home/imath/Projects/FR_demo/train_list.txt',
        shuffle            = True,
        batch_size         = 32,
        test_data_root     = '/home/imath/Downloads/lfw-deepfunneled',
        test_list_filename = '/home/imath/Projects/FR_demo/test_list.txt',
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
