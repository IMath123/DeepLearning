import imath as im
import torch

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
