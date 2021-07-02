from imath.Trainer import Trainer
import torch
import imath as pt
import os
import torch.nn as nn
import numpy as np


class VAETrainer(Trainer):

    def __init__(self, optim, lant_dim, criterion, **kwargs):
        super(VAETrainer, self).__init__(**kwargs)

        self.lant_dim = lant_dim

        self.Distribute = self.model.Distribute
        self.Decoder = self.model.Decoder

        self.optim = optim

        self.criterion = criterion

    def forward(self, data, forward_mode, **kwargs):
        img = data['x']
        # print('??')
        batch = img.size(0)

        distrib = self.Distribute(img)
        mul = distrib[:, 0:1]
        log_sig = distrib[:, 1:2]

        epi = torch.randn(batch, self.lant_dim).to(img.device)
        # print(mul.shape, log_sig.shape, epi.shape)
        z = mul + epi * torch.exp(log_sig / 2.)
        z = z.view((batch, 6, 7, 6))

        reconstrust = self.Decoder(z)

        loss = self.criterion(img, reconstrust) * 10
        self.update(self.optim, loss)

        if forward_mode == 'train':
            return {'loss': loss.item()}
        else:
            return {'rec': reconstrust}

    def evaluate(self, batch_size, device, **kwargs):
        # loss = []
        for n, data in enumerate(self._evaluate(batch_size, device, **kwargs)):
            # loss.append(data['loss'])
            rec = data['rec']
            # print(rec.shape)
            if n == 1:
                for r in rec[:2]:
                    im.imshow(r.squeeze())
                    # im.imsave('rec.png', r.squeeze()*256)
        # loss = np.mean(np.array(loss))

        return 1.