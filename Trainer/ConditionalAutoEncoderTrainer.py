from imath.Trainer import Trainer
import torch
from imath import backend as P
import imath as pt
import os
import torch.nn as nn
import numpy as np


class ConditionalAutoEncoderTrainer(Trainer):

    def __init__(self, model, ops, criterion, **kwargs):
        super(ConditionalAutoEncoderTrainer, self).__init__(**kwargs)

        self.Encoder = model.Encoder
        self.Decoder = model.Decoder
        self.Classifier = model.Classifier

        self.encoder_op, self.classify_op = ops

        self.criterion = criterion
        self.cla_criterion = nn.CrossEntropyLoss()

    def forward(self, data, forward_mode, **kwargs):
        img = data['x']
        label = data['y']
        batch = img.size(0)

        encode = self.Encoder(img)
        decode = self.Decoder(encode.view(batch, 6, 7, 6))

        reg_img = P.where(img != 0, decode, img)/2. + 0.5
        feature, pred = self.Classifier.train_forward(reg_img, label)

        # loss_dec = self.criterion(img, torch.where(img != 0, decode, img))
        loss_dec = self.criterion(img, decode)
        if loss_dec.item() < 0.0008:

            loss_cla = self.cla_criterion(pred, label)
            acc = torch.mean((torch.argmax(pred, dim=1) == label).float())
            loss = loss_cla*0.01 + loss_dec*100
        else:
            loss_cla = torch.zeros(1).to(img.device)
            acc = torch.zeros(1).to(img.device)
            loss = loss_dec*100

        self.update(self.encoder_op, loss, retain_graph=True)
        self.update(self.classify_op, loss)

        if forward_mode == 'train':

            return {'loss_d': loss_dec.item(), 'loss_c': loss_cla.item(), 'acc': acc.item()}
        else:
            return {'rec': decode, 'x': img, 'feature': feature, 'label': label}

    def evaluate(self, batch_size, device, **kwargs):
        # loss = []
        for n, data in enumerate(self._evaluate(batch_size, device, **kwargs)):
            # loss.append(data['loss'])
            rec = data['rec']
            x = data['x']
            # print(rec.shape)
            index = np.random.choice(np.arange(len(x)))

            for y, r in zip(x[index: index+1], rec[index: index+1]):
                # im.imshow(r.squeeze())
                # print(r)
                origin_img = (y.squeeze()/2.) + 0.5
                save_img = (r.squeeze()/2.) + 0.5
                # print(origin_img, save_img)
                cat = torch.cat([origin_img, save_img], dim=1)
                im.imsave('rec%s.png' % n, cat*256)
        # loss = np.mean(np.array(loss))

        feature_map = {label: [] for label in range(kwargs.get('n_labels'))}
        for forward_test in self._evaluate(batch_size, device, **kwargs):
            labels = forward_test['label']
            features = forward_test['feature']

            for label, feature in zip(labels, features):
                feature_map[label.item()].append(feature.unsqueeze(0))

        for key, value in list(feature_map.items()):
            feature_map[key] = P.cat(value, dim=0) if len(value) is not 0 else []

        print(('Key:', len(list(feature_map.keys()))))
        FAR, FRR, sort_far, sort_frr = im.Evaluate.FR_ROC(feature_map)

        percent = 0.001
        index = min([int(len(sort_far) * (1 - percent)), len(sort_far) - 1])
        thre = sort_far[min([index, len(sort_far) - 1])]
        # thre = fake[index]
        print(('thre', thre, 'index', index, 'len', len(sort_far), len(sort_frr)))
        acc = 0.001
        for i, frr in enumerate(sort_frr):
            if frr.item() >= thre.item():
                acc = 1 - float(i) / len(sort_frr)
                break
        # plt.plot(FAR, FRR)
        # plt.show()
        return acc


    # def save(self):
    #     return {'encoder': self.Encoder.state_dict(), 'decoder': self.Decoder.state_dict(), 'optim': self.optim.state_dict()}
    #
    # def load(self, version_dict):
    #     self.model.Encoder.load_state_dict(version_dict['encoder'])
    #     self.model.Decoder.load_state_dict(version_dict['decoder'])
    #     self.optim.load_state_dict(version_dict['optim'])