from imath.Trainer import Trainer
import torch
import imath as pt
import os
import torch.nn as nn
import numpy as np


class AutoEncoderTrainer(Trainer):

    def __init__(self, optim, criterion, model, FR_model, **kwargs):
        super(AutoEncoderTrainer, self).__init__(**kwargs)
        # self.model = model
        self.Encoder = model.Encoder
        self.Decoder = model.Decoder
        # self.Segment = self.model.Segment
        self.FR_model = FR_model

        self.optim = optim

        self.criterion = criterion

    def forward(self, data, forward_mode, **kwargs):
        img = data['x']
        batch = img.size(0)
        # print(img.shape)
        # patch_h = int(np.random.uniform(0, 30))
        # patch_w = int(np.random.uniform(0, 30))
        # patch_u = int(np.random.uniform(0, 112 - patch_h))
        # patch_v = int(np.random.uniform(0, 96 - patch_w))
        # que_img = img.clone()
        # que_img[:, patch_u: patch_u + patch_h, patch_v: patch_v + patch_w] = 0
        encode = self.Encoder(img)
        noise = torch.randn(batch, 6 * 7 * 6).to(encode.device)
        encode_noise = encode# + 0.5 * noise
        # print('encode', encode.shape)
        # segment = self.Segment(encode.view(batch, 6, 7, 6))
        # decode = torch.where(segment <= 0., self.Decoder(encode.view(batch, 6, 7, 6)), torch.zeros(1).to(img.device))
        decode = self.Decoder(encode_noise.view(batch, 6, 7, 6))

        loss_face = self.criterion(img, torch.where(img != 0, decode, img))
        loss_blank = self.criterion(img, torch.where(img == 0, decode, img))
        loss = loss_face#*5 + loss_blank
        self.update(self.optim, loss)

        if forward_mode == 'train':
            return {'loss_face': loss_face.item(), 'loss_blank': loss_blank.item()}
        else:
            noise = torch.randn(batch, 6*7*6).to(encode.device)
            encode_noise = encode + 3*noise
            decode_noise = self.Decoder(encode_noise.view(batch, 6, 7, 6))
            return {'rec': decode, 'rec_noise': decode_noise, 'x': img, 'feature': encode,
                    'feature_noise': encode_noise, 'label': data['y']}


    def evaluate(self, batch_size, device, **kwargs):
        # loss = []
        with torch.no_grad():
            feature_map = {label: [] for label in range(kwargs.get('n_labels'))}
            features = []
            features_noise = []
            for n, data in enumerate(self._evaluate(batch_size, device, **kwargs)):
                # loss.append(data['loss'])
                rec = data['rec']
                rec_noise = data['rec_noise']
                # que_img = data['que_img']
                x = data['x']
                features.append(data['feature'].view(-1, 6 * 7 * 6))
                features_noise.append(data['feature_noise'].view(-1, 6 * 7 * 6))

                labels = data['label']
                rec = torch.where(x != 0, rec, x)
                FR_features = self.FR_model.get_feature(rec/2 + 0.5)
                # print(labels)

                for label, feature in zip(labels, FR_features):
                    feature_map[label.item()].append(feature.unsqueeze(0))


                # print(features[0])
                # print(rec.shape)
                index = np.random.choice(np.arange(len(x)))
                # print(rec.shape, index)
                for y, r, r_n in zip(x[index: index+1], rec[index: index+1], rec_noise[index: index+1]):
                    # im.imshow(r.squeeze())
                    origin_img = (y.squeeze()/2.) + 0.5
                    save_img = (r.squeeze()/2.) + 0.5
                    save_img_noise = (r_n.squeeze()/2.) + 0.5
                    # q_i_img = (q_i.squeeze()/2.) + 0.5
                    cat = torch.cat([origin_img, save_img, save_img_noise], dim=1)
                    im.imsave('rec%s.png' % n, cat*256)

            for key, value in list(feature_map.items()):
                feature_map[key] = torch.cat(value, dim=0) if len(value) is not 0 else []

            # print(('Key:', len(list(feature_map.keys()))))
            FAR, FRR, sort_far, sort_frr = im.Evaluate.FR_ROC(feature_map)

            # loss = np.mean(np.array(loss))
            # features = torch.cat(features, dim=0)
            # features = im.Array(features)
            # # pca_features = im.Tensor(im.Evaluate.PCA(features, 3))
            #
            # features_noise = torch.cat(features_noise, dim=0)
            # features_noise = im.Array(features_noise)
            #
            # pca = np.concatenate([features, features_noise], axis=0)
            # print((pca.shape, len(features)))
            # pca_features = im.Tensor(im.Evaluate.PCA(pca, 3))
            #
            # im.visualise.Plot3D(pca_features[:len(features)], (0, 0, 1), 0.1)
            # im.visualise.Plot3D(pca_features[len(features):], (0, 1, 0), 0.1)
            #
            # im.visualise.show()

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