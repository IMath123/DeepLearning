from imath.Trainer import Trainer
import torch
import imath as pt
import os
import torch.nn as nn


class GenerativeAdversarialTrainer(Trainer):

    def __init__(self, n_dim, K, opG, opD, loss_function, **kwargs):
        super(GenerativeAdversarialTrainer, self).__init__(**kwargs)

        self.K = K
        self.opG = opG
        self.opD = opD
        self.G = self.model.G
        self.D = self.model.D
        self.n_dim = n_dim

        try:
            self.step += 0
        except:
            self.step = 0

        self.criterion = loss_function

    def forward(self, data, forward_mode, **kwargs):
        data = data['x']
        self.step += 1

        batch = data.size(0)
        z = torch.randn(batch, self.n_dim).to(data.device)

        gen_img = self.G(z)
        # print(gen_img.shape)
        rea_img = data.contiguous()

        #Train D

        output_D_real = self.D(rea_img)
        output_D_fake = self.D(gen_img.detach())

        label_real = torch.ones((batch,)).to(data.device)
        # print(output_D_real.shape, label_real.shape)
        # print(output_D_real)
        # loss_D_real = self.criterion(output_D_real.view(-1), label_real)
        loss_D_real = self.criterion(-output_D_real.view(-1)).mean()
        label_fake = torch.zeros((batch,)).to(data.device)
        # loss_D_fake = self.criterion(output_D_fake.view(-1), label_fake)
        loss_D_fake = self.criterion(output_D_fake.view(-1)).mean()

        loss_D = loss_D_real + loss_D_fake

        if self.step % self.K == 0:
            self.update(self.opD, loss_D)
        else:
            self.clean_grad(self.opD)

        #Train G
        label_real = torch.ones((batch,)).to(data.device)
        z = torch.randn(batch, self.n_dim).to(data.device)
        gen_img = self.G(z)
        output_G = self.D(gen_img)

        # loss_G = self.criterion(output_G.view(-1), label_real)
        loss_G = self.criterion(-output_G.view(-1)).mean()
        self.update(self.opG, loss_G)

        package = {'loss_D': loss_D, 'loss_G': loss_G}

        if forward_mode == 'test':
            package['output'] = [output_D_real, output_D_fake, output_G]
            package['gen_img'] = gen_img

        return package

    def evaluate(self, batch_size, device, **kwargs):
        argmax = []
        save_dir = kwargs.get('save_dir')
        one = torch.ones(1).to(device)
        zero = torch.zeros(1).to(device)

        for n, forward_test in enumerate(self._evaluate(batch_size, device, **kwargs)):
            output = forward_test['output']
            if n == 1:
                gem_img = forward_test['gen_img']
                # print(gem_img.shape)
                for i, img in enumerate(gem_img[:3]):
                    im.imsave(os.path.join(save_dir, '{step}_{i}.png'.format(step=self.step, i=i)), (img.squeeze()/2 + 0.5)*256)
                    # im.imshow(img.squeeze())

            # output_D_real = torch.where(output[0] >= 0.5, 1, 0) == torch.ones_like(output[0])
            # output_D_fake = torch.where(output[1] >= 0.5, 1, 0) == torch.zeros_like(output[1])
            output_G = torch.where(output[2] >= 0.5, one, zero) == torch.zeros_like(output[2])
            argmax.append(output_G)

        argmax = torch.cat(argmax, dim=0).float()
        return torch.mean(argmax).item()

    def save(self):
        return {'G': self.model.G.state_dict(), 'D': self.model.D.state_dict(), 'steps': self.step}

    def load(self, version_dict):
        self.model.G.load_state_dict(version_dict['G'])
        self.model.D.load_state_dict(version_dict['D'])
        self.step = version_dict['steps']