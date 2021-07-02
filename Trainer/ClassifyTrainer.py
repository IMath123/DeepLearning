from imath.Trainer import Trainer
import torch


class ClassifyTrainer(Trainer):

    def __init__(self, nlabels, optimizer, loss_function, **kwargs):
        super(ClassifyTrainer, self).__init__(**kwargs)

        self.nlabels = nlabels
        self.optimizer = optimizer
        self.model = kwargs.get('model')

        load_optimizer = kwargs.get('load_optimizer')
        if load_optimizer is not None and load_optimizer and 'optimizer' in list(self.checkpoint.keys()):
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])

        self.loss_function = loss_function

    def forward(self, data, forward_mode, **kwargs):
        x = data['x']
        y = data['y']

        predict = self.model(x)
        argmax = torch.argmax(predict, dim=-1) == y

        if forward_mode == 'train':
            loss = self.loss_function(predict, y)
            acc = torch.mean(argmax.float())
            self.update(self.optimizer, loss)
            return {'loss': loss, 'acc': acc}
        else:
            return {'argmax': argmax}

    def evaluate(self, batch_size, device, **kwargs):
        argmax = []
        for forward_test in self._evaluate(batch_size, device, **kwargs):
            argmax.append(forward_test['argmax'])
        argmax = torch.cat(argmax, dim=0).float()
        return torch.mean(argmax).item()

    def callback(self, forward_train, forward_valid, print_info):
        valid_info = 'valid_loss: %s, valid_acc: %s, ' % (
        forward_valid['loss'].item(), forward_valid['acc'].item()) if forward_valid is not None else ''
        train_info = 'loss: %s, acc: %s,' % (forward_train['loss'].item(), forward_train['acc'].item())
        self.print_log(print_info + train_info + valid_info)