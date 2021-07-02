from imath.DataLoader import DataLoader, InitializeSample, epo_epos, to_device, Bar
from torch.utils.data import Dataset as torch_Dataset
import torch as P
import os
import imath as pt
import numpy as np
import random
import time


def sum_list(lines):

    ret = []
    for line in lines:
        ret += line
    return ret


class MixtureList():

    def __init__(self, lines):
        # print(len(lines))
        self.index = np.concatenate([i * np.ones(len(lines[i]), dtype=int) for i in range(len(lines))], axis=0)
        self.lines = np.concatenate(lines)

    def shuffle(self, random_seed):
        np.random.seed(random_seed)
        random.seed(random_seed)
        permutation = np.random.permutation(len(self.lines))
        self.index = self.index[permutation]
        self.lines = self.lines[permutation]

    def __getitem__(self, item):
        return (self.index[item], self.lines[item])

    def __len__(self):
        return len(self.lines)



class MixtureDataSet(torch_Dataset):

    def __init__(self, data_roots, mixture_lines, dataloaders):
        self.mixture_lines = mixture_lines
        self.data_roots = data_roots
        self.dataloaders = dataloaders

    def __getitem__(self, item):
        index, line = self.mixture_lines[item]
        # print(line)
        return self.dataloaders[index]._get_data(self.data_roots[index], line)

    def __len__(self):
        return len(self.lines)


class MixtureDataLoader(DataLoader):

    def __init__(self, dataloaders, **kwargs):
        kwargs.get('test_ratio')
        kwargs.get('validation_ratio')

        self.load_info(kwargs)

        self.num_workers = kwargs.get('num_workers')
        self.pin_memory = kwargs.get('pin_memory')

        self.index = kwargs.get('index')
        self.index_valid = kwargs.get('index_valid')
        self.epoch = kwargs.get('epoch')
        self.step = kwargs.get('step')
        self.batch_size = kwargs.get('batch_size')
        self.test_batch_size = kwargs.get('test_batch_size')
        self.valid_batch_size = kwargs.get('valid_batch_size')

        if self.index is None:
            self.index = 0
        if self.index_valid is None:
            self.index_valid = 0
        if self.epoch is None:
            self.epoch = 0
        if self.step is None:
            self.step = 0
        if self._data_root is None:
            self._data_root = ''
        if self.num_workers is None:
            self.num_workers = 0
        if self.pin_memory is None:
            self.pin_memory = False
        if self.test_batch_size is None:
            self.test_batch_size = self.batch_size
        if self.valid_batch_size is None:
            self.valid_batch_size = self.batch_size

        self.loaded = False

        self._shuffle = kwargs.get('shuffle')
        self._random_seed = kwargs.get('random_seed')

        if self._random_seed is None:
            self._random_seed = 0
        if self._shuffle is None:
            self._shuffle = False

        self.dataloaders = dataloaders

    def load(self):
        """TODO: Docstring for load.
        :returns: TODO

        """
        for dataloader in self.dataloaders:
            dataloader.load()

        self._data_roots = [dl._data_root for dl in self.dataloaders]
        self._test_data_roots = [dl._test_data_root if dl._test_data_root is not None else dl._data_root for dl in self.dataloaders]
        self._validation_data_roots = [dl._validation_data_root if dl._validation_data_root is not None else dl._data_root for dl in self.dataloaders]

        self._Train = MixtureList([dl._Train for dl in self.dataloaders])
        self._Test = MixtureList([dl._Test for dl in self.dataloaders])
        self._Valid = MixtureList([dl._Valid for dl in self.dataloaders])

        if self._shuffle:
            self._Train.shuffle(self._random_seed)
            self._Test.shuffle(self._random_seed)
            self._Valid.shuffle(self._random_seed)

        self.running = False
        self.index_test = 0

        train_sampler = InitializeSample(self.index, self._Train)
        valid_sample = InitializeSample(self.index_valid, self._Valid)

        self.train_loader = P.utils.data.DataLoader(MixtureDataSet(self._data_roots, self._Train, self.dataloaders),
                                                        batch_size=self.batch_size,
                                                        num_workers=self.num_workers, pin_memory=self.pin_memory,
                                                        sampler=train_sampler)
        self.test_loader = P.utils.data.DataLoader(MixtureDataSet(self._test_data_roots, self._Test, self.dataloaders),
                                                       batch_size=self.test_batch_size,
                                                       num_workers=0, pin_memory=self.pin_memory)
        self.valid_loader = P.utils.data.DataLoader(MixtureDataSet(self._validation_data_roots, self._Valid, self.dataloaders),
                                                        batch_size=self.valid_batch_size,
                                                        num_workers=0, pin_memory=self.pin_memory,
                                                        sampler=valid_sample)

        self.mode = 'train'
        self.loaded = True

    def summary(self):
        return '\n'.join([dl.summary() for dl in self.dataloaders])

    def name(self):
        return self.__class__.__name__

    def state_dict(self):
        state_dict = {
            'name': str(self.__class__.__name__),
            'shuffle': self._shuffle,

            'random_seed': self._random_seed,
            'test_ratio': self._test_ratio,
            'validation_ratio': self._validation_ratio,
            # =====================================================================
            'index': self.index,
            'index_valid': self.index_valid,
            'epoch': self.get_epoch(),
            'step': self.get_step(),
            'batch_size': self.batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_batch_size': self.valid_batch_size
        }
        return state_dict

    def load_state_dict(self, state_dict):
        if state_dict['name'] != self.name():
            print(f"WARNING: you are loading a state dict with name {state_dict['name']}, but this dataloader's name "
                  f"is {self.name()}")

        super(self.__class__, self).__init__(**state_dict)

    def _mode_2_data_root(self, mode, id):
        if mode == 'train':
            return self._data_roots[id]
        elif mode == 'test':
            return self._test_data_roots[id]
        elif mode == 'valid':
            return self._validation_data_roots[id]
        else:
            assert False

    def _mode_2_lines(self, mode):
        if mode == 'train':
            return self._Train
        elif mode == 'test':
            return self._Test
        elif mode == 'valid':
            return self._Valid
        else:
            assert False

    def _mode_2_len(self, mode):
        if mode == 'train':
            return len(self._Train)
        elif mode == 'test':
            return len(self._Test)
        elif mode == 'valid':
            return len(self._Valid)
        else:
            assert False

    def _mode_2_index(self, mode):
        if mode == 'train':
            return self.index
        elif mode == 'test':
            return self.index_test
        elif mode == 'valid':
            return self.index_valid
        else:
            assert False

    def _mode_2_loader(self, mode):
        if mode == 'train':
            return self.train_loader
        elif mode == 'test':
            return self.test_loader
        elif mode == 'valid':
            return self.valid_loader
        else:
            assert False

    def _mode_2_iter(self, mode):
        if mode == 'train':
            return self.train_iter
        elif mode == 'test':
            return self.test_iter
        elif mode == 'valid':
            return self.valid_iter
        else:
            assert False

    def _set_iter(self, mode):
        if mode == 'train':
            if len(self.train_loader) > 0:
                self.train_iter = iter(self.train_loader)
            else:
                assert False, '没有训练数据'
        elif mode == 'test':

            if len(self.test_loader) > 0:
                self.test_iter = iter(self.test_loader)
            else:
                assert False, '没有测试数据'
        elif mode == 'valid':
            if len(self.valid_loader) > 0:
                self.valid_iter = iter(self.valid_loader)
            else:
                assert False, '没有验证数据'
        else:
            assert False

    def _index_add(self, mode, add):
        mode_len = self._mode_2_len(mode) - 1
        if mode == 'train':
            self.index += add
            self.step += 1
            if self.index >= mode_len:
                self.index = 0
                self.epoch += 1
        elif mode == 'test':
            self.index_test += add
            if self.index_test >= mode_len:
                self.index_test = 0
        elif mode == 'valid':
            self.index_valid += add
            if self.index_valid >= mode_len:
                self.index_valid = 0
        else:
            assert False

    def _get_batch(self, data_mode, **kwargs):
        self.mode = data_mode
        try:
            loader = self._mode_2_iter(data_mode)
            ret = next(loader)
        except AttributeError:
            self._set_iter(data_mode)
            try:
                loader = self._mode_2_iter(data_mode)

                ret = next(loader)
                self._index_add(data_mode, len(ret[0]))
                return ret
            except:
                assert False, ''
        except StopIteration:
            self._set_iter(data_mode)
            try:
                loader = self._mode_2_iter(data_mode)

                ret = next(loader)
                self._index_add(data_mode, ret[0].size(0))
                return ret
            except:
                assert False, ''

        self._index_add(data_mode, len(ret[0]))
        return ret

    def __call__(self, epochs=1, **kwargs):
        self.load()
        mode = kwargs.get('mode')
        device = kwargs.get('device')
        if device is None:
            device = im.default_device

        if mode == 'train':
            data_name = 'train_data'
            data_mode = 'train'

        elif mode == 'test':
            self.index_test = 0
            data_name = 'test_data'
            data_mode = 'test'
        elif mode == 'valid':
            self.index_valid = 0
            data_name = 'valid_data'
            data_mode = 'valid'

        else:
            assert False, 'mode must be train or test'

        len_of_lines = self._mode_2_len(mode)
        # loader = self._mode_2_loader(mode)

        while True:

            package = dict(valid_data=None)

            if mode is 'train' and self.epoch + 1 > epochs:
                break

            time_start = time.time()
            train_data = [value.to(device) if type(value) is P.Tensor else value for value in
                          self._get_batch(data_mode)]

            if mode is 'train' and len(self._Valid) > 0:
                valid_data = [value.to(device) if type(value) is P.Tensor else value for value in
                              self._get_batch('valid')]
                package['valid_data'] = valid_data

            index = self._mode_2_index(mode)

            progress = float(index + 1) / len_of_lines if index != 0 else 1.0

            package[data_name] = train_data

            package['index_train'] = self.index

            if mode == 'train':
                package['print_info'] = f'\r[Epoch {epo_epos(self.epoch, epochs)}][{Bar(progress)}] '
            else:
                package['print_info'] = f'\r[Evaluating][{Bar(progress)}]'

            time_used = time.time() - time_start
            package['time_used'] = time_used

            yield package

            if (mode is 'test' or mode is 'valid') and index == 0:
                break

    def summary(self):
        return '\n'.join([dl.summary() for dl in self.dataloaders])


if __name__ == '__main__':
    mul = MixtureList([[1, 1, 1, 1], [2, 2, 3, 4, 5, 6]])
    mul.shuffle(0)
    print(mul.lines)
    print(mul.index)
