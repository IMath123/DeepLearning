from imath.DataLoader import DataLoader, epo_epos, Bar
import torch
import os
import imath as pt
import time
import numpy as np
import h5py

class Hd5fLoader(DataLoader):

    def __init__(self, hdf5_filename=None, **kwargs):
        # super(Hd5fLoader, self).__init__()

        self.loaded = False
        self.hd5f_filename = hdf5_filename
        self.hdf5_file = kwargs.get('hdf5_file')

        self.index_dict = {name: 0 for name in ['train', 'valid', 'test']}
        self.variable_num_dict = {name: 0 for name in ['train', 'valid', 'test']}
        self.len_of_X_dict = {name: 0 for name in ['train', 'valid', 'test']}
        self.variable_dict = {name: dict() for name in ['train', 'valid', 'test']}

        self.epoch = 0
        self.step = 0

    def load(self):

        if self.loaded:
            return

        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hd5f_filename, 'r')

        if 'train' not in self.hdf5_file.keys():
            assert False, '使用MultiHdf5Loader来读取此数据集'

        for mode in ['train', 'test', 'valid']:
            self.variable_num_dict[mode] = self.hdf5_file[mode]['variable_num'][()]
            self.len_of_X_dict[mode] = self.hdf5_file[mode]['0'].shape[0]
            print(mode, self.hdf5_file[mode].keys(), self.hdf5_file[mode]['variable_num'].value)
            for key in range(self.hdf5_file[mode]['variable_num'][()]):
                key = str(key)
                print(self.hdf5_file[mode]['batch_size'][()])
                first_data = self.hdf5_file[mode][key][0]
                print(first_data.shape)
                batch = min([self.hdf5_file[mode]['batch_size'][()], self.len_of_X_dict[mode]])
                self.variable_dict[mode][key] = np.empty((batch, *(first_data.shape)), dtype=first_data.dtype)

        self.get_batch_ret = [self.variable_dict[mode][str(i)] for i in range(self.hdf5_file[mode]['variable_num'][()])]

        self.loaded = True


    def unload(self):

        if not self.loaded:
            return

        self.hdf5_file.close()
        self.loaded = False

    def _mode_2_len(self, mode):
        return self.len_of_X_dict[mode]

    def _mode_2_index(self, mode):
        return self.index_dict[mode]

    def get_epoch(self):
        return self.epoch

    def get_step(self):
        return self.step

    def _get_batch(self, data_mode, **kwargs):
        index = self.index_dict[data_mode]
        len_of_X = self.len_of_X_dict[data_mode]
        batch_size = self.hdf5_file[data_mode]['batch_size'][()]

        if index <= len_of_X - batch_size:
            for i in range(self.variable_num_dict[data_mode]):
                data = self.hdf5_file[data_mode][str(i)]
                try:
                    data.read_direct(self.variable_dict[data_mode][str(i)], np.s_[index: index + batch_size])
                except:
                    print('hehe', data_mode, index, batch_size, len_of_X, '000', '\n')
                    assert False

            self.index_dict[data_mode] += batch_size

            return self.get_batch_ret
        else:
            ret = []
            for i in range(self.variable_num_dict[data_mode]):
                data = self.hdf5_file[data_mode][str(i)]
                batch = divmod(len_of_X, batch_size)[1]
                ret_data = self.variable_dict[data_mode][str(i)]
                try:
                    data.read_direct(ret_data[:batch], np.s_[index:])
                except:
                    print('hehe2', data_mode, index, batch_size, len_of_X, '000', '\n')
                    assert False
                ret.append(ret_data[:batch])

            self.index_dict[data_mode] = 0
            if data_mode == 'train':
                self.epoch += 1
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
            self.index_dict['test'] = 0
            data_name = 'test_data'
            data_mode = 'test'

        else:
            assert False, 'mode must be train or test'

        len_of_lines = self.len_of_X_dict[mode]

        while True:

            if mode is 'train' and self.epoch + 1 > epochs:
                break

            time_start = time.time()

            train_data = [torch.from_numpy(value).to(device) for value in self._get_batch(data_mode)]

            if mode is 'train' and self.len_of_X_dict['valid'] > 0:
                valid_data = [torch.from_numpy(value).to(device) for value in self._get_batch('valid')]
            else:
                valid_data = None

            index = self.index_dict[mode]
            self.step += 1

            progress = float(index + 1) / len_of_lines if index != 0 else 1.0

            package = dict()
            package[data_name] = train_data
            package['valid_data'] = valid_data
            package['index_train'] = self.index_dict['train']

            if mode == 'train':
                package['print_info'] = f'\r[Epoch {epo_epos(self.epoch, epochs)}][{Bar(progress)}] '
            else:
                package['print_info'] = f'\r[Evaluating][{Bar(progress)}]'

            time_used = time.time() - time_start
            package['time_used'] = time_used

            yield package

            if mode is 'test' and index == 0:
                break


    def state_dict(self):
        state_dict = {
            'name': str(self.__class__.__name__),
            'hdf5_filename': self.hd5f_filename,

            # =====================================================================
            'index': self.index_dict['train'],
            'index_valid': self.index_dict['valid'],
            'epoch': self.get_epoch(),
            'step': self.get_step(),
        }

        return state_dict

    def load_state_dict(self, state_dict):
        self.hd5f_filename = state_dict['hdf5_filename']
        self.index_dict['train'] = state_dict['index']
        self.index_dict['valid'] = state_dict['index_valid']
        self.epoch = state_dict['epoch']
        self.step = state_dict['step']


class MulitHd5fLoader(DataLoader):

    def __init__(self, hdf5_filename, **kwargs):
        # super(Hd5fLoader, self).__init__()

        self.loaded = False
        self.hd5f_filename = hdf5_filename

        self.step = 0

    def load(self):

        if self.loaded:
            return

        self.hdf5_file = h5py.File(self.hd5f_filename, 'r')

        if 'train' in self.hdf5_file.keys():
            assert False, '使用Hdf5Loader来读取此数据集'

        self.dataloaders = [Hd5fLoader(hdf5_file=self.hdf5_file[str(i)]) for i in range(len(self.hdf5_file.keys()))]

        for mode in ['train', 'test', 'valid']:
            self.variable_num_dict[mode] = len(self.hdf5_file[mode].keys())
            self.len_of_X_dict[mode] = len(self.hdf5_file[mode]['0'].keys())

            for key in self.hdf5_file[mode].keys():
                first_data = self.hdf5_file[mode][key]['0']
                self.variable_dict[mode][key] = np.empty(first_data.shape, dtype=first_data.dtype)

        self.get_batch_ret = [self.variable_dict[mode][str(i)] for i in range(len(self.hdf5_file[mode].keys()))]

        self.loaded = True

    def unload(self):

        if not self.loaded:
            return

        self.hdf5_file.close()
        self.loaded = False

    def _get_batch(self, data_mode, **kwargs):
        index = self.index_dict[data_mode]
        len_of_X = self.len_of_X_dict[data_mode]

        if index < len_of_X - 1:
            for i in range(self.variable_num_dict[data_mode]):
                data = self.hdf5_file[data_mode][str(i)][str(index)]
                data.read_direct(self.variable_dict[data_mode][str(i)])

            self.index_dict[data_mode] += 1
            return self.get_batch_ret
        else:
            ret = []
            for i in range(self.variable_num_dict[data_mode]):
                data = self.hdf5_file[data_mode][str(i)][str(index)]
                batch = data.shape[0]
                ret_data = self.variable_dict[data_mode][str(i)]
                data.read_direct(ret_data[:batch])
                ret.append(ret_data[:batch])

            self.index_dict[data_mode] = 0
            if data_mode == 'train':
                self.epoch += 1
            return ret

    def __call__(self, epochs=None, **kwargs):
        mode = kwargs.get('mode')
        device = kwargs.get('device')

        if mode == 'train':
            data_name = 'train_data'
            data_mode = 'train'
        elif mode == 'test':
            # self.index_test = 0
            data_name = 'test_data'
            data_mode = 'test'
        else:
            assert False, 'mode must be train or test'

        if epochs is None:
            assert False, 'epochs can not be None'
        else:
            if isinstance(epochs, int):
                epochs = (epochs for _ in range(len(self.dataloaders)))

        data_len = 0
        run = np.ones(len(self.dataloaders))

        for i, dataloader in enumerate(self.dataloaders):
            if dataloader._mode_2_len(mode) == 0:
                dataloader.load()
                run[i] = 0

        while True:
            if np.sum(run) == 0:
                break

            train_data = {i: {} for i in range(len(self.dataloaders)) if run[i]}
            progresses = []

            total_index = 0
            for i, dataloader in enumerate(self.dataloaders):

                if run[i]:

                    data_package = dataloader._get_batch(data_mode)
                    train_data[i] = [torch.from_numpy(value).to(device) for value in data_package]

                    data_len = len(train_data[i])
                    dataloader_index = dataloader._mode_2_index(mode)

                    if mode is 'train':
                        if dataloader.get_epoch() + 1 > epochs[i]:
                            run[i] = 0
                    else:
                        if dataloader_index == 0:
                            run[i] = 0

                    total_index += dataloader_index
                    progresses.append(float(dataloader_index + 1) / len(dataloader._mode_2_len(mode))
                                      if dataloader_index != 0 else 1.0)
                else:
                    progresses.append(1.0)

            if mode == 'train':
                self.step += 1

                valid_data = {i: {} for i in range(len(self.dataloaders)) if run[i]}

                count = 0
                for i, dataloader in enumerate(self.dataloaders):
                    if run[i] and dataloader._mode_2_len('valid') > 0:
                        data_package = dataloader._get_batch('valid')
                        valid_data[i] = [torch.from_numpy(value).to(device) for value in data_package]
                        count += 1

                if count == 0:
                    valid_data = None
            else:
                valid_data = None

            package = dict()
            if self.merge:
                train_data = [torch.cat([train_data[j][i] for j in range(len(self.dataloaders)) if j in train_data.keys()], dim=0) for i in range(data_len)]
                if valid_data is not None:
                    valid_data = [torch.cat([valid_data[j][i] for j in range(len(self.dataloaders)) if j in valid_data.keys()], dim=0) for i in
                                  range(data_len)]
            package[data_name] = train_data
            package['valid_data'] = valid_data

            if mode == 'train':
                print_infos = [f'[Epoch {self.dataloaders[i].epoch + 1}/{epochs[i]}][{im.ProgressBar.SimpleBar(progress)}]'
                               for i, progress in enumerate(progresses)]

                package['print_info'] = '\r' + ''.join(print_infos)
            else:
                print_infos = [f'[{im.ProgressBar.SimpleBar(progress)}]' for progress in progresses]
                package['print_info'] = '\r[Evaluating]' + ''.join(print_infos)

            yield package

    def state_dict(self):
        state_dict = {
            'name': str(self.__class__.__name__),
            'hdf5_filename': self.hd5f_filename,

            # =====================================================================
            'index': self.index_dict['train'],
            'index_valid': self.index_dict['valid'],
            'epoch': self.get_epoch(),
            'step': self.get_step(),
        }

        return state_dict

    def load_state_dict(self, state_dict):
        self.hd5f_filename = state_dict['hdf5_filename']
        self.index_dict['train'] = state_dict['index']
        self.index_dict['valid'] = state_dict['index_valid']
        self.epoch = state_dict['epoch']
        self.step = state_dict['step']