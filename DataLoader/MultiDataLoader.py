from imath.DataLoader import DataLoader, Bar
import torch as P
import os
import imath as pt
import numpy as np
import json
import time
import random


def list_concatenate(data, axis):
    """TODO: Docstring for jfunction.

    :arg1: TODO
    :returns: TODO

    """
    return np.concatenate(np.array(data), axis=axis)

def merge_method(dtype):
    """TODO: Docstring for merge_method.

    :dtype: TODO
    :returns: TODO

    """
    if dtype == P.Tensor:
        return P.cat
    elif dtype == np.ndarray:
        return np.concatenate
    elif dtype == list or dtype == tuple:
        return list_concatenate
    else:
        assert False, f'undefined date type({dtype})'

def merge_function(data, dataloaders, data_len):
    """to merge data

    :datas: TODO
    :returns: TODO

    """
    return [merge_method(type(data[list(data.keys())[0]][i]))([data[j][i] for j in range(len(dataloaders)) if j in data.keys()], 0) for i in range(data_len)]


class MultiDataLoader(DataLoader):

    def __init__(self, dataloaders, merge=False, **kwargs):
        # assert len(dataloaders) == len(weights)

        self.step = 0
        self.merge = merge

        self.dataloaders = dataloaders

    def save_info(self, save_name):
        for n, dataloader in enumerate(self.dataloaders):
            Map = dict()
            Map[n] = dataloader.get_info()

        with open(save_name, 'w') as file:
            json.dump(Map, file)

    def load_info(self, info):
        if isinstance(info, str):
             with open(info, 'r') as file:
                 info_map = json.load(file)
                 abs_path = os.path.dirname(os.path.abspath(info))
                 info_map['info_path'] = abs_path
                 for n in range(len(self.dataloaders)):
                     info_map[n]['info_path'] = abs_path

             return self.load_info(info_map)
        else:
            for n, dataloader in enumerate(self.dataloaders):
                dataloader.load_info(info[n])

    def get_epoch(self):
        return min([dataloader.get_epoch() for dataloader in self.dataloaders if dataloader._mode_2_len('train') > 0])

    def load(self):
        for dataloader in self.dataloaders:
            dataloader.load()

    def __call__(self, epochs=1, **kwargs):
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
        elif mode is None:
            data_name = 'data'
            data_mode = 'train'
            mode      = 'train'
        else:
            assert False, 'mode must be train or test'

        data_len = 0
        run = np.ones(len(self.dataloaders))

        for i, dataloader in enumerate(self.dataloaders):
            if dataloader._mode_2_len(data_mode) == 0:
                dataloader.load()
                run[i] = 0

        while True:
            if np.sum(run) == 0:
                break

            time_start = time.time()

            train_data = {i: {} for i in range(len(self.dataloaders)) if run[i]}
            progresses = []

            total_index = 0
            for i, dataloader in enumerate(self.dataloaders):

                if run[i]:

                    data_package = dataloader._get_batch(data_mode)
                    train_data[i] = [value.to(device) if type(value) is P.Tensor else value for value in data_package]

                    data_len = len(train_data[i])
                    dataloader_index = dataloader._mode_2_index(data_mode)

                    if mode is 'train':
                        if self.get_epoch() + 1 > epochs:
                            run[i] = 0
                    else:
                        if dataloader_index == 0:
                            run[i] = 0

                    total_index += dataloader_index
                    progresses.append(float(dataloader_index + 1) / len(dataloader._mode_2_lines(data_mode))
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
                        valid_data[i] = [value.to(device) if type(value) is P.Tensor else value for value in data_package]
                        count += 1

                if count == 0:
                    valid_data = None
            else:
                valid_data = None

            package = {'step': self.get_step()}
            if data_mode == 'train':
                package['epoch'] = self.get_epoch()

            if self.merge:
                train_data = merge_function(train_data, self.dataloaders, data_len)
                if valid_data is not None:
                    valid_data = merge_function(valid_data, self.dataloaders, data_len)
            package[data_name] = train_data
            package['valid_data'] = valid_data

            if mode == 'train':
                print_infos = [f'[Epoch {self.dataloaders[i].get_epoch() + 1}][{im.ProgressBar.SimpleBar(progress)}]'
                               for i, progress in enumerate(progresses)]

                package['print_info'] = f'\r[Total Epoch {self.get_epoch() + 1} / {epochs}]' + ''.join(print_infos)
            else:
                print_infos = [f'[{im.ProgressBar.SimpleBar(progress)}]' for progress in progresses]
                package['print_info'] = '\r[Evaluating]' + ''.join(print_infos)


            time_used = time.time() - time_start
            package['time_used'] = time_used

            yield package


    def state_dict(self):
        state_dict = {'step': self.get_step()}
        for i, dataloader in enumerate(self.dataloaders):
            state_dict[i] = dataloader.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        for i, dataloader in enumerate(self.dataloaders):
            dataloader.load_state_dict(state_dict[i])

        self.step = state_dict['step']

    def summary(self):
        return '\n'.join([dl.summary() for dl in self.dataloaders])

    def __make_hdf5_process_multi(self, hdf5_file, mode):

        data_set = hdf5_file.create_group(mode)
        variable_num = None
        i = 0

        while True:
            dataloader_data = [dataloader._get_batch(mode) for dataloader in self.dataloaders]
            data = [np.concatenate([dataloader_data[j][k] for j in range(len(dataloader_data))], axis=0) for k in range(len(dataloader_data[0]))]

            if variable_num is None:
                variable_num = len(data)
                variable_groups = [data_set.create_group(str(n)) for n in range(variable_num)]

            for variable, group in zip(data, variable_groups):
                group.create_dataset(str(i), data=variable)

            i += 1
            bar = '\r[making {mode} data]'
            for dataloader in self.dataloaders:
                len_of_lines = dataloader._mode_2_len(mode)
                index = dataloader._mode_2_index(mode)
                progress = float(index + 1) / len_of_lines if index != 0 else 1.0
                bar += f'[{Bar(progress)}]'

            yield bar

            if index == 0:
                if mode == 'train':
                    self.epoch -= 1
                break
    #
    def make_hdf5(self, hdf5_filename, **kwargs):
        import h5py
        if os.path.exists(hdf5_filename):
            assert False, f'ERROE: file {hdf5_filename} exists.'

        self.load()

        hdf5_file = h5py.File(hdf5_filename, 'w')

        for mode in ['train', 'valid', 'test']:
            for i, dataloader in enumerate(self.dataloaders):
                hdf5_dataloader_group = hdf5_file.create_group(str(i))
                for progress in dataloader._make_hdf5_process(hdf5_dataloader_group, mode):
                    print(progress, end='\r')

        hdf5_file.close()
