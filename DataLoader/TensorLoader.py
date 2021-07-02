from imath.DataLoader import DataLoader
import torch as P
import os
import imath as pt
import numpy as np
import random

class TensorLoader(DataLoader):

    def __init__(self, tensors: dict, **kwargs):

        self.tensors = tensors
        self._shuffle = kwargs.get('shuffle')
        self._random_seed = kwargs.get('random_seed')
        if self._random_seed is None:
            self._random_seed = 0
        if self._shuffle is None:
            self._shuffle = False
        N = 0

        self._inputs = dict()
        for name, value in self.tensors.items():
            self._inputs[name] = (value.dtype, value.shape[1:])
            if N != 0 and N != value.size(0):
                assert False, 'dim 0 of each tensor in tensors must be same'
            N = value.size(0)

        self._test_tensors = kwargs.get('test_tensors')
        self._test_ratio = kwargs.get('test_ratio')
        self._validation_tensors = kwargs.get('validation_tensors')
        self._validation_ratio = kwargs.get('validation_ratio')

        assert self._test_ratio is None or self._test_tensors is None, 'you can only choose at most one param ' \
                                                                       'in test_ratio and test_tensors.'
        assert self._validation_ratio is None or self._validation_tensors is None, 'you can only choose at most one param ' \
                                                                                   'in validation_ratio and validation_tensors.'

        test_cut_index = int(N * self._test_ratio) if self._test_ratio is not None else 0
        validation_cut_index = -int(N * self._validation_ratio) if self._validation_ratio is not None else N

        self._Train = dict()
        self._Test = dict()
        self._Valid = dict()
        for name, value in self.tensors.items():
            self._Train[name] = value[test_cut_index: validation_cut_index]
            self._Valid[name] = value[validation_cut_index:]
            self._Test[name] = value[:test_cut_index]

            self.len_of_train = len(self._Train[name])
            self.len_of_test = len(self._Test[name])
            self.len_of_valid = len(self._Valid[name])

        if self._test_tensors is not None:
            self._Test = self._test_tensors
        if self._validation_tensors is not None:
            self._Valid = self._validation_tensors

        self.index = kwargs.get('index')
        self.index_valid = kwargs.get('index_valid')
        self.index_test = 0
        self.epoch = kwargs.get('epoch')
        self.step = kwargs.get('step')

        if self.index is None:
            self.index = 0
        if self.index_valid is None:
            self.index_valid = 0
        if self.epoch is None:
            self.epoch = 0
        if self.step is None:
            self.step = 0


    def _mode_2_len(self, mode):
        if mode == 'train':
            return self.len_of_train
        elif mode == 'test':
            return self.len_of_test
        elif mode == 'valid':
            return self.len_of_valid
        else:
            assert False

    def _get_batch(self, data_mode, batch, **kwargs):
        Tensors = self._mode_2_lines(data_mode)
        start_index = self._mode_2_index(data_mode)

        tensors = {}
        for name, value in Tensors.items():
            tensors[name] = value[start_index: start_index + batch]

        self._index_add(data_mode, batch)

        return tensors

    def state_dict(self):
        state_dict = {
            'name': str(self.__class__.__name__),
            'inputs': DataLoader.save_inputs(self._inputs),
            # 'list_filename': self._list_filename,
            # 'data_root': self._data_root,
            'shuffle': self._shuffle,
            'random_seed': self._random_seed,
            'test_ratio': self._test_ratio,
            # 'test_data_root': self._test_data_root,
            # 'test_list_filename': self._test_list_filename,
            'validation_ratio': self._validation_ratio,
            # 'validation_data_root': self._validation_data_root,
            # 'validation_list_filename': self._validation_list_filename,
            # =====================================================================
            'index': self.index,
            'index_valid': self.index_valid,
            'epoch': self.epoch,
            'step': self.step
        }
        return state_dict

    def load_state_dict(self, state_dict):
        if 'inputs' in state_dict.keys():
            state_dict['inputs'] = DataLoader.load_inputs(state_dict['inputs'])

        if state_dict['name'] != self.name():
            print(f"WARNING: you are loading a state dict with name {state_dict['name']}, but this dataloader's name "
                  f"is {self.name()}")
        super(self.__class__, self).__init__(**state_dict)
