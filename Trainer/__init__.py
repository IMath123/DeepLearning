import json
import torch
from imath.DataLoader import DataLoader
from imath.Controller import Controller, ControllerSet
import imath.ProgressBar
import os
import numpy as np
import torch.nn as nn
import torch.optim as op
from matplotlib import pyplot as plt
import time, datetime
from imath.timer import Timer
import h5py as h5


save_type = [nn.Module, op.Optimizer, DataLoader, Controller]


def is_saved_subclass(value):
    global save_type
    for t in save_type:
        if issubclass(type(value), t):
            return True

    return False


class Trainer(object):
    '''
    these methods must be defined:
     - __init__
     - forward
     - evaluate

     - save(choice)
     - load(choice)
     - callback(choice)

    '''

    def __init__(self, **kwargs):
        self.kwargs      = kwargs

        self.dataloader  = kwargs.get('dataloader')
        self.save_ignore = kwargs.get('save_ignore')
        controllers = kwargs.get('controllers')

        if self.save_ignore is not None:
            if isinstance(self.save_ignore, list) is False:
                self.save_ignore = [self.save_ignore]

        if controllers is None:
            controllers = []
            self.controllerset = ControllerSet()
        elif type(controllers) == ControllerSet:
            self.controllerset = controllers
        else:
            self.controllerset = ControllerSet(*controllers)

        self.checkpoint   = {'best_eval_score': -np.inf, 'version_names': list()}
        self._basic_names = list(self.checkpoint.keys())

        self._training    = True
        self._validing    = False
        self._testing     = False

        self._updated     = False

        self._logs        = []
        self.log          = True

    def resume(self, model_dir, version_name=None, ignore=None, old_version=False, compatible_mode=False):

        if isinstance(ignore, list) is False:
            ignore = [ignore]
        # print(self.checkpoint.keys())
        # TODO 以后的版本移除旧版的读取模型的方式
        if old_version:
            self.checkpoint = torch.load(model_dir)
            self._load_oldversion(version_name, ignore)
        else:
            checkpoint_filename = os.path.join(model_dir, 'checkpoint.json')
            if os.path.isfile(checkpoint_filename) == False:
                c = input(f'checkpoint file "{checkpoint_filename}" not exists. Continue?(y/n)')
                if c.lower() == 'y':
                    print("[Warning] resume fail!")
                    return
                elif c.lower() == 'n':
                    exit(-1)
                else:
                    exit(-1)
            try:
                # 由于保存json的方式变了，为了兼容旧版，先尝试用新方法读取，如果出错就用旧方法读取
                self.checkpoint = json.load(open(checkpoint_filename, 'r'))
            except:
                self.checkpoint = torch.load(checkpoint_filename)
            self._load(version_name, model_dir, ignore, compatible_mode)

    def update(self, optimizer, loss, retain_graph=False):
        if self._training:
            optimizer.zero_grad()
            if torch.isnan(loss).item() == 0:
                loss.backward(retain_graph=retain_graph)
                optimizer.step()
                self._updated = True
            else:
                print('WARNING: the optimising tensor is nan')

    def clean_grad(self, optimizer):
        optimizer.zero_grad()

    def forward(self, data, forward_mode, **kwargs):
        assert False, 'The method "forward" must be defined'

    def _to_train(self):
        for model in self.models:
            self.__dict__[model].train()

    def _to_eval(self):
        for model in self.models:
            self.__dict__[model].eval()

    def _get_models(self):
        self.models = []
        for name, value in self.__dict__.items():
            if issubclass(type(value), nn.Module):
                self.models.append(name)

    def _fit(self, epochs, **kwargs):
        self._get_models()

        eval_per_step  = kwargs.get('eval_per_step')
        eval_per_epoch = kwargs.get('eval_per_epoch')
        save_per_step  = kwargs.get('save_per_step')
        save_per_epoch = kwargs.get('save_per_epoch')
        save_dir       = kwargs.get('save_dir')
        log            = kwargs.get('log')

        # 注册触发器
        controllers = self.controllerset.controllers.copy()
        events = []
        for controller in controllers:
            events.append(controller.event)

        if log is True:
            self.log = True
            self._logs.append(f"Training [{datetime.date.today()} {time.strftime('%H:%M:%S')}]")
            self._logs.append('')

        kwargs['mode'] = 'train'
        device         = kwargs.get('device')
        self.dataloader.load()
        current_epoch  = self.dataloader.get_epoch()

        for data_package in self.dataloader(epochs, **kwargs):
            time_start = time.time()

            triggered  = [False] * len(events)

            print_info = data_package['print_info']

            self._to_train()
            self._training = True
            self._updated  = False
            ret_train = self.forward(data_package['train_data'], 'train', **kwargs)
            if not self._updated:
                raise RuntimeWarning("权重未被更新，请在Trainer.forward的实现中使用Trainer.update来更新权重")

            data_package['ret_train'] = ret_train

            # 每forward一个step就触发一次所有触发器
            destroy_trigger = []
            for i, event in enumerate(events):
                if event(data_package):
                    triggered[i] = controllers[i]()
                    if not controllers[i].infinite_trigger and not controllers[i].enable:
                        destroy_trigger.append((event, controllers[i]))
            for e, c in destroy_trigger:
                events.remove(e)
                controllers.remove(c)

            with torch.no_grad():
                valid_data = data_package['valid_data']
                self._training = False
                self._to_eval()
                ret_valid = self.forward(valid_data, 'valid', **kwargs) if valid_data is not None else None
                self._to_train()
                self._training = True

            time_used = time.time() - time_start + data_package['time_used']
            yield ret_train, ret_valid, print_info + ' ' + str(time_used)[:5] + 's/it '

            if (eval_per_step is not None and (self.dataloader.get_step() + 1) % eval_per_step == 0) or \
               (eval_per_epoch is not None and current_epoch < self.dataloader.get_epoch() and (self.dataloader.get_epoch() + 1) % eval_per_epoch == 0):

                self.dataloader.index_test = 0
                eval_score = self.evaluate(**kwargs)

                if eval_score is not None:
                    # 每evaluate一次就触发一次所有触发器
                    data_package['eval_score'] = eval_score

                    destroy_trigger = []
                    for i, event in enumerate(events):
                        if triggered[i] == False and event(data_package):
                            triggered[i] = controllers[i]()
                            if not controllers[i].infinite_trigger and not controllers[i].enable:
                                destroy_trigger.append((event, controllers[i]))
                    for e, c in destroy_trigger:
                        events.remove(e)
                        controllers.remove(c)

                    if eval_score >= self.checkpoint['best_eval_score']:
                        self.checkpoint['best_eval_score'] = eval_score
                        __save = True
                    else:
                        __save = False
                    self.print_log(f'Evaluate Result: {eval_score}, Best: {self.checkpoint["best_eval_score"]}')
                    if __save:
                        self._save('model_best', save_dir, is_model_best=True, model_best_version_name=f'epoch:{self.dataloader.get_epoch()}|step:{self.dataloader.get_step() + 1}')

            if save_per_step is not None and (self.dataloader.get_step() + 1) % save_per_step == 0:
                self._save(f'step:{self.dataloader.get_step() + 1}', save_dir)

            if save_per_epoch is not None and current_epoch < self.dataloader.get_epoch() and (self.dataloader.get_epoch() + 1) % save_per_epoch == 0:
                # 每一个epoch就触发一次所有触发器
                destroy_trigger = []
                for i, event in enumerate(events):
                    if triggered[i] == False and event(data_package):
                        triggered[i] = controllers[i]()
                        if not controllers[i].infinite_trigger and not controllers[i].enable:
                            destroy_trigger.append((event, controllers[i]))
                for e, c in destroy_trigger:
                    events.remove(e)
                    controllers.remove(c)

                self._save(f'epoch:{self.dataloader.get_epoch()}', save_dir)

            current_epoch = self.dataloader.get_epoch()

        if save_per_epoch is None and save_per_step is None:
            self._save('model_epoch:%s' % self.dataloader.get_epoch(), save_dir)

        print('\n')

    def callback(self, forward_train, forward_valid, print_info):
        if forward_train is None:
            return
        train_info = 'train:'
        for name, value in list(forward_train.items()):
            train_info += '{name}: {value}, '.format(name=name, value=value)

        if forward_valid is not None and len(forward_valid) > 0:
            valid_info = 'valid:'
            for name, value in list(forward_valid.items()):
                valid_info += '{name}: {value}, '.format(name=name, value=value)
        else:
            valid_info = ''

        self.print_log(print_info + train_info + valid_info)

    def fit(self, epochs, save_dir, **kwargs):

        eps = kwargs.get('eval_per_step')
        epe = kwargs.get('eval_per_epoch')
        sps = kwargs.get('save_per_step')
        spe = kwargs.get('save_per_epoch')
        device = kwargs.get('device')

        if device is None:
            device = imath.default_device
            kwargs['device'] = device

        kwargs['save_dir'] = save_dir

        for forward_train, forward_valid, print_info in self._fit(epochs, **kwargs):
            self.callback(forward_train, forward_valid, print_info)

    def _evaluate(self, **kwargs):
        if hasattr(self, '_models') is False:
            self._get_models()

        self._training = False
        self._to_eval()
        kwargs['mode'] = 'test'
        device = kwargs.get('device')
        log = kwargs.get('log')
        if log is True:
            self.log = True
            self._logs.append(f"Evaluating [{datetime.date.today()} {time.strftime('%H:%M:%S')}]")
            self._logs.append('')

        if device is None:
            device = imath.default_device
            kwargs['device'] = device
        print('\n', end=' ')

        if kwargs.get('dataloader') is not None:
            dataloader = kwargs.get('dataloader')
        else:
            dataloader = self.dataloader

        # self.dataloader.index_test = 0
        dataloader.load()

        with torch.no_grad():
            for data_package in dataloader(**kwargs):
                ret_test = self.forward(data_package['test_data'], 'test', **kwargs)

                yield ret_test

                print_info = data_package['print_info']

                print(print_info, end=' ')

        self._to_train()
        self._training = True
        print('\r\n', end=' ')

    def evaluate(self, **kwargs):
        assert False, 'The method "evaluate" must be defined'

    def _save(self, version_name, save_dir, **kwargs):
        if os.path.exists(save_dir) is False:
            os.mkdir(save_dir)
        else:
            assert os.path.isdir(save_dir), 'save_dir must be a dir'

        save_dict = {}
        for name, value in self.__dict__.items():
            if self.save_ignore is not None and value in self.save_ignore:
                continue
            if is_saved_subclass(value):

                save_dict[name] = value.state_dict()

        for key, value in self.save().items():
            if self.save_ignore is not None and value in self.save_ignore:
                continue
            if key in save_dict.keys():
                print(f'WARNING: 变量"{key}"被覆盖保存')
            save_dict[key] = value

        self.checkpoint['last_version'] = version_name
        self.checkpoint['save_names'] = []
        if kwargs.get('is_model_best') is not None:
            self.checkpoint['model_best_version'] = kwargs.get('model_best_version_name')

        try:

            for key, value in list(save_dict.items()):
                # self.checkpoint['[{version_name}]{key}'.format(version_name=version_name, key=key)] = value
                self.checkpoint['save_names'].append(key)
                torch.save(value, os.path.join(save_dir, f'[{version_name}]{key}'), _use_new_zipfile_serialization=False)

            if version_name not in self.checkpoint['version_names']:
                self.checkpoint['version_names'].append(version_name)

            # torch.save(self.checkpoint, os.path.join(save_dir, 'checkpoint.json'))
            save_json = open(os.path.join(save_dir, 'checkpoint.json'), 'w')
            json.dump(self.checkpoint, save_json)
            save_json.close()

            if self.log is True:
                log_file = open(os.path.join(save_dir, 'log.txt'), 'a')
                for line in self._logs:
                    log_file.write(line.replace('\r', '').replace('\n', '') + '\n')
                log_file.close()
                self._logs = []
        except:
            print(('ERROR: save unsuccessful, %s' % save_dir))

    def save(self):
        return {}

    def _load(self, version_name, model_dir, ignore=None, compatible_mode=False):
        if True:
            if version_name is None or version_name is 'last_version':
                version_name = self.checkpoint['last_version']

            # print('version_name', version_name)
            version_dict = {key: torch.load(os.path.join(model_dir, f'[{version_name}]{key}'), map_location=im.default_device)
                            for key in self.checkpoint['save_names']}

            for name, value in version_dict.items():
                if hasattr(self, name):
                    var = self.__dict__[name]
                    if ignore is not None and var in ignore:
                        continue

                    if is_saved_subclass(var):

                        if compatible_mode and issubclass(type(var), nn.Module):
                            current_state = var.state_dict()
                            for k, v in value.items():
                                if k in current_state.keys():
                                    current_state[k] = v
                            value = current_state

                        var.load_state_dict(value)

            self.load(version_dict)
        else:
            print('WARNING: load model failed.')

    def _load_oldversion(self, version_name, ignore=None):
        print('This loading method will be removed in future version')
        if True:
            if version_name is None or version_name is 'last_version':
                version_name = self.checkpoint['last_version']

            version_dict = {key: self.checkpoint[f'[{version_name}]{key}']
                            for key in self.checkpoint['save_names']}

            for name, value in version_dict.items():
                if hasattr(self, name):
                    var = self.__dict__[name]
                    if ignore is not None and var in ignore:
                        continue
                    if is_saved_subclass(var):

                        var.load_state_dict(value)

            self.load(version_dict)
        else:
            print('WARNING: load model failed.')

    def load(self, version_dict):
        pass

    def print_log(self, info):
        # python2
        # print(info, end=' ')
        # python3

        print(info)
        if self.log == True:
            self._logs.append(info)




from .ConditionalAutoEncoderTrainer import *
from .AutoEncoderTrainer import *
from .VAETrainer import *
from .GenerativeAdversarialTrainer import *
from .ClassifyTrainer import *
