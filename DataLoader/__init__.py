import torch
from torch.utils.data import Dataset as torch_Dataset
from torch.utils.data import Sampler
from tqdm import tqdm
import re
from ..ProgressBar import SimpleBar

def Bar(x):
    return SimpleBar(x)


def epo_epos(e, es):
    e = min([e + 1, es])
    return f'{e} / {es}'


def to_device(tensor, device):
    return tensor.to(device)

def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)
    random.seed(seed)


class DataSet(torch_Dataset):

    def __init__(self, data_root, lines, dataloader):
        self.data_root = data_root
        self.lines = lines
        self.get_data_func = dataloader._get_data

    def __getitem__(self, item):
        line = self.lines[item]
        # print(line)
        return self.get_data_func(self.data_root, line)

    def __len__(self):
        return len(self.lines)


class InitializeSample(Sampler):

    def __init__(self, start_index, data_score):
        '''
        用于断点恢复，首次迭代将在设定的点开始，随后的迭代都是从头开始
        '''
        self.start_index = start_index
        self.data_len = len(data_score)

    def __iter__(self):
        ret = iter(range(self.start_index, self.data_len))
        self.start_index = 0
        return ret

    def __len__(self):
        return self.data_len - self.start_index


class DataLoader(object):

    dtype2str = {torch.int:     'int',
                 torch.int32:   'int32',
                 torch.int8:    'int8',
                 torch.int16:   'int16',
                 torch.int64:   'int64',
                 torch.long:    'long',
                 torch.short:   'short',
                 torch.float:   'float',
                 torch.float32: 'float32',
                 torch.float64: 'float64',
                 torch.float16: 'float16',
                 torch.double:  'double',
                 torch.uint8:   'uint8',
                 }
    str2dtype = {value: key for key, value in dtype2str.items()}

    def check(self, data_root=None, list_filename=None, replace=False, **kwargs):
        '''
        此工具用于检查训练集中的错误数据，分别返回成功和失败的行，可用imath.ListWrite直接写到新的txt去
        :param data_root:
        :param list_filename:
        :param replace:
        :param kwargs:
        :return:
        '''

        if data_root is None and list_filename is None:
            data_root = self._data_root
            list_filename = self._list_filename

        succeed = []
        fail = []
        for line in tqdm(open(list_filename, 'r').readlines()):
            try:
                _ = self._get_data(data_root, line, **kwargs)
                succeed.append(line)
            except:
                fail.append(line)

        if replace:
            file = open(list_filename, 'w')
            for line in succeed:
                file.write(line)
            file.close()

        return succeed, fail

    def get_info(self):
        Map = {}
        Map['list_filename'] = self._list_filename
        Map['data_root'] = self._data_root
        Map['test_ratio'] = self._test_ratio
        Map['test_data_root'] = self._test_data_root
        Map['test_list_filename'] = self._test_list_filename
        Map['valid_ratio'] = self._valid_ratio
        Map['valid_data_root'] = self._valid_data_root
        Map['valid_list_filename'] = self._valid_list_filename

        return Map

    def save_info(self, save_name):

        with open(save_name, 'w') as file:
            json.dump(self.get_info(), file)

    def load_info(self, info):

        if isinstance(info, str):
            with open(info, 'r') as file:
                info_map = json.load(file)
                info_map['info_path'] = os.path.dirname(os.path.abspath(info))

            return self.load_info(info_map)
        else:
            cwd = info.get('info_path')

            def smart_load(inp):
                if inp is None:
                    return None
                elif isinstance(inp, str):
                    if cwd is None:
                        return inp
                    else:
                        if os.path.isabs(inp):
                            return inp
                        else:
                            return os.path.join(cwd, inp)
                else:
                    return inp

            self._list_filename            = smart_load(info.get('list_filename'))
            self._data_root                = smart_load(info.get('data_root'))
            self._test_ratio               = smart_load(info.get('test_ratio'))
            self._test_data_root           = smart_load(info.get('test_data_root'))
            self._test_list_filename       = smart_load(info.get('test_list_filename'))
            self._valid_ratio         = smart_load(info.get('valid_ratio'))
            self._valid_data_root     = smart_load(info.get('valid_data_root'))
            self._valid_list_filename = smart_load(info.get('valid_list_filename'))

    def __init__(self, **kwargs):
        '''
        必选参数:
            data_root                                          = str: 数据根目录
            list_filename                                      =
        可选参数:
            test_ratio/(test_data_root, test_list_filename)    =
            valid_ratio/(valid_data_root, valid_list_filename) =
            shuffle                                            =
            random_seed                                        =
            num_workers                                        =
            test_num_workers                                   =
            valid_num_workers                                  =
            pin_memory                                         =
            collate_fn_map                                     = dict: 储存数据类型以及对应的收集函数
            collate_err_msg                                    = str : 收集数据失败时打印的信息模板
        '''

        if len(kwargs) == 0:
            return

        kwargs.get('data_root')
        kwargs.get('list_filename')
        kwargs.get('test_ratio')
        kwargs.get('test_data_root')
        kwargs.get('test_list_filename')
        kwargs.get('valid_ratio')
        kwargs.get('valid_data_root')
        kwargs.get('valid_list_filename')
        kwargs.get('collate_fn_map')

        self.load_info(kwargs)

        self._shuffle          = kwargs.get('shuffle')
        self._random_seed      = kwargs.get('random_seed')

        self.num_workers       = kwargs.get('num_workers')
        self.test_num_workers  = kwargs.get('test_num_workers')
        self.valid_num_workers = kwargs.get('valid_num_workers')
        self.pin_memory        = kwargs.get('pin_memory')

        self.index             = kwargs.get('index')
        self.index_valid       = kwargs.get('index_valid')
        self.epoch             = kwargs.get('epoch')
        self.step              = kwargs.get('step')
        self.batch_size        = kwargs.get('batch_size')
        self.test_batch_size   = kwargs.get('test_batch_size')
        self.valid_batch_size  = kwargs.get('valid_batch_size')

        self.collate_fn_map    = kwargs.get('collate_fn_map')
        self.collate_err_msg   = kwargs.get('collate_err_msg_format')

        if self.index is None:
            self.index             = 0
        if self.index_valid is None:
            self.index_valid       = 0
        if self.epoch is None:
            self.epoch             = 0
        if self.step is None:
            self.step              = 0
        if self._data_root is None:
            self._data_root        = ''
        if self._random_seed is None:
            self._random_seed      = 0
        if self._shuffle is None:
            self._shuffle          = False
        if self.num_workers is None:
            self.num_workers       = 0
        if self.test_num_workers is None:
            self.test_num_workers  = self.num_workers
        if self.valid_num_workers is None:
            self.valid_num_workers = 0
        if self.pin_memory is None:
            self.pin_memory        = False
        if self.test_batch_size is None:
            self.test_batch_size   = self.batch_size
        if self.valid_batch_size is None:
            self.valid_batch_size  = self.batch_size
        if self.collate_err_msg is None:
            self.collate_err_msg   = default_collate_err_msg_format
        if self.collate_fn_map is None:
            self.collate_fn_map    = {}

        #  self.collate_fn        = get_collate_fn(self.collate_fn_map, self.collate_err_msg)
        self.collate_fn = None
        self.loaded = False

    def load(self):

        if self.loaded:
            return

        if self._list_filename is not None:
            self._LINES = self._get_lines(self._list_filename, self._shuffle)
        else:
            self._LINES = []

        # 测试集或者验证集只能从整个数据集中按比例取或者另外设置的数据集中取
        assert self._test_ratio is None or self._test_data_root is None, 'you can only choose at most one param ' \
                                                                         'in test_ratio and test_data.'
        assert self._valid_ratio is None or self._valid_data_root is None, 'you can only choose at most one param ' \
                                                                                     'in valid_ratio and valid_data.'

        test_cut_index = int(
            len(self._LINES) * self._test_ratio) if self._test_ratio is not None and self._test_ratio > 0 else 0
        valid_cut_index = -int(len(
            self._LINES) * self._valid_ratio) if self._valid_ratio is not None and self._valid_ratio > 0 else len(
            self._LINES)

        self._Train = self._LINES[test_cut_index: valid_cut_index]
        self._Test  = self._LINES[:test_cut_index]
        self._Valid = self._LINES[valid_cut_index:]

        if self._test_data_root is not None:
            # 读取测试集时并不会打乱顺序
            self._Test                = self._get_lines(self._test_list_filename, False)
            temp_test_data_root       = self._test_data_root
        else:
            temp_test_data_root       = self._data_root
        if self._valid_data_root is not None:
            self._Valid               = self._get_lines(self._valid_list_filename, self._shuffle)
            temp_valid_data_root = self._valid_data_root
        else:
            temp_valid_data_root = self._data_root

        self.running    = False
        self.index_test = 0

        train_sampler   = InitializeSample(self.index, self._Train)
        valid_sample    = InitializeSample(self.index_valid, self._Valid)


        self.train_loader = torch.utils.data.DataLoader(DataSet(self._data_root, self._Train, self),
                                                        batch_size     = self.batch_size,
                                                        num_workers    = self.num_workers,
                                                        pin_memory     = self.pin_memory,
                                                        sampler        = train_sampler,
                                                        worker_init_fn = worker_init_fn_seed,)
        self.test_loader  = torch.utils.data.DataLoader(DataSet(temp_test_data_root, self._Test, self),
                                                        batch_size     = self.test_batch_size,
                                                        num_workers    = self.test_num_workers,
                                                        pin_memory     = self.pin_memory,
                                                        worker_init_fn = worker_init_fn_seed,)
        self.valid_loader = torch.utils.data.DataLoader(DataSet(temp_valid_data_root, self._Valid, self),
                                                        batch_size     = self.valid_batch_size,
                                                        num_workers    = self.valid_num_workers,
                                                        pin_memory     = self.pin_memory,
                                                        sampler        = valid_sample,
                                                        worker_init_fn = worker_init_fn_seed,)

        self.mode   = 'train'
        self.loaded = True

        # warning
        if self._shuffle is False:
            print(f"warning: DataLoader {self.name()}'s shuffle is False")

    def name(self):
        return self.__class__.__name__

    def state_dict(self):
        state_dict = {
            'name':                     str(self.__class__.__name__),
            # 'inputs': DataLoader.save_inputs(self._inputs),
            'list_filename':            self._list_filename,
            'data_root':                self._data_root,
            'shuffle':                  self._shuffle,

            'num_workers':              self.num_workers,
            'test_num_workers':         self.test_num_workers,
            'valid_num_workers':        self.valid_num_workers,
            'pin_memory':               self.pin_memory,

            'random_seed':              self._random_seed,
            'test_ratio':               self._test_ratio,
            'test_data_root':           self._test_data_root,
            'test_list_filename':       self._test_list_filename,
            'valid_ratio':         self._valid_ratio,
            'valid_data_root':     self._valid_data_root,
            'valid_list_filename': self._valid_list_filename,
            # =====================================================================
            'index':                    self.index,
            'index_valid':              self.index_valid,
            'epoch':                    self.get_epoch(),
            'step':                     self.get_step(),
            'batch_size':               self.batch_size,
            'test_batch_size':          self.test_batch_size,
            'valid_batch_size':         self.valid_batch_size
        }
        return state_dict

    def load_state_dict(self, state_dict):
        if state_dict['name'] != self.name():
            print(f"WARNING: you are loading a state dict with name {state_dict['name']}, but this dataloader's name "
                  f"is {self.name()}")
        super(self.__class__, self).__init__(**state_dict)

    def _get_lines(self, list_filename, shuffle):

        lines = np.array(list(open(list_filename, 'r').readlines()), dtype=str)

        if shuffle:
            random.seed(self._random_seed)
            random.shuffle(lines)

        return lines

    def _summary(self):
        if not self.loaded:
            self.load()
        info = '====================================================================================\n' \
               'General Infomation\n' \
               'data_root: {data_root},\n' \
               'list_filename: {list_filename},\n' \
               'shuffle: {shuffle},\n' \
               'Train_Data: {train_len},\n' \
               'Test_Data: {test_len},\n' \
               'Valid_Data: {valid_len}\n' \
               '===================================================================================='
        info = info.format(data_root=self._data_root, list_filename=self._list_filename,
                           shuffle=self._shuffle, train_len=len(self._Train), test_len=len(self._Test),
                           valid_len=len(self._Valid))
        return info

    def summary(self):
        return self._summary()

    def __str__(self):
        return self.summary()

    def _get_data(self, data_root, line, **kwargs):
        return self.get_data(data_root, line, **kwargs)

    def get_data(self, data_root, line, **kwargs):
        assert False, 'method "get_data" must be defined'

    def augmentation(self, data_package, data_mode, **kwargs):
        return data_package

    def _mode_2_data_root(self, mode):
        if mode == 'train':
            return self._data_root
        elif mode == 'test':
            return self._test_data_root if self._test_data_root is not None else self._data_root
        elif mode == 'valid':
            return self._valid_data_root if self._valid_data_root is not None else self._data_root
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
            self.step  += 1
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

    def get_epoch(self):
        return self.epoch

    def get_step(self):
        return self.step

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
                self._index_add(data_mode, len(ret[0]))
                return ret
            except:
                assert False, ''

        self._index_add(data_mode, len(ret[0]))
        return ret


    def __call__(self, epochs=1, **kwargs):

        self.load()

        mode   = kwargs.get('mode')

        device = kwargs.get('device')
        if device is None:
            device = im.default_device

        if mode              == 'train':
            data_name         = 'train_data'
            data_mode         = 'train'

        elif mode            == 'test':
            self.index_test   = 0
            data_name         = 'test_data'
            data_mode         = 'test'

        elif mode            == 'valid':
            self.index_valid  = 0
            data_name         = 'valid_data'
            data_mode         = 'valid'
        elif mode            is None:
            data_name         = 'data'
            data_mode         = 'train'
            mode              = 'train'

        else:
            assert False, 'mode must be train or test'

        len_of_lines = self._mode_2_len(mode)
        # loader = self._mode_2_loader(mode)

        while True:

            package = {'valid_data': None, 'epoch': self.get_epoch(), 'step': self.get_step()}

            if mode is 'train' and self.epoch + 1 > epochs:
                break

            time_start = time.time()
            train_data = [value.to(device) if type(value) is torch.Tensor else value for value in
                          self._get_batch(data_mode)]

            if mode is 'train' and len(self._Valid) > 0:
                valid_data = [value.to(device) if type(value) is torch.Tensor else value for value in
                              self._get_batch('valid')]
                package['valid_data'] = valid_data

            index = self._mode_2_index(mode)

            progress = float(index + 1) / len_of_lines if index != 0 else 1.0

            package[data_name] = train_data

            package['index_train'] = self.index

            if mode == 'train':
                package['print_info'] = f'\r[Epoch {epo_epos(self.epoch, epochs)}][{Bar(progress)}]'
            else:
                package['print_info'] = f'\r[Evaluating][{Bar(progress)}]'

            time_used = time.time() - time_start
            package['time_used'] = time_used

            yield package

            if (mode is 'test' or mode is 'valid') and index == 0:
                break

    def _make_hdf5_process(self, hdf5_file, mode):
        len_of_lines = self._mode_2_len(mode)
        data_set = hdf5_file.create_group(mode)
        if mode == 'train':
            batch_size = self.batch_size
        elif mode == 'test':
            batch_size = self.test_batch_size
        elif mode == 'valid':
            batch_size = self.valid_batch_size
        else:
            assert False

        data_set['batch_size'] = batch_size

        variable_num = None
        i = 0

        while True:
            data = [value.numpy() for value in self._get_batch(mode)]

            if variable_num is None:
                variable_num = len(data)
                data_set['variable_num'] = variable_num

                for n, variable in enumerate(data):
                    data_set.create_dataset(str(n), shape=(self._mode_2_len(mode), *variable.shape[1:]), maxshape=(None, *variable.shape[1:]), chunks=(batch_size, *variable.shape[1:]), dtype=variable.dtype)
                    # data_set.create_dataset(str(n), shape=(self._mode_2_len(mode), *variable.shape[1:]), maxshape=(None, *variable.shape[1:]), chunks=(1, *variable.shape[1:]), dtype=variable.dtype)

            for n, variable in enumerate(data):
                batch = variable.shape[0]
                data_set[str(n)][i: i + batch] = variable

            i += batch

            index = self._mode_2_index(mode)
            progress = float(index + 1) / len_of_lines if index != 0 else 1.0
            progress = f'\r[making {mode} data][{Bar(progress)}] '

            yield progress

            if index == 0:
                if mode == 'train':
                    self.epoch -= 1
                break

    def make_hdf5(self, hdf5_filename, **kwargs):
        import h5py
        if os.path.exists(hdf5_filename):
            assert False, f'ERROE: file {hdf5_filename} exists.'

        self.load()

        hdf5_file = h5py.File(hdf5_filename, 'w')

        for mode in ['train', 'valid', 'test']:
            for progress in self._make_hdf5_process(hdf5_file, mode):
                print(progress, end='\r')

        hdf5_file.close()


from .ImageLoader import *
from .MultiDataLoader import *
from .MixtureDataLoader import *
from .TensorLoader import *
from .Hdf5Loader import *

def LambdaDataLoader(get_data_func):
    class loader(DataLoader):

        def __init__(self, **kwargs):
            super(loader, self).__init__(**kwargs)

        def get_data(self, data_root, line, **kwargs):
            return get_data_func(data_root, line, **kwargs)

    return loader
