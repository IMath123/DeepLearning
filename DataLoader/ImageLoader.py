from imath.DataLoader import DataLoader
import torch
import os
import imath as pt

class ImageLoader(DataLoader):

    def __init__(self, get_path_label, **kwargs):
        super(ImageLoader, self).__init__(**kwargs)

        self.dtype = kwargs.get('dtype')
        self.get_path_label = get_path_label
        self.image_size = kwargs.get('image_size')
        # self.dtypes = [self.dtype, torch.long]
        # self.data_shapes = [self.image_size, None]
        self._inputs = {'x': (self.dtype, self.image_size),
                        'y': (torch.long, None)}
        self.Aug_Function = kwargs.get('Aug_Function')

        # self.train_labels = self.get_labels(self._Train)
        # self.test_labels = self.get_labels(self._Test)
        # self.valid_labels = self.get_labels(self._Valid)

    def get_labels(self, LINES):
        Maps = {}
        for line in LINES:
            path, label = line.split()
            if label not in list(Maps.keys()):
                Maps[label] = 0
            Maps[label] += 1
        return Maps

    def _get_data(self, data_root, line, **kwargs):
        # print('reading')

        try:
            path, label = self.get_path_label(line)
            path = os.path.join(data_root, path)
            img = im.imread(path)
            if self.Aug_Function is not None:

                img = self.Aug_Function(img, label=label, **kwargs)

        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            # print(('Read Image Error:', path))
            return None

        return {'x': img, 'y': int(label)}


    def summary(self):

        info = self._summary()
        info += '\n' \
               'dtype: {dtype},\n' \
               'image_size: {image_size}\n'.format(dtype=self.dtype, image_size=self.image_size)

        # info += str(self.train_labels) if len(self.train_labels) > 0 else ''
        # info += str(self.test_labels) if len(self.test_labels) > 0 else ''
        # info += str(self.valid_labels) if len(self.valid_labels) > 0 else ''

        return info