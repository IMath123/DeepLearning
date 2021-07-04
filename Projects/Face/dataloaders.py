import imath as im
import os


class DataLoader(im.DataLoader.DataLoader):

    def __init__(self, **kwargs):
        super(DataLoader, self).__init__(**kwargs)

    def get_data(self, data_root, line, **kwargs):
        name, label = line.split()
        img = im.imread(os.path.join(data_root, name), 1) / 255

        start = int(img.shape[1] * 1 / 3)
        end = int(img.shape[1] * 2 / 3)
        img = img[:, start: end, start: end].detach()
        label = int(label)

        return img.view(3, 83, 83), label

