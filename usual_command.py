import cv2 as cv
from imath import backend as P
from matplotlib import pyplot as plt
import numpy as np
import imath
import os

#TODO : 完成此表
numpytype2torchtype = {
    np.float32: P.float32,
    np.float64: P.float64,
    np.int8: P.int8,
    np.int64: P.int64
}
torchtype2numpytype = {value: key for key, value in numpytype2torchtype.items()}


def imread(filename, flag=None, channel_first=True, dtype=P.float32, type='tensor'):
    img = cv.imread(filename, flags=flag)
    if channel_first and flag is not 0:
        img = img.transpose(2, 0, 1)
        # img = img[:channels]
    else:
        pass
        # img = img[..., :channels]

    if type is 'tensor':
        ret = P.from_numpy(img).type(dtype)
    elif type is 'array' or 'numpy':
        ret = img.astype(torchtype2numpytype[dtype])

    return ret


def imshow(*images, **kwargs):

    show_image = np.concatenate([imath.Array(img) if isinstance(img, str) == False  else imath.imread(img)/255 for img in images], axis=-1)
    if kwargs.get('no_show') is True:
        return show_image
    mode = kwargs.get('mode')
    show = kwargs.get('show')
    if show is False:
        return show_image

    elif show is True or show is None:
        if len(show_image) == 3:
            show_image = show_image.transpose((1, 2, 0))
        if mode is None:
            mode = 'plt'
        if mode is 'plt':
            plt.imshow(show_image)
            plt.show()
        elif mode is 'cv':
            cv.imshow('imath', show_image)
            cv.waitKey()

        return show_image

def imsave(filename, obj, mode='cv'):

    img = imath.Array(obj)

    if mode == 'cv':
        if len(img.shape) == 3:
            img = img.transpose(1, 2, 0)
        if img.dtype != np.uint8:
            img = np.clip(img * 256, 0, 255).astype(np.uint8)
        cv.imwrite(filename, img)
    elif mode == 'plt':
        plt.imsave(filename, img)
    else:
        assert False, 'mode must be cv or plt'

def savetensor(filename, obj):
    array = imath.Array(obj)
    np.save(filename, array)

def loadtensor(filename):
    array = np.load(filename)
    return imath.Tensor(array)


def checkdir(path, mode=''):
    if os.path.exists(path) == False:
        if mode is '':
            os.mkdir(path)
        elif mode == '-p':
            os.system(f'mkdir -p {path}')
        else:
            assert False

def checkfiledir(filename, mode='-p'):
    checkdir(os.path.split(filename)[0], mode=mode)

def loadtxt(filename, dtype=P.float32, length=None):
    if length is None:
        txt = np.loadtxt(filename)
        return P.from_numpy(txt).type(dtype)
    else:
        lines = open(filename, 'r').readlines()
        output = np.empty((len(lines), length))
        for n, line in enumerate(lines):
            output[n] = np.array(line.split())
        return P.from_numpy(output).type(dtype)


def savetxt(filename, obj):
    array = obj.cpu().detach().numpy()
    np.savetxt(filename, array)


def __GetAllFile(root, postfix='.txt', result=[]):
    if os.path.isdir(root):
        for line in os.listdir(root):
            __GetAllFile(os.path.join(root, line), postfix, result)
    else:
        if root[-len(postfix):] == postfix:
            result.append(root)

def GetAllFiles(root, postfix='.txt'):
    ret = []
    __GetAllFile(root, postfix, ret)
    return ret


def ListWrite(filename, lines):
    with open(filename, 'w') as file:
        for line in lines:
            file.write(line.replace('\n', '') + '\n')





def Face_Align_to_StandardFace(pointcloud, keypoints, valid_index=None, return_RT=False):
    '''

    :param pointcloud: N * 3
    :param keypoints: 5 * 3
    :return: aligned pointcloud
    '''
    dirname = os.path.dirname(imath.__file__)
    standardface = loadtxt(os.path.join(dirname, 'standard_keypoint.txt'))
    if valid_index is not None:
        keypoints = keypoints[valid_index]
        standardface = standardface[valid_index]
    R, T = ICP_Transorm_Matrix(keypoints, standardface)

    if return_RT:
        return R, T

    return P.matmul(pointcloud, R) + T


def Crop_PointCloud(pointcloud, range, return_area=False):
    area = (pointcloud[:, 0] > range[0]) * (pointcloud[:, 0] < range[1]) * \
           (pointcloud[:, 1] > range[2]) * (pointcloud[:, 1] < range[3]) * \
           (pointcloud[:, 2] > range[4]) * (pointcloud[:, 2] < range[5])

    if return_area:
        return area
    else:
        return pointcloud[area]


