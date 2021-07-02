import imath as pt
import numpy as np


def GetRandomSphereVector(batch=1):
    u = np.random.uniform(0, 1, size=(batch,))
    v = np.random.uniform(0, 1, size=(batch,))
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    x = np.sin(theta) * np.sin(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.cos(phi)

    return im.Tensor(np.concatenate([x.reshape(batch, 1), y.reshape(batch, 1), z.reshape(batch, 1)], axis=1)).float()
