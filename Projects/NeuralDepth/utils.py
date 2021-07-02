import numpy as np
import os
import struct
import time


def load_obj(filename):
    v = []
    vn = []
    f = []
    fn = []

    for line in open(filename, 'r').readlines():
        if line[0] == '#':
            continue

        split = line.split()
        if split[0] == 'v':
            v.extend(map(float, split[1:4]))
        elif split[0] == 'vn':
            vn.extend(map(float, split[1:4]))
        elif split[0] == 'f':
            for s in split[1: 4]:
                v_i, vt_i, vn_i = s.split('/')
                if len(v_i) != 0:
                    f.append(v_i)
                if len(vn_i) != 0:
                    fn.append(vn_i)

        else:
            continue

    v = np.array(v, dtype=np.float32).reshape(-1, 3)
    vn = np.array(vn, dtype=np.float32).reshape(-1, 3)
    f = np.array(f, dtype=np.int32).reshape(-1, 3)
    fn = np.array(fn, dtype=np.int32).reshape(-1, 3)

    return v, vn, f, fn

def save_obj_to_bin(src_filename, dst_filename):
    v, vn, f, fn = load_obj(src_filename)
    bin = open(dst_filename, 'wb')
    bin.write(struct.pack('ii', len(v), len(f)))
    #  bin.write(struct.pack('i', len(f)))
    for arr, typ in zip([v, vn, f, fn], ['fff', 'fff', 'III', 'III']):
        print(arr.shape, typ)
        for x, y, z in arr:
            bin.write(struct.pack(typ, x, y, z))

    bin.close()
    print(f"byte = {2 * 4 + 2 * (len(v) * 4 * 3 + len(f) * 4 * 3)}")

if __name__ == "__main__":
    obj_filename = '/home/dj/Downloads/1a1ec1cfe633adcdebbf11b1629fc16a.obj'
    t1 = time.time()
    save_obj_to_bin(obj_filename, '/home/dj/anaconda3/lib/python3.8/site-packages/imath/Projects/NeuralDepth/temp.bin')
    t2 = time.time()
    print(t2 - t1)
