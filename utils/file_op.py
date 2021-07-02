import os
from tqdm import tqdm

def GetListForClassify(data_root, output_filename):
    '''
    根目录下的每个文件夹代表一种类别
    :param data_root:
    :return:
    '''
    file = open(output_filename, 'w')
    i = 0
    for cla in os.listdir(data_root):
        if os.path.isdir(os.path.join(data_root, cla)):
            for dir, _, names in os.walk(os.path.join(data_root, cla)):
                for name in names:
                    filename = os.path.join(dir, name)

                    file.write(f'{filename} {i}\n')
            i += 1

    file.close()


def GenerateFRPair(list_filename, output_filename):
    id2files = dict()
    files2index = dict()
    for n, line in enumerate(open(list_filename, 'r').readlines()):
        name, id = line.split()
        if id not in id2files.keys():
            id2files[id] = []

        id2files[id].append(name)
        files2index[name] = n

    keys = list(id2files.keys())
    output = open(output_filename, 'w')

    for i in tqdm(range(len(keys))):
        for j in range(i, len(keys)):
            list_i = id2files[keys[i]]
            list_j = id2files[keys[j]]

            if i == j:
                for u in range(len(list_i)):
                    for v in range(u+1, len(list_i)):
                        output.write(f'{files2index[list_i[u]]} {files2index[list_j[v]]} 1\n')
            else:
                for u in range(len(list_i)):
                    for v in range(len(list_j)):
                        output.write(f'{files2index[list_i[u]]} {files2index[list_j[v]]} 0\n')

    output.close()
