from .file_op import *


def MergeListFile(output_filename, merge_function, data_roots, list_filenames, **kwargs):
    '''
    合并多个文件, 一般用于合并多个不同文件夹下的训练集的txt
    :param output_filename: 输出文件名
    :param merge_function: 融合函数, merge_function(data_root, line, **kwargs) -> str or None, None则不写入
    :param data_roots:
    :param list_filenames:
    :param kwargs:
    :return:
    '''
    file = open(output_filename, 'w')
    for data_root, list_filename in zip(data_roots, list_filenames):
        for line in open(list_filename, 'r').readlines():
            new_line = merge_function(data_root, line, **kwargs)
            if new_line is not None:
                file.write(new_line.replace('\n', '') + '\n')
    file.close()


def mkdir(dir):
    os.system("mkdir -p " + dir)