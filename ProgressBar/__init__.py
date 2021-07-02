import time


def SimpleBar(progress, long=20, char_done='=', char_undone='.', arrow='>', **kwargs):
    """
    a simple bar
    """

    num = int(progress * long)
    bar = char_done * (num - 1) + arrow + char_undone * (long - num) + '({percent}%)'.format(percent=str(progress * 100.)[:5])

    return bar


if __name__ == '__main__':
    print((SimpleBar(0, show_percent=True)))
