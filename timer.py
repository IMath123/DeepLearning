import time

SHOW = True

class Timer(object):

    def __init__(self, name):
        self.name = name

    def __enter__(self, name=None):

        self.time_start = time.time()
        return self

    def __exit__(self, a, b, c):
        self.time_end = time.time()
        if SHOW:
            if self.name is not None:
                print(("Timer %s's result: %s s" % (self.name, self.time_end - self.time_start)))
            else:
                print(("Timer's result: %s s" % (self.time_end - self.time_start)))

def show(show):
    global SHOW

    if type(show) == bool:
        SHOW = show
    else:
        assert False, 'Please put in True or False'
