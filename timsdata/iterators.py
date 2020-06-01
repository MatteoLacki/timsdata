from functools import update_wrapper


class ComfyIter(object):
    """Using ComfyIter enables parsing of range exressions.
    This way, you might do selection with square brackets, e.g. D.iter[1:10, 50:100:2]
    """
    def __init__(self, iterator):
        self.iterator = iterator
        self = update_wrapper(self, iterator)

    def __getitem__(self, x):
        return self.iterator(x)


def infinite_range(start, step):
    i = start
    while True:
        yield i
        i += step