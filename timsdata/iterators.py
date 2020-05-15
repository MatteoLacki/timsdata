class ComfyIter(object):
    """A simple trick to use D.iter[1:10, 50:100:2] like expressions.

    Simply, add self.iter = ComfyIter(self) and define in that class the __iter method that can ise these slices and voila.
    """
    def __init__(self, D):
        self.D = D

    def __getitem__(self, x):
        return self.D._iter(x)


def infinite_range(start, step):
    i = start
    while True:
        yield i
        i += step