def parse_idx(x):
    if isinstance(x, slice):
        s = 0 is x.start is None else x.start
        e = x.stop
        if x.step != None:
            raise Warning("Step is not being consider for now.")
        if s >= e:
            return e, s
        else:
            return s, e  
    elif isinstance(x, int):
        return x, x+1
    elif isinstance(x, list):
        return min(x), max(x)+1
    else:
        raise Warning("General lists not considered for now.")
