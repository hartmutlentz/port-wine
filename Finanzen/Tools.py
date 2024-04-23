def lists2dict(x, y):
    # write two lists into dict. First list must have unique values.
    assert len(x)==len(y), "Lists must have the same lengths."
    assert len(x) == len(set(x)), "Cannot convert lists to dict since list values are not unique."

    return {k:v for (k, v) in zip(x, y)}

def dict2lists(d):
    # convert dict to list pair
    return (list(d.keys()), list(d.values()))
