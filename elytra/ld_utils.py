import torch


def unsort(L, sort_idx):
    assert isinstance(sort_idx, list)
    assert isinstance(L, list)
    LL = zip(sort_idx, L)
    LL = sorted(LL, key=lambda x: x[0])
    _, L = zip(*LL)
    return list(L)


def add_prefix_postfix(mydict, prefix="", postfix=""):
    assert isinstance(mydict, dict)
    return dict(
            (prefix + key + postfix, value)
            for (key, value) in mydict.items())


def ld2dl(LD):
    assert isinstance(LD, list)
    assert isinstance(LD[0], dict)
    """
    A list of dict (same keys) to a dict of lists
    """
    dict_list = {k: [dic[k] for dic in LD] for k in LD[0]}
    return dict_list


class NameSpace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def dict2ns(mydict):
    """
    Convert dict objec to namespace
    """
    return NameSpace(mydict)


def ld2dev(ld, dev):
    """
    Convert tensors in a list or dict to a device recursively
    """
    if isinstance(ld, torch.Tensor):
        return ld.to(dev)
    if isinstance(ld, dict):
        for k, v in ld.items():
            ld[k] = ld2dev(v, dev)
        return ld
    if isinstance(ld, list):
        return [ld2dev(x, dev) for x in ld]
    return ld
