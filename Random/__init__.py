import torch as P


def shuffle(tensor, dim, seed=None, inplace=False):
    if seed is not None:
        P.manual_seed(seed)
    permutation = P.randperm(tensor.size(dim)).to(tensor.device)

    if inplace:
        P.index_select(tensor, dim, permutation, out=tensor)
    else:
        return P.index_select(tensor, dim, permutation)

def SynchronousShuffle(tensors, dim, seed=None, inplace=False):
    if seed is not None:
        P.manual_seed(seed)
    permutation = P.randperm(tensors[0].size(dim))

    if inplace:
        for tensor in tensors:
            P.index_select(tensor, dim, permutation, out=tensor)
    else:
        return [P.index_select(tensor, dim, permutation) for tensor in tensors]
