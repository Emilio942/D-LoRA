import torch


def neuron_cosine_matrix(h):
    h = h - h.mean(dim=0, keepdim=True)
    h = h / (h.norm(dim=0, keepdim=True) + 1e-8)
    return h.T @ h


def conflict_loss(h):
    C = neuron_cosine_matrix(h)
    off = C - torch.eye(C.shape[0])
    n = C.shape[0]
    return (off ** 2).sum() / (n * (n - 1))


def conflict_score(h, i=0, j=1):
    return neuron_cosine_matrix(h)[i, j].item()
