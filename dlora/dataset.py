import torch

CLASS_NAMES = ["s=0,a=0", "s=0,a=1", "s=1,a=0", "s=1,a=1"]


def make_xor_dataset(n_per_class=256, noise=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    combos = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    xs, ys = [], []
    for y, (s, a) in enumerate(combos):
        base = torch.stack(
            [torch.full((n_per_class,), s), torch.full((n_per_class,), a)], dim=1
        )
        xs.append(base + noise * torch.randn(base.shape, generator=g))
        ys.append(torch.full((n_per_class,), y, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


def train_test_split(x, y, test_frac=0.25, seed=1):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(x), generator=g)
    n_test = int(len(x) * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    return x[tr], y[tr], x[te], y[te]
