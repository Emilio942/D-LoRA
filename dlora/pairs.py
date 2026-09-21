import math

import torch


def sample_pairs(n, d, cos_target, generator, norm_range=(0.5, 2.0)):
    assert d >= 2
    raw1 = torch.randn(n, d, generator=generator)
    raw_u = torch.randn(n, d, generator=generator)
    proj = (raw_u * raw1).sum(1, keepdim=True) / raw1.pow(2).sum(1, keepdim=True).clamp_min(1e-12)
    u = raw_u - proj * raw1
    u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n1 = raw1 / raw1.norm(dim=1, keepdim=True).clamp_min(1e-12)
    lo, hi = norm_range
    s1 = lo + (hi - lo) * torch.rand(n, 1, generator=generator)
    s2 = lo + (hi - lo) * torch.rand(n, 1, generator=generator)
    sin_t = math.sqrt(max(0.0, 1.0 - cos_target * cos_target))
    x1 = s1 * n1
    x2 = s2 * (cos_target * n1 + sin_t * u)
    c = torch.full((n,), float(cos_target))
    return x1, x2, c


def sample_pairs_1d(n, b, generator, a_range=(0.8, 1.2)):
    a = a_range[0] + (a_range[1] - a_range[0]) * torch.rand(n, 1, generator=generator)
    x1 = a
    x2 = b * a
    c = torch.full((n,), float(b))
    return x1, x2, c
