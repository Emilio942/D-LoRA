import math

import torch


def unit_rows(X):
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)


def sample_scaled_gauss(n, d, g, lo=1.0, hi=1.0):
    X = torch.randn(n, d, generator=g)
    s = lo + (hi - lo) * torch.rand(n, 1, generator=g)
    return unit_rows(X) * s


def orthogonal_unit(n, d, e, g):
    U = torch.randn(n, d, generator=g)
    U = U - (U * e).sum(1, keepdim=True) * e
    return unit_rows(U)


def make_pairs(n, d, g, bias=0.0, r=1.0, s_lo=1.0, s_hi=1.0):
    x1 = sample_scaled_gauss(n, d, g, s_lo, s_hi)
    n1 = x1.norm(dim=1, keepdim=True)
    e = unit_rows(x1)
    u = unit_rows(torch.randn(n, d, generator=g))
    dir2 = bias * (-e) + math.sqrt(max(0.0, 1.0 - bias * bias)) * u
    x2 = (r * n1) * unit_rows(dir2)
    return x1, x2


def cosine(x1, x2):
    a = unit_rows(x1)
    b = unit_rows(x2)
    return (a * b).sum(1)


def rho_stat(x1, x2):
    return (x1 + x2).norm(dim=1) / (x1.norm(dim=1) + x2.norm(dim=1)).clamp_min(1e-12)


def psi_stat(c, r):
    return 2 * c * r ** -0.5 - (r + 1.0 / r) * c ** 2


def ridge_fit(Z, Y, lam=1e-3):
    dim = Z.shape[1]
    return torch.linalg.solve(Z.T @ Z + lam * torch.eye(dim), Z.T @ Y)


def ls_head(Z, y, train_mask, lam=1e-3):
    w = ridge_fit(Z[train_mask], y[train_mask].unsqueeze(1), lam).squeeze(1)
    pred = Z @ w
    acc_tr = ((pred[train_mask] > 0) == (y[train_mask] > 0)).float().mean().item()
    acc_te = ((pred[~train_mask] > 0) == (y[~train_mask] > 0)).float().mean().item()
    return acc_tr, acc_te


def r_star_apply(X, x1, x2, g, beta=1.0):
    e = unit_rows(x1)
    n = X.shape[0]
    E = torch.randn(n, X.shape[1], generator=g)
    E = E - (E * e).sum(1, keepdim=True) * e
    fallback = unit_rows(E)
    alpha = (x2 * e).sum(1, keepdim=True)
    w = x2 - alpha * e
    beta_w = w.norm(dim=1, keepdim=True)
    n2 = x2.norm(dim=1, keepdim=True).clamp_min(1e-12)
    mask = (beta_w < 1e-5 * n2).float()
    wh = mask * fallback + (1 - mask) * (w / beta_w.clamp_min(1e-12))
    c0 = (alpha / n2).clamp(-1.0, 1.0)
    s0 = (1 - c0 * c0).clamp_min(0.0).sqrt()
    phi = torch.atan2(c0, s0)
    cb = torch.cos(beta * phi)
    sb = torch.sin(beta * phi)
    xe = (X * e).sum(1, keepdim=True)
    xw = (X * wh).sum(1, keepdim=True)
    rest = X - xe * e - xw * wh
    return rest + (cb * xe - sb * xw) * e + (sb * xe + cb * xw) * wh


def binom_ci(acc, n):
    se = math.sqrt(max(acc * (1 - acc), 1e-9) / n)
    return 1.96 * se


def mc_se(p, n):
    return math.sqrt(max(p * (1 - p), 1e-9) / n)
