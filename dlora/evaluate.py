import torch
import torch.nn as nn


def ridge_fit(z, y, lam=1e-4):
    dim = z.shape[1]
    a = z.T @ z + lam * torch.eye(dim)
    return torch.linalg.solve(a, z.T @ y)


def fit_probes(z, y, lam=1e-4):
    return ridge_fit(z, y, lam)


class MLPProbe(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(1)


def fit_c_probe(z, c, epochs=1500, lr=1e-3, seed=0, max_n=8192, hidden=128):
    torch.manual_seed(seed)
    n = z.shape[0]
    idx = torch.randperm(n)[: min(n, max_n)]
    probe = MLPProbe(z.shape[1], hidden=hidden)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    with torch.enable_grad():
        for _ in range(epochs):
            opt.zero_grad()
            loss = ((probe(z[idx]) - c[idx]) ** 2).mean()
            loss.backward()
            opt.step()
    return probe


@torch.no_grad()
def eval_probes(w_dec, c_probe, z, y, c):
    pred = z @ w_dec
    recon_mse = ((pred - y) ** 2).mean().item()
    sse = ((pred - y) ** 2).sum(0)
    sst = ((y - y.mean(0, keepdim=True)) ** 2).sum(0)
    r2 = torch.where(sst > 1e-6, 1 - sse / sst.clamp_min(1e-12), torch.ones_like(sst))
    if c_probe is not None:
        c_pred = c_probe(z)
        ca, cb = c_pred - c_pred.mean(), c - c.mean()
        corr = abs(((ca * cb).sum() / (ca.norm() * cb.norm() + 1e-12)).item())
        c_mse = ((c_pred - c) ** 2).mean().item()
    else:
        corr, c_mse = float("nan"), float("nan")
    return {
        "recon_mse": recon_mse,
        "r2_mean": r2.mean().item(),
        "r2_min": r2.min().item(),
        "c_mse": c_mse,
        "c_corr": corr,
    }
