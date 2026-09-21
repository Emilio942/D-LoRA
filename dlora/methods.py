import torch
import torch.nn as nn
import torch.nn.functional as F

from .evaluate import fit_c_probe, fit_probes


def _add_noise(z, noise):
    return z + noise * torch.randn_like(z)


class SumBaseline:
    name = "Summe (Baseline)"
    short = "Summe"

    def _z(self, x1, x2):
        return x1 + x2, None

    def _fit_probes(self, x1, x2, c, noise, lam, probe_epochs=2500, max_n=8192):
        z, _ = self._z(x1, x2)
        z = _add_noise(z, noise)
        y = torch.cat([x1, x2], dim=1)
        self.w_dec = fit_probes(z, y, lam)
        self.c_probe = fit_c_probe(z, c, epochs=probe_epochs, max_n=max_n)

    def fit(self, x1, x2, c, noise=0.05, lam=1e-4, **kw):
        self._fit_probes(x1, x2, c, noise, lam)

    def calibrate(self, x1, x2, c, noise=0.05, lam=1e-4):
        self._fit_probes(x1, x2, c, noise, lam, probe_epochs=2500, max_n=10**9)

    def forward(self, x1, x2):
        return self._z(x1, x2)


class DiffReference:
    name = "Referenz [Summe|Differenz]"
    short = "Referenz"

    def _z(self, x1, x2):
        return torch.cat([x1 + x2, x1 - x2], dim=1), None

    def _fit_probes(self, x1, x2, c, noise, lam, probe_epochs=2500, max_n=8192):
        z, _ = self._z(x1, x2)
        z = _add_noise(z, noise)
        y = torch.cat([x1, x2], dim=1)
        self.w_dec = fit_probes(z, y, lam)
        self.c_probe = fit_c_probe(z, c, epochs=probe_epochs, max_n=max_n)

    def fit(self, x1, x2, c, noise=0.05, lam=1e-4, **kw):
        self._fit_probes(x1, x2, c, noise, lam)

    def calibrate(self, x1, x2, c, noise=0.05, lam=1e-4):
        self._fit_probes(x1, x2, c, noise, lam, probe_epochs=2500, max_n=10**9)

    def forward(self, x1, x2):
        return self._z(x1, x2)


class PooledConflictEstimator(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x1, x2):
        s11 = (x1 * x1).mean(1)
        s22 = (x2 * x2).mean(1)
        s12 = (x1 * x2).mean(1)
        cos_raw = s12 / (s11 * s22).clamp_min(1e-12).sqrt()
        p = torch.stack([cos_raw, s11.log().clamp_min(-10), s22.log().clamp_min(-10)], dim=1)
        return torch.tanh(self.net(p)).squeeze(1)


class GatedDiffAdapter(nn.Module):
    name = "D-LoRA (gelernt)"
    short = "D-LoRA"

    def __init__(self, hidden=16):
        super().__init__()
        self.estimator = PooledConflictEstimator(hidden)
        self.gates = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.gates[0].weight)
        nn.init.zeros_(self.gates[0].bias)
        nn.init.zeros_(self.gates[2].weight)
        self.gates[2].bias.data = torch.tensor([0.0, -2.0])

    def forward(self, x1, x2):
        c_hat = self.estimator(x1, x2)
        g = F.softplus(self.gates(c_hat.unsqueeze(1)))
        z = torch.cat([g[:, 0:1] * (x1 + x2), g[:, 1:2] * (x1 - x2)], dim=1)
        return z, c_hat

    def fit(self, x1, x2, c, noise=0.05, lam=1e-4, epochs=4000, lr=3e-3,
            lambda_c=0.5, seed=0, verbose=False):
        torch.manual_seed(seed)
        d = x1.shape[1]
        y = torch.cat([x1, x2], dim=1)
        dec = nn.Linear(2 * d, 2 * d)
        opt = torch.optim.Adam(list(self.parameters()) + list(dec.parameters()), lr=lr)
        for epoch in range(1, epochs + 1):
            opt.zero_grad()
            z, _ = self(x1, x2)
            z = z + noise * torch.randn_like(z)
            c_hat = self.estimator(x1, x2)
            loss = F.mse_loss(dec(z), y) + lambda_c * F.mse_loss(c_hat, c)
            loss.backward()
            opt.step()
            if verbose and epoch % 1000 == 0:
                print(f"  Adapter-Training Epoche {epoch:5d} | Loss {loss.item():.5f}")
        self.calibrate(x1, x2, c, noise=noise, lam=lam)

    def calibrate(self, x1, x2, c, noise=0.05, lam=1e-4):
        with torch.no_grad():
            z, _ = self(x1, x2)
            z = _add_noise(z, noise)
            y = torch.cat([x1, x2], dim=1)
            self.w_dec = fit_probes(z, y, lam)
        self.c_probe = fit_c_probe(z, c, epochs=800, max_n=4096)
