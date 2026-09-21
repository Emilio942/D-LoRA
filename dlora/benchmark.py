import csv
import os

import torch

from .evaluate import eval_probes
from .methods import DiffReference, GatedDiffAdapter, SumBaseline


def build_methods(seed=0):
    torch.manual_seed(seed)
    return [SumBaseline(), DiffReference(), GatedDiffAdapter()]


def _pooled_corr(pairs):
    preds = torch.cat([p for p, _ in pairs])
    cs = torch.cat([c for _, c in pairs])
    ca, cb = preds - preds.mean(), cs - cs.mean()
    return ((ca * cb).sum() / (ca.norm() * cb.norm() + 1e-12)).abs().item()


def run_all(methods, x1_tr, x2_tr, c_tr, configs, noise=0.05, csv_path=None,
            compact=False, calibrations=None):
    print("Alle Methoden werden auf jeder Konfiguration vollstaendig ausgewertet (kein Abbruch).\n")
    rows = []
    c_store = {m.name: [] for m in methods}
    for m in methods:
        m.fit(x1_tr, x2_tr, c_tr, noise=noise)
    current_dim = x1_tr.shape[1]
    for label, x1, x2, c in configs:
        d = x1.shape[1]
        if d != current_dim:
            current_dim = d
            print(f"--- Dimensionswechsel auf d={d}: Kalibrierung der Probes mit gesehenen Winkeln ---")
            cx1, cx2, cc = calibrations[d]
            for m in methods:
                m.calibrate(cx1, cx2, cc, noise=noise)
        y = torch.cat([x1, x2], dim=1)
        for m in methods:
            z, c_hat = m.forward(x1, x2)
            z = z + noise * torch.randn_like(z)
            res = eval_probes(m.w_dec, m.c_probe, z, y, c)
            res.update(config=label, method=m.name)
            extra = ""
            if c_hat is not None:
                mae = (c_hat - c).abs().mean().item()
                res["c_hat_mae"] = mae
                extra = f" | cMAE(int)={mae:.3f}"
            with torch.no_grad():
                c_store[m.name].append((m.c_probe(z).cpu(), c.cpu()))
            rows.append(res)
            if compact:
                print(
                    f"{label:>12s} | {m.name:<28s} | MSE={res['recon_mse']:.4f} "
                    f"| cMSE={res['c_mse']:.4f}{extra}"
                )
            else:
                print(
                    f"{label:>22s} | {m.name:<28s} | R2={res['r2_mean']:>8.3f} "
                    f"| R2min={res['r2_min']:>8.3f} | MSE={res['recon_mse']:.4f} "
                    f"| cMSE={res['c_mse']:.4f}{extra}"
                )
        print()
    print("Gepoolte Konflikterkennung (ueber alle Konfigurationen):")
    for m in methods:
        print(f"  {m.name:<28s} | corr(c_pred, c) = {_pooled_corr(c_store[m.name]):.3f}")
    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        keys = ["config", "method", "recon_mse", "r2_mean", "r2_min",
                "c_mse", "c_corr", "c_hat_mae"]
        for r in rows:
            r.pop("c_corr", None)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nErgebnisse gespeichert: {csv_path}")
    return rows


def gate_report(adapter):
    with torch.no_grad():
        cs = torch.linspace(-1, 1, 9)
        g = torch.nn.functional.softplus(adapter.gates(cs.unsqueeze(1)))
        print("Gelernte Gate-Politik:  c_hat -> (gamma_summe, gamma_differenz)")
        for ci, (gs, gd) in zip(cs, g):
            print(f"  c={ci:+.2f} | gamma_summe={gs.item():.3f} | gamma_differenz={gd.item():.3f}")
