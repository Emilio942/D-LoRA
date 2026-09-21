import csv
import math

import torch

from dlora.verification import (cosine, make_pairs, mc_se, r_star_apply,
                                rho_stat, ridge_fit, unit_rows)
from dlora.verification import psi_stat

N_BIG = 10_000
ALPHA = 0.01
RHO_CRIT = 0.25
R_LIST = [0.25, 0.5, 1.0, 2.0, 4.0]
DIMS_Q2 = [64, 256, 1024]


def rankdata(x):
    order = torch.argsort(x)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(len(x), dtype=x.dtype)
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return ((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12)).item()


def psi_detektor():
    print("=" * 96)
    print("ZUSATZ A: Psi(c,r) aus Cycle 41 als Ausloeschungs-Detektor (Ground Truth: rho < %.2f)" % RHO_CRIT)
    print("=" * 96)
    c_grid = torch.linspace(-1, 1, 41)
    r_grid = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0])
    rho_grid = torch.sqrt(1 + r_grid ** 2 + 2 * r_grid * c_grid.unsqueeze(1)) / (1 + r_grid)
    psi_grid = psi_stat(c_grid.unsqueeze(1).expand_as(rho_grid).contiguous(),
                        r_grid.unsqueeze(0).expand_as(rho_grid).contiguous())
    sp = spearman(psi_grid.flatten(), rho_grid.flatten())
    sp_neg = spearman(psi_grid.flatten(), -rho_grid.flatten())
    print(f"Monotonie-Check ueber (c,r)-Grid: Spearman(Psi, rho) = {sp:+.3f}, "
          f"Spearman(Psi, -rho) = {sp_neg:+.3f}")
    print("Cycle-41-Behauptung: Psi<0 bei Ausloeschung, Psi>0 bei antipod+unaligned -> "
          "erwartet Spearman(Psi, -rho) > 0.")
    print()
    print(f"{'d':>5} {'r':>5} | {'Praez_psi':>10} {'Praez_c':>9} {'Praez_rho':>10} {'Alarm_psi':>10}")
    rows = []
    for d in DIMS_Q2:
        g = torch.Generator().manual_seed(300 + d)
        x1n, x2n = make_pairs(N_BIG, d, g, bias=0.0)
        cn = cosine(x1n, x2n)
        rn = rho_stat(x1n, x2n)
        pn = psi_stat(cn, torch.full_like(cn, 1.0))
        tau_psi = torch.quantile(pn, ALPHA).item()
        tau_c = torch.quantile(cn, ALPHA).item()
        tau_rho = torch.quantile(rn, ALPHA).item()
        for r in R_LIST:
            x1b, x2b = make_pairs(N_BIG, d, g, bias=0.9, r=r)
            cb = cosine(x1b, x2b)
            rb = rho_stat(x1b, x2b)
            pb = psi_stat(cb, torch.full_like(cb, r))
            truth = rb < RHO_CRIT
            res = {}
            for name, score, fires_low in [
                ("psi", pb, True), ("c", cb, True), ("rho", rb, True)
            ]:
                thr = tau_psi if name == "psi" else (tau_c if name == "c" else tau_rho)
                flagged = score < thr
                res[name] = (truth[flagged]).float().mean().item() if flagged.any() else 0.0
            rows.append(dict(d=d, r=r, prec_psi=res["psi"], prec_c=res["c"],
                             prec_rho=res["rho"]))
            print(f"{d:>5} {r:>5.2f} | {res['psi']:>10.3f} {res['c']:>9.3f} "
                  f"{res['rho']:>10.3f} {(pb < tau_psi).float().mean().item():>10.3f}")
    with open("results/zusatz_psi.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["d", "r", "prec_psi", "prec_c", "prec_rho"])
        w.writeheader()
        w.writerows(rows)
    print()


def recovery_claims():
    print("=" * 96)
    print("ZUSATZ B: Recovery-Fraktion von R* vs. Cycle-255 (>=80%) und Cycle-450 (>=50%)")
    print("=" * 96)
    import torch.nn as nn
    from run_verification import MLP, x2_at_angle
    from dlora.verification import binom_ci
    rows = []
    print(f"{'d':>5} | {'Baseline':>9} {'R*':>9} {'Oracle':>9} | {'G_d':>7} | {'>=50%':>6} {'>=80%':>6}")
    for d in [64, 256, 1024]:
        n = 4000
        train_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[: 2 * n // 3] = True
        n_test = int((~train_mask).sum())
        gg = torch.Generator().manual_seed(1200 + d)
        u0 = torch.randn(d, generator=gg)
        v0 = torch.randn(d, generator=gg)
        x1, _ = make_pairs(n, d, gg)
        g2 = torch.Generator().manual_seed(1300 + d)
        n1 = x1.norm(dim=1, keepdim=True)
        e = unit_rows(x1)
        dir2 = -e
        x2 = (1.0 * n1) * dir2
        y = ((x1 @ u0 + x2 @ v0) > 0).float() * 2 - 1
        x2_rot = r_star_apply(x2, x1, x2, g2)
        h = x1 + x2
        h_rot = x1 + x2_rot
        Z_or = torch.cat([x1, x2], dim=1)

        def ls_acc(Z):
            w = ridge_fit(Z[train_mask], y[train_mask].unsqueeze(1)).squeeze(1)
            return ((Z[~train_mask] @ w > 0) == (y[~train_mask] > 0)).float().mean().item()

        torch.manual_seed(d)
        net = MLP(d, hidden=64)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        y01 = (y > 0).float()
        for _ in range(300):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                net(h[train_mask]), y01[train_mask])
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc_mlp = ((net(h[~train_mask]) > 0) == (y01[~train_mask] > 0)).float().mean().item()

        acc_b = ls_acc(h)
        acc_r = ls_acc(h_rot)
        acc_o = ls_acc(Z_or)
        gap = acc_o - acc_b
        frac_r = (acc_r - acc_b) / gap if gap > 1e-9 else float("nan")
        frac_m = (acc_mlp - acc_b) / gap if gap > 1e-9 else float("nan")
        rows.append(dict(d=d, acc_base=acc_b, acc_rot=acc_r, acc_oracle=acc_o,
                         acc_mlp=acc_mlp, frac_rot=frac_r, frac_mlp=frac_m))
        print(f"{d:>5} | {acc_b:>9.3f} {acc_r:>9.3f} {acc_o:>9.3f} | {frac_r:>7.3f} "
              f"| {'JA' if frac_r >= 0.5 else 'NEIN':>6} {'JA' if frac_r >= 0.8 else 'NEIN':>6}")
        print(f"      MLP post-sum: {acc_mlp:.3f} (G_m = {frac_m:+.3f})")
    with open("results/zusatz_recovery.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["d", "acc_base", "acc_rot", "acc_oracle",
                                          "acc_mlp", "frac_rot", "frac_mlp"])
        w.writeheader()
        w.writerows(rows)
    print()


def sas_test():
    print("=" * 96)
    print("ZUSATZ C: SAS (Cycle 1520) an Pythia-70M: trainierte Token-Folge vs. Token-shuffle-Null")
    print("=" * 96)

    def sas_stat(X, Y):
        Xc = X - X.mean(0, keepdim=True)
        Yc = Y - Y.mean(0, keepdim=True)
        S = (Xc.T @ Yc + Yc.T @ Xc) / (2 * len(X))
        ev = torch.linalg.eigvalsh(S)
        return (ev.min().abs() / ev.abs().max().clamp_min(1e-12)).item()

    from dlora.latents import collect_activations, load_corpus, load_model
    text = load_corpus()
    model, tok = load_model()
    layers = [0, 1, 2, 3, 4, 5]
    acts = collect_activations(model, tok, text, layers, max_tokens=4000)
    g = torch.Generator().manual_seed(77)
    print(f"{'Layer':>5} | {'SAS_trainiert':>14} {'SAS_shuffle(±se)':>22} {'z':>7}")
    out_rows = []
    for l in layers:
        H = acts[l][: 4000]
        X, Y = H[:-1], H[1:]
        sas_t = sas_stat(X, Y)
        vals = []
        for k in range(20):
            perm = torch.randperm(len(Y), generator=g)
            vals.append(sas_stat(X, Y[perm]))
        mt = sum(vals) / len(vals)
        se = (sum((v - mt) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        z = (sas_t - mt) / se if se > 0 else 0.0
        out_rows.append(dict(layer=l, sas_trained=sas_t, sas_null_mean=mt,
                             sas_null_se=se, z=z))
        print(f"{l:>5} | {sas_t:>14.4f} {mt:>14.4f} ± {se:.4f} {z:>7.2f}")
    with open("results/zusatz_sas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "sas_trained", "sas_null_mean",
                                          "sas_null_se", "z"])
        w.writeheader()
        w.writerows(out_rows)
    print()


def real_alle_layer(rows):
    print("=" * 96)
    print("ZUSATZ D: Real-Anker ueber ALLE Layer (DeltaP, E[c], E[rho], W1 der cos-Verteilung)")
    print("=" * 96)
    from dlora.latents import (collect_activations, load_corpus, load_model,
                               mine_pairs)
    text = load_corpus()
    model, tok = load_model()
    layers = [0, 1, 2, 3, 4, 5]
    acts = collect_activations(model, tok, text, layers, max_tokens=2000)
    print(f"{'Layer':>5} {'E[c]':>8} {'E[rho]':>8} {'DeltaP(-.25)':>13} {'+-MC':>7} {'W1(cos)':>9}")
    g = torch.Generator().manual_seed(55)
    for l in layers:
        H = acts[l]
        x1, x2, c_real = mine_pairs(H[:1000], H[:1000], n_random=384,
                                    n_conflict=384, generator=g)
        x1n, x2n = make_pairs(len(c_real), H.shape[1], g, bias=0.0)
        c_null = cosine(x1n, x2n)
        tau = -0.25
        pr = (c_real < tau).float().mean().item()
        pn = (c_null < tau).float().mean().item()
        se = math.sqrt(mc_se(pr, len(c_real)) ** 2 + mc_se(pn, len(c_null)) ** 2)
        rho_real = rho_stat(x1, x2).mean().item()
        w1 = (c_real.sort().values - c_null.sort().values).abs().mean().item()
        rows.append(dict(q="REAL_LAYER", layer=l, E_c=c_real.mean().item(),
                         E_rho=rho_real, delta_p=pr - pn, se=se, w1_cos=w1))
        print(f"{l:>5} {c_real.mean():>+8.3f} {rho_real:>8.3f} {pr-pn:>+13.4f} {se:>7.4f} {w1:>9.4f}")
    print()


def main():
    import os
    torch.manual_seed(0)
    os.makedirs("results", exist_ok=True)
    rows = []
    psi_detektor()
    recovery_claims()
    sas_test()
    real_alle_layer(rows)
    with open("results/zusatz_reallayer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["q", "layer", "E_c", "E_rho", "delta_p",
                                          "se", "w1_cos"])
        w.writeheader()
        w.writerows(rows)
    print("Gespeichert: results/zusatz_psi.csv, zusatz_recovery.csv, zusatz_sas.csv, zusatz_reallayer.csv")


if __name__ == "__main__":
    main()
