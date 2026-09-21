import csv
import math

import torch
import torch.nn as nn

from dlora.verification import (binom_ci, make_pairs, r_star_apply,
                                ridge_fit, unit_rows)

D = 128
N_LIST = [2, 4, 8, 16]
N = 4000
Q_ANTI = 0.5
BIAS = 0.8


def ensemble(N_comp, n, d, g):
    h1, _ = make_pairs(n, d, g)
    n1 = h1.norm(dim=1, keepdim=True)
    e1 = unit_rows(h1)
    hs = [h1]
    pick = (torch.rand(n, N_comp - 1, generator=g) < Q_ANTI).float()
    u = unit_rows(torch.randn(n, d, generator=g))
    anti = unit_rows(BIAS * (-e1) + math.sqrt(1 - BIAS ** 2) * u)
    rnd = unit_rows(torch.randn(n, d, generator=g))
    s = 0.5 + 1.5 * torch.rand(n, 1, generator=g)
    for i in range(N_comp - 1):
        dir_i = pick[:, i].unsqueeze(1) * anti + (1 - pick[:, i].unsqueeze(1)) * rnd
        hs.append(s * dir_i)
    return hs


def main():
    torch.manual_seed(0)
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[: 2 * N // 3] = True
    rows = []
    print("=" * 104)
    print(f"MEHRPFAD-EXTENSION: h = Sum_i h_i | N-Komponenten, q_anti={Q_ANTI}, bias={BIAS}, d={D}")
    print("=" * 104)
    print(f"{'N':>3} | {'rho_N':>7} {'rho_N_dec':>9} | {'L_base':>7} {'L_dec':>7} {'L_gate':>7} {'rot%':>6} | {'L_mtl':>7} {'L_mtl_dec':>9} | corr(cos,rho)")
    for N_comp in N_LIST:
        g = torch.Generator().manual_seed(2000 + N_comp)
        hs = ensemble(N_comp, N, D, g)
        us = [torch.randn(D, generator=g) for _ in range(N_comp)]
        S = torch.stack(hs, 0).sum(0)
        rho_n = S.norm(dim=1) / torch.stack([h.norm(dim=1) for h in hs], 1).sum(1)
        pw = []
        for i in range(N_comp):
            for j in range(i + 1, N_comp):
                a = unit_rows(hs[i])
                b = unit_rows(hs[j])
                pw.append((a * b).sum(1))
        mean_pw = torch.stack(pw, 1).mean(1)
        corr = (((mean_pw - mean_pw.mean()) * (rho_n - rho_n.mean())).sum()
                / ((mean_pw - mean_pw.mean()).norm() * (rho_n - rho_n.mean()).norm() + 1e-12)).item()

        y_mtl = torch.stack([((hs[i] @ us[i]) > 0).float() * 2 - 1 for i in range(N_comp)], 1)
        y_single = ((torch.stack([hs[i] @ us[i] for i in range(N_comp)], 1).sum(1)) > 0).float() * 2 - 1

        S_dec = hs[0].clone()
        gates = []
        for i in range(1, N_comp):
            rho_i = (S_dec + hs[i]).norm(dim=1) / (S_dec.norm(dim=1) + hs[i].norm(dim=1))
            gate_i = (rho_i < 0.25)
            gates.append(gate_i.float())
            rot = r_star_apply(hs[i], S_dec, hs[i], g)
            S_dec = S_dec + torch.where(gate_i.unsqueeze(1), rot, hs[i])
        gate_rate = torch.stack(gates, 1).mean().item()
        S_ung = torch.stack([hs[0]] + [r_star_apply(hs[i], S_dec, hs[i], g) for i in range(1, 0, -1)], 0).sum(0) if False else None

        S_ungated = hs[0].clone()
        for i in range(1, N_comp):
            S_ungated = S_ungated + r_star_apply(hs[i], S_ungated, hs[i], g)
        rho_dec = S_ungated.norm(dim=1) / torch.stack([h.norm(dim=1) for h in hs], 1).sum(1)

        def head_acc(Z, y):
            accs = []
            for k in range(y.shape[1]):
                w = ridge_fit(Z[train_mask], y[train_mask, k].unsqueeze(1)).squeeze(1)
                accs.append(((Z[~train_mask] @ w > 0) == (y[~train_mask, k] > 0)).float().mean().item())
            return sum(accs) / len(accs)

        L_base_s = 1 - head_acc(S, y_single.unsqueeze(1))
        L_dec_s = 1 - head_acc(S_ungated, y_single.unsqueeze(1))
        L_gate_s = 1 - head_acc(S_dec, y_single.unsqueeze(1))
        L_base_m = 1 - head_acc(S, y_mtl)
        L_dec_m = 1 - head_acc(S_ungated, y_mtl)

        rows.append(dict(N=N_comp, rho_N=rho_n.mean().item(), rho_N_dec=rho_dec.mean().item(),
                         L_base=L_base_s, L_dec=L_dec_s, L_gate=L_gate_s,
                         L_mtl=L_base_m, L_mtl_dec=L_dec_m, corr_pw_rho=corr))
        print(f"{N_comp:>3} | {rho_n.mean():>7.3f} {rho_dec.mean():>9.3f} | "
              f"{L_base_s:>7.3f} {L_dec_s:>7.3f} {L_gate_s:>7.3f} {gate_rate*100:>5.0f}% | "
              f"{L_base_m:>7.3f} {L_dec_m:>9.3f} | {corr:>+7.3f}")
    with open("results/mehrpfad.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "rho_N", "rho_N_dec", "L_base", "L_dec",
                                          "L_gate", "L_mtl", "L_mtl_dec", "corr_pw_rho"])
        w.writeheader()
        w.writerows(rows)
    print()
    print("Interpretation: rho_N = kollektive Ausloeschung | L_* = Task-Loss (single/ multi-task) |")
    print("corr(cos_pairwise, rho_N) beantwortet, ob paarweise Statistik kollektive Ausloeschung vorhersagt.")
    print("Gespeichert: results/mehrpfad.csv")


if __name__ == "__main__":
    main()
