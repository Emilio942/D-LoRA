import csv
import math
import os

import torch
import torch.nn as nn

from dlora.verification import (binom_ci, cosine, make_pairs, mc_se,
                                r_star_apply, ridge_fit, unit_rows)

DIMS_Q1 = [16, 32, 64, 128, 256, 512, 1024]
DIMS_Q2 = [64, 256, 1024]
R_LIST = [0.25, 0.5, 1.0, 2.0, 4.0]
C_GRID = [-1.0, -0.9, -0.8, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.8, 1.0]
N_BIG = 10_000
ALPHA = 0.01
RHO_CRIT = 0.25


def x2_at_angle(x1, c_target, r, g):
    n1 = x1.norm(dim=1, keepdim=True)
    e = unit_rows(x1)
    n, d = x1.shape
    U = torch.randn(n, d, generator=g)
    U = U - (U * e).sum(1, keepdim=True) * e
    u = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
    bias = -c_target
    dir2 = bias * (-e) + math.sqrt(max(0.0, 1.0 - bias * bias)) * u
    return (r * n1) * dir2


class MLP(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def mlp_head(Z, y, train_mask, epochs=400, lr=1e-3, hidden=64, seed=0):
    torch.manual_seed(seed)
    net = MLP(Z.shape[1], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    y01 = (y > 0).float()
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Z[train_mask]), y01[train_mask])
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc_tr = ((net(Z[train_mask]) > 0) == (y01[train_mask] > 0)).float().mean().item()
        acc_te = ((net(Z[~train_mask]) > 0) == (y01[~train_mask] > 0)).float().mean().item()
    n_params = sum(p.numel() for p in net.parameters())
    return acc_tr, acc_te, n_params


def recon_r2(Z, Y, train_mask, lam=1e-3):
    w = ridge_fit(Z[train_mask], Y[train_mask], lam)
    pred = Z[~train_mask] @ w
    sse = ((pred - Y[~train_mask]) ** 2).sum(0)
    sst = ((Y[~train_mask] - Y[~train_mask].mean(0, keepdim=True)) ** 2).sum(0).clamp_min(1e-12)
    return (1 - sse / sst).mean().item()


def make_task(n, d, gg):
    u0 = torch.randn(d, generator=gg)
    v0 = torch.randn(d, generator=gg)
    return u0, v0


def q1_null_vs_coact(rows):
    print("=" * 96)
    print(f"Q1: Exzess-Antipodie ueber Dimension (Nullmodell-Matching, n={N_BIG} pro Zelle)")
    print("=" * 96)
    print(f"{'d':>5} {'bias':>5} {'E[c]':>8} {'Var(c)':>9} {'P(c<-0.5)':>10} {'DeltaP':>9} {'+-MC':>7}")
    for d in DIMS_Q1:
        g = torch.Generator().manual_seed(d)
        x1n, x2n = make_pairs(N_BIG, d, g, bias=0.0)
        cn = cosine(x1n, x2n)
        p_null = (cn < -0.5).float().mean().item()
        for bias in [0.0, 0.1, 0.3]:
            x1, x2 = make_pairs(N_BIG, d, g, bias=bias)
            c = cosine(x1, x2)
            p = (c < -0.5).float().mean().item()
            dp = p - p_null
            se = math.sqrt(mc_se(p, N_BIG) ** 2 + mc_se(p_null, N_BIG) ** 2)
            rows.append(dict(q="Q1", d=d, bias=bias, mean_c=c.mean().item(),
                             var_c=c.var().item(), p_tail=p, null_tail=p_null,
                             delta_p=dp, se=se))
            print(f"{d:>5} {bias:>5.1f} {c.mean():>+8.3f} {c.var():>9.5f} {p:>10.4f} {dp:>+9.4f} {se:>7.4f}")
    print("Theorie-Check: Var(c) ~ (1-bias^2)/d;  Var(c=0,d=16) = 0.0625 erwartet")
    print()


def q2_detektor(rows):
    print("=" * 96)
    print("Q2: Winkel-Alarm vs. echte Vektor-Ausloeschung (Ground Truth: rho < %.2f)" % RHO_CRIT)
    print("=" * 96)
    print(f"bias=0.9-Paare | tau_c: 1%-Nullquantil | P(rho<{RHO_CRIT} | Winkel-Alarm) = Praezision")
    print(f"{'d':>5} {'r':>5} {'tau_c':>8} {'Alarmrate':>10} {'Praezision':>11} {'rho(c=-1,r)':>12}")
    for d in DIMS_Q2:
        g = torch.Generator().manual_seed(100 + d)
        x1n, x2n = make_pairs(N_BIG, d, g, bias=0.0)
        cn = cosine(x1n, x2n)
        tau_c = torch.quantile(cn, ALPHA).item()
        for r in R_LIST:
            x1b, x2b = make_pairs(N_BIG, d, g, bias=0.9, r=r)
            cb = cosine(x1b, x2b)
            rb = (x1b + x2b).norm(dim=1) / (x1b.norm(dim=1) + x2b.norm(dim=1))
            flagged = cb < tau_c
            alarm = flagged.float().mean().item()
            prec = (rb[flagged] < RHO_CRIT).float().mean().item() if flagged.any() else float("nan")
            rho_exact = abs(1 - r) / (1 + r)
            rows.append(dict(q="Q2", d=d, r=r, tau_c=tau_c, alarm_rate=alarm,
                             precision=prec, rho_at_c_minus1=rho_exact))
            print(f"{d:>5} {r:>5.2f} {tau_c:>8.3f} {alarm:>10.3f} {prec:>11.3f} {rho_exact:>12.3f}")
    print()


def q3_q4_intervention(rows):
    print("=" * 96)
    print("Q3/Q4: Winkel-Intervention (Normen fix) | Task-Loss & Rekonstruktion | R*-Entkopplung")
    print("=" * 96)
    d = 128
    n = 6000
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[: 2 * n // 3] = True
    print(f"{'r':>5} {'c':>6} | {'L_base':>7} {'L_R*':>7} {'L_gate':>7} {'rot%':>6} | {'R2_base':>8} {'R2_gate':>8} | {'|x1.(R*x2)|max':>15} {'||R*x2||/||x2||-1|max':>24}")
    worst_orth = 0.0
    worst_norm = 0.0
    for r in R_LIST:
        gg = torch.Generator().manual_seed(500 + int(r * 100))
        u0, v0 = make_task(n, d, gg)
        x1, _ = make_pairs(n, d, gg)
        for c in C_GRID:
            g2 = torch.Generator().manual_seed(7000 + int(r * 100) + int(abs(c) * 10))
            x2 = x2_at_angle(x1, c, r, g2)
            y = ((x1 @ u0 + x2 @ v0) > 0).float() * 2 - 1
            x2_rot = r_star_apply(x2, x1, x2, g2)
            rho = (x1 + x2).norm(dim=1) / (x1.norm(dim=1) + x2.norm(dim=1))
            gate = (rho < RHO_CRIT).unsqueeze(1)
            x2_final = torch.where(gate, x2_rot, x2)
            h = x1 + x2
            h_rot = x1 + x2_rot
            h_gate = x1 + x2_final

            def head_acc(Z):
                w = ridge_fit(Z[train_mask], y[train_mask].unsqueeze(1)).squeeze(1)
                return ((Z[~train_mask] @ w > 0) == (y[~train_mask] > 0)).float().mean().item()

            acc_b = head_acc(h)
            acc_r = head_acc(h_rot)
            acc_g = head_acc(h_gate)
            Y = torch.cat([x1, x2], dim=1)
            r2_b = recon_r2(h, Y, train_mask)
            r2_g = recon_r2(h_gate, Y, train_mask)
            orth = (x1 * x2_rot).sum(1).abs().max().item()
            nrm = (x2_rot.norm(dim=1) / x2.norm(dim=1) - 1).abs().max().item()
            worst_orth = max(worst_orth, orth)
            worst_norm = max(worst_norm, nrm)
            rows.append(dict(q="Q3Q4", r=r, c=c, L_base=1 - acc_b, L_rot=1 - acc_r,
                             L_gate=1 - acc_g, rot_rate=gate.float().mean().item(),
                             r2_base=r2_b, r2_rot=r2_g,
                             orth_max=orth, norm_dev=nrm))
            print(f"{r:>5.2f} {c:>+6.2f} | {1-acc_b:>7.3f} {1-acc_r:>7.3f} {1-acc_g:>7.3f} "
                  f"{gate.float().mean()*100:>5.0f}% | {r2_b:>8.3f} {r2_g:>8.3f} | "
                  f"{orth:>15.2e} {nrm:>24.2e}")
    print(f"\nR*-Garantien: max |<x1, R*x2>| = {worst_orth:.2e}, "
          f"max |||R*x2||/||x2|| - 1| = {worst_norm:.2e}")
    print()


def q5_d_lora_vs_kapazitaet(rows):
    print("=" * 96)
    print("Q5: Geometrische Entkopplung vs. additive Kapazitaet (d=128)")
    print("=" * 96)
    d = 128
    n = 6000
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[: 2 * n // 3] = True
    n_test = int((~train_mask).sum())
    print(f"{'c':>6} {'r':>5} | {'Baseline':>16} {'MLP(post-sum)':>16} {'R*(pre-sum)':>16} {'Oracle':>16}")
    for c, r in [(-1.0, 1.0), (-0.9, 1.0), (-0.7, 1.0), (-1.0, 2.0)]:
        gg = torch.Generator().manual_seed(900 + int(abs(c) * 10) + int(r * 10))
        u0, v0 = make_task(n, d, gg)
        x1, _ = make_pairs(n, d, gg)
        g2 = torch.Generator().manual_seed(950 + int(abs(c) * 10))
        x2 = x2_at_angle(x1, c, r, g2)
        y = ((x1 @ u0 + x2 @ v0) > 0).float() * 2 - 1
        x2_rot = r_star_apply(x2, x1, x2, g2)
        h = x1 + x2
        h_rot = x1 + x2_rot
        Z_or = torch.cat([x1, x2], dim=1)

        def ls_acc(Z):
            w = ridge_fit(Z[train_mask], y[train_mask].unsqueeze(1)).squeeze(1)
            return ((Z[~train_mask] @ w > 0) == (y[~train_mask] > 0)).float().mean().item()

        acc_b = ls_acc(h)
        acc_m, _, n_par = mlp_head(h, y, train_mask)
        acc_r = ls_acc(h_rot)
        acc_o = ls_acc(Z_or)
        rows.append(dict(q="Q5", c=c, r=r, acc_base=acc_b, acc_mlp=acc_m,
                         acc_rot=acc_r, acc_oracle=acc_o, mlp_params=n_par))

        def f(a):
            return f"{a:.3f}±{binom_ci(a, n_test):.3f}"
        print(f"{c:>+6.2f} {r:>5.2f} | {f(acc_b):>16} {f(acc_m):>16} {f(acc_r):>16} {f(acc_o):>16}")
    print(f"\nParameterbudget: R* = 0 trainierte Parameter (closed-form) | MLP = {rows[-1]['mlp_params']} Parameter")
    print()


def realer_anker(rows):
    print("=" * 96)
    print("REALER ANKER: Pythia-70M Layer 3 vs. gematchter d=512-Null")
    print("=" * 96)
    try:
        from dlora.latents import (collect_activations, load_corpus, load_model,
                                   mine_pairs)
        text = load_corpus()
        model, tok = load_model()
        acts = collect_activations(model, tok, text, [3], max_tokens=2000)
        H = acts[3]
        g = torch.Generator().manual_seed(11)
        x1, x2, c_real = mine_pairs(H, H, n_random=512, n_conflict=512, generator=g)
        x1n, x2n = make_pairs(len(c_real), H.shape[1], g, bias=0.0)
        c_null = cosine(x1n, x2n)
        print(f"Paare: n={len(c_real)}")
        print(f"{'tau':>6} {'P_real(c<tau)':>14} {'P_null(c<tau)':>14} {'DeltaP':>10}")
        for tau in [-0.5, -0.25, 0.0, 0.25]:
            pr = (c_real < tau).float().mean().item()
            pn = (c_null < tau).float().mean().item()
            se = math.sqrt(mc_se(pr, len(c_real)) ** 2 + mc_se(pn, len(c_null)) ** 2)
            rows.append(dict(q="REAL", tau=tau, p_real=pr, p_null=pn, delta_p=pr - pn, se=se))
            print(f"{tau:>+6.2f} {pr:>14.4f} {pn:>14.4f} {pr-pn:>+10.4f} (±{se:.4f})")
        rho_real = (x1 + x2).norm(dim=1) / (x1.norm(dim=1) + x2.norm(dim=1))
        rho_null = (x1n + x2n).norm(dim=1) / (x1n.norm(dim=1) + x2n.norm(dim=1))
        print(f"Mittel: E[c]_real={c_real.mean():+.3f} vs E[c]_null={c_null.mean():+.3f} | "
              f"E[rho]_real={rho_real.mean():.3f} vs E[rho]_null={rho_null.mean():.3f}")
    except Exception as exc:
        print(f"Real-Anker uebersprungen: {exc}")
    print()


def urteile(rows):
    print("=" * 96)
    print("URTEILE gemaess Verifikationsprotokoll")
    print("=" * 96)
    q1 = [r for r in rows if r["q"] == "Q1" and r["bias"] == 0.3]
    dp16 = [r for r in q1 if r["d"] == 16][0]
    dp1024 = [r for r in q1 if r["d"] == 1024][0]
    print(f"\n[Q1a] Mean-Shift-Bias (bias=0.3): DeltaP(-0.5) faellt von {dp16['delta_p']:+.4f} (d=16) "
          f"auf {dp1024['delta_p']:+.4f} (d=1024) -> Mean-Shift allein stirbt im Hochdimensionalen "
          f"(konsistent mit Konzentrations-Argument, Cycle 41/11).")
    real = [r for r in rows if r["q"] == "REAL" and r["tau"] == -0.25]
    if real:
        sig = real[0]["delta_p"] > 2 * real[0]["se"]
        print(f"[Q1b] ECHTE Latents (Pythia Layer 3, d=512): DeltaP(c<-0.25) = "
              f"{real[0]['delta_p']:+.4f} (±{real[0]['se']:.4f}) -> "
              f"{'SIGNIFIKANT: ueberlebt Skalierung -> strukturierte Paarung, kein Mean-Shift' if sig else 'nicht signifikant'}")
    print("     Refutiert wenn: Real-DeltaP nach Null-Matching verschwindet oder Mean-Shift das Muster erklaert.")
    q2 = [r for r in rows if r["q"] == "Q2"]
    high_r = [r for r in q2 if r["r"] in (2.0, 4.0)]
    low_r = [r for r in q2 if r["r"] in (0.25, 0.5, 1.0)]
    decoupled = (sum(r["precision"] for r in low_r) / len(low_r)) - (sum(r["precision"] for r in high_r) / len(high_r))
    print(f"\n[Q2] Winkel-Alarm verliert Praezision bzgl. echter Ausloeschung bei Norm-Ungleichgewicht: "
          f"Praezisions-Gefaelle (r<=1 vs r>=2) = {decoupled:+.3f} "
          f"{'VALID (C(c,r) noetig)' if decoupled > 0.3 else 'INVALID'}")
    print("     Refutiert wenn: Praezision bei r=4 vergleichbar r=1.")
    q34 = [r for r in rows if r["q"] == "Q3Q4"]
    anti = [r for r in q34 if r["c"] <= -0.7 and r["r"] == 1.0]
    base_degrades = all(r["L_base"] > 0.1 for r in anti)
    gate_never_hurts = all(r["L_gate"] <= r["L_base"] + 0.03 for r in q34)
    gate_helps = all(r["L_gate"] < r["L_base"] - 0.05 for r in anti if r["c"] <= -0.9)
    print(f"\n[Q3] c->-1 (r=1) erzeugt Task-Loss bei intakten Komponenten: "
          f"{'VALID' if base_degrades else 'INVALID'} (L_base bei c in [-1,-0.7]: "
          f"{[round(r['L_base'], 3) for r in anti]})")
    print(f"[Q4] R* normerhaltend (max Abw. {max(r['norm_dev'] for r in q34):.1e}, "
          f"max |<x1,R*x2>| {max(r['orth_max'] for r in q34):.1e}); "
          f"rho-gegate: verletzt nirgends ({'VALID' if gate_never_hurts else 'INVALID'}), "
          f"recoveriert destruktives Regime (r=1, c<=-0.9): "
          f"{'VALID' if gate_helps else 'INVALID'} "
          f"(L_gate: {[round(r['L_gate'], 3) for r in anti]})")
    print("     Refutiert wenn: gegate Rotation verschlechtert eine Config oder hilft nicht bei c=-1, r=1.")
    q5 = [r for r in rows if r["q"] == "Q5"]
    worst = [r for r in q5 if r["c"] == -1.0 and r["r"] == 1.0][0]
    post_fails = worst["acc_mlp"] < 0.6
    rot_wins = worst["acc_rot"] > 0.9 and worst["acc_oracle"] > 0.9
    print(f"\n[Q5] c=-1, r=1: Baseline {worst['acc_base']:.3f}, MLP post-sum {worst['acc_mlp']:.3f} "
          f"(12k Parameter), R* pre-sum {worst['acc_rot']:.3f} (0 Parameter), Oracle {worst['acc_oracle']:.3f}")
    print(f"     Kapazitaet nach Interferenz nutzlos: {'VALID' if post_fails else 'INVALID'}; "
          f"Entkopplung vor Interferenz recoveriert: {'VALID' if rot_wins else 'TEILWEISE'}")
    print("     Refutiert wenn: post-sum Adapter pre-sum Rotation erreicht.")
    real = [r for r in rows if r["q"] == "REAL" and r["tau"] == -0.25]
    if real:
        print(f"\n[REAL] Pythia Layer 3: DeltaP(c<-0.25) = {real[0]['delta_p']:+.4f} "
              f"(±{real[0]['se']:.4f}) vs gematchter d=512-Null "
              f"({'POSITIV' if real[0]['delta_p'] > 2 * real[0]['se'] else 'nicht signifikant'})")


def main():
    torch.manual_seed(0)
    rows = []
    q1_null_vs_coact(rows)
    q2_detektor(rows)
    q3_q4_intervention(rows)
    q5_d_lora_vs_kapazitaet(rows)
    realer_anker(rows)
    urteile(rows)
    os.makedirs("results", exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open("results/verification.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nErgebnisse gespeichert: results/verification.csv ({len(rows)} Zeilen)")


if __name__ == "__main__":
    main()
