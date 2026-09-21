import csv
import math
import os
import sys

import torch

from dlora.verification import (binom_ci, cosine, make_pairs, r_star_apply,
                                ridge_fit, unit_rows)
from run_kausal import forward_with_patch, load_by_name, solve_s

DOSE = 0.2
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def beta_sweep_synthetic(rows):
    print("=" * 96)
    print("BETA-SWEEP synthetisch (Q4-Harness, c=-1, r=1, d=128): Accuracy vs. Rotationsanteil")
    print("=" * 96)
    d, n = 128, 6000
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[: 2 * n // 3] = True
    print(f"{'beta':>6} | {'Acc':>7} {'+-CI':>7} | {'R2':>7}")
    for beta in BETAS:
        g = torch.Generator().manual_seed(3100)
        gg = torch.Generator().manual_seed(3200)
        u0 = torch.randn(d, generator=gg)
        v0 = torch.randn(d, generator=gg)
        x1, _ = make_pairs(n, d, g)
        n1 = x1.norm(dim=1, keepdim=True)
        x2 = -(1.0 * n1) * unit_rows(x1)
        y = ((x1 @ u0 + x2 @ v0) > 0).float() * 2 - 1
        x2_rot = r_star_apply(x2, x1, x2, g, beta=beta)
        h = x1 + x2_rot
        w = ridge_fit(h[train_mask], y[train_mask].unsqueeze(1)).squeeze(1)
        acc = ((h[~train_mask] @ w > 0) == (y[~train_mask] > 0)).float().mean().item()
        Y = torch.cat([x1, x2], dim=1)
        wr = ridge_fit(h[train_mask], Y[train_mask])
        pred = h[~train_mask] @ wr
        sse = ((pred - Y[~train_mask]) ** 2).sum(0)
        sst = ((Y[~train_mask] - Y[~train_mask].mean(0, keepdim=True)) ** 2).sum(0).clamp_min(1e-12)
        r2 = (1 - sse / sst).mean().item()
        rows.append(dict(kind="synth", beta=beta, acc=acc, r2=r2))
        print(f"{beta:>6.2f} | {acc:>7.3f} {binom_ci(acc, n // 3):>7.3f} | {r2:>7.3f}")
    print()


def beta_sweep_kausal(rows, name, layers):
    print("=" * 96)
    print(f"BETA-SWEEP kausal ({name}, Dosis {DOSE}): KL(B_beta) relativ zu Baseline; "
          f"KL_C als Referenz")
    print("=" * 96)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, _ = load_by_name(name)
    model = model.to(device).eval()
    d_model = model.config.hidden_size
    text = ""
    from dlora.latents import load_corpus
    text = load_corpus()
    ids = tok.encode(text)[: 150 * 128 + 1]
    chunks = torch.tensor([ids[k * 128:(k + 1) * 128] for k in range(150)], device=device)
    targets = torch.tensor([ids[(k + 1) * 128] for k in range(150)], device=device)
    layer_modules = (model.transformer.h if hasattr(model, "transformer")
                     else model.gpt_neox.layers)
    g = torch.Generator().manual_seed(77)
    for layer in layers:
        lb, H_full = forward_with_patch(model, layer_modules[layer], chunks, None,
                                        device, capture=True)
        H_last = H_full.reshape(-1, d_model)
        Hn = unit_rows(H_last)
        lb_sm = torch.log_softmax(lb, dim=1)
        base_arg = lb.argmax(1)
        keep_idx, deltas_by_beta, kl_c = [], {b: [] for b in BETAS}, []
        for k in range(150):
            t_abs = k * 128 + 127
            h = H_last[t_abs]
            cos = Hn @ (h / h.norm().clamp_min(1e-8))
            cos[max(0, t_abs - 130): t_abs + 130] = -2.0
            j_min = int(cos.argmin())
            j_max = int(cos.argmax())
            if cos[j_min] < 0.0 and cos[j_min] > -1.5:
                q = H_last[j_min]
            elif cos[j_max] > 0.05:
                q = -H_last[j_max]
            else:
                continue
            a, b, m = h.norm().item(), q.norm().item(), float(h @ q)
            if m >= 0.0:
                continue
            s = solve_s(a, b, m, DOSE)
            if s is None:
                continue
            keep_idx.append(k)
            kl_c.append(s * q)
            for beta in BETAS:
                rot = r_star_apply(q.unsqueeze(0), h.unsqueeze(0), q.unsqueeze(0),
                                   g, beta=beta).squeeze(0)
                deltas_by_beta[beta].append((s * b) * unit_rows(rot.unsqueeze(0)).squeeze(0))
        n = len(keep_idx)
        sel_chunks = chunks[keep_idx]
        sel_targets = targets[keep_idx]
        lc, _ = forward_with_patch(model, layer_modules[layer], sel_chunks,
                                   torch.stack(kl_c).to(device), device)
        lc_sm = torch.log_softmax(lc, dim=1)
        loc = torch.arange(len(keep_idx), device=lc_sm.device)
        gid = torch.arange(len(keep_idx), device=lc_sm.device)
        kl_c_vals = (lb_sm[gid] * (lb_sm[gid] - lc_sm)).sum(1).cpu()
        print(f"\nLayer {layer} | n={n} | KL_C (Dosis {DOSE}) = {kl_c_vals.mean():.1f}")
        print(f"{'beta':>6} | {'KL(B_beta)':>12} {'dKL(C-B_beta)':>14} {'+-CI':>10} {'z':>7}")
        for beta in BETAS:
            db = torch.stack(deltas_by_beta[beta]).to(device)
            lcb, _ = forward_with_patch(model, layer_modules[layer],
                                        sel_chunks, db, device)
            lcb_sm = torch.log_softmax(lcb, dim=1)
            klb = (lb_sm[gid] * (lb_sm[gid] - lcb_sm)).sum(1).cpu()
            dkl = kl_c_vals - klb
            ci = 1.96 * dkl.std(unbiased=True).item() / math.sqrt(n)
            zf = (dkl > 0).float().mean().item()
            zscore = (zf - 0.5) / (0.5 / math.sqrt(n))
            rows.append(dict(kind="kausal", model=name, layer=layer, beta=beta,
                             kl_b=klb.mean().item(), dkl=dkl.mean().item(), ci=ci,
                             z=zscore))
            print(f"{beta:>6.2f} | {klb.mean():>12.1f} {dkl.mean():>+14.1f} {ci:>10.1f} {zscore:>+7.2f}")
        torch.cuda.empty_cache()
    print()


def main():
    torch.manual_seed(0)
    rows = []
    beta_sweep_synthetic(rows)
    name = sys.argv[1] if len(sys.argv) > 1 else "pythia"
    layers = [3, 5] if name == "pythia" else [10]
    beta_sweep_kausal(rows, name, layers)
    os.makedirs("results", exist_ok=True)
    fname = f"results/beta_sweep_{name}.csv"
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "model", "layer", "beta", "acc",
                                          "r2", "kl_b", "dkl", "ci", "z"])
        w.writeheader()
        w.writerows(rows)
    print(f"Gespeichert: {fname}")


if __name__ == "__main__":
    main()
