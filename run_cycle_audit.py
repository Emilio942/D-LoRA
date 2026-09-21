import csv
import math
import os
import sys

import torch

from dlora.latents import load_corpus
from dlora.verification import r_star_apply, unit_rows
from run_kausal import forward_with_patch, load_by_name, solve_s

DOSES = [0.05, 0.2]
BETAS = [1.0]
NPROBE = 3
KTOP = 50


def entropy(logits):
    p = torch.softmax(logits.double(), dim=-1)
    return (-(p * torch.log(p.clamp_min(1e-45))).sum(-1)).float()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    text = load_corpus()
    rows = []
    per_cell = {}
    for name, layers in [("pythia", [1, 3, 5]), ("gpt2", [2, 6, 10])]:
        model, tok, _ = load_by_name(name)
        model = model.to(device).eval()
        d_model = model.config.hidden_size
        W_U = (model.lm_head.weight if hasattr(model, "lm_head")
               else model.embed_out.weight).detach().float().cpu()
        G = W_U @ W_U.T if False else W_U.T @ W_U
        evals, evecs = torch.linalg.eigh(G)
        v_max = evecs[:, -1]
        Uk = evecs[:, -KTOP:]
        ids = tok.encode(text)[: 150 * 128 + 1]
        chunks = torch.tensor([ids[k * 128:(k + 1) * 128] for k in range(150)],
                              device=device)
        lm = (model.transformer.h if hasattr(model, "transformer")
              else model.gpt_neox.layers)
        for layer in layers:
            lb, H_full = forward_with_patch(model, lm[layer], chunks, None,
                                            device, capture=True)
            H_last = H_full.reshape(-1, d_model)
            lb_sm = torch.log_softmax(lb.double(), dim=1)
            Hl = H_last[: len(chunks) * 128].reshape(len(chunks), 128, d_model)[:, -1, :]
            Hn = unit_rows(Hl)
            z0 = (lb @ W_U.to(device)).double() if False else None
            logits_base = lb.double().cpu()
            p0 = torch.softmax(logits_base, dim=-1)
            H0 = entropy(lb.cpu())
            g = torch.Generator().manual_seed(77)
            Xc = H_last - H_last.mean(0, keepdim=True)
            Sig = (Xc.T @ Xc) / len(Xc)
            seval, sevec = torch.linalg.eigh(Sig)
            seval = seval.flip(0)
            sevec = sevec.flip(1)
            k90 = int((torch.cumsum(seval, 0) / seval.sum() < 0.9).sum().item()) + 1
            keep, data = [], []
            for k in range(150):
                t_abs = k * 128 + 127
                h = Hl[k]
                cos = Hn @ (h / h.norm().clamp_min(1e-8))
                cos[max(0, t_abs - 130): t_abs + 130] = -2.0
                cos[cos.abs() > 0.995] = -2.0
                j_min, j_max = int(cos.argmin()), int(cos.argmax())
                if cos[j_min] < 0.0 and cos[j_min] > -1.5:
                    q = Hl[j_min]
                elif cos[j_max] > 0.05:
                    q = -Hl[j_max]
                else:
                    continue
                a, b, m = h.norm().item(), q.norm().item(), float(h @ q)
                if m >= 0.0:
                    continue
                qperp_chk = q - (q @ h) / (h @ h + 1e-12) * h
                if qperp_chk.norm().item() / q.norm().clamp_min(1e-12).item() < 0.05:
                    continue
                ss = {}
                ok = True
                for rho in DOSES:
                    s = solve_s(a, b, m, rho)
                    if s is None:
                        ok = False
                        break
                    ss[rho] = s
                if not ok:
                    continue
                Rq = r_star_apply(q.unsqueeze(0), h.unsqueeze(0), q.unsqueeze(0),
                                  g).squeeze(0)
                keep.append(k)
                data.append((h, q, Rq, ss, a, b, m))
            n = len(keep)
            sel_chunks = chunks[keep]
            gid = torch.arange(n, device=device)
            out = {}
            for dose in DOSES:
                for kind in ["C", "B"]:
                    deltas = torch.stack([
                        (data[i][3][dose] * data[i][1]) if kind == "C"
                        else (data[i][3][dose] * data[i][2]) for i in range(n)
                    ]).to(device)
                    lc, _ = forward_with_patch(model, lm[layer], sel_chunks,
                                               deltas, device)
                    lc_sm = torch.log_softmax(lc.double(), dim=1)
                    out[(dose, kind)] = (lb_sm[gid] * (lb_sm[gid] - lc_sm)).sum(1).cpu()
                    if kind == "C" and dose == DOSES[0]:
                        out["logits_C"] = lc.float().cpu()
                        out["deltas_C"] = deltas.cpu()
            base_H = H0[keep]
            cand = {k: [] for k in ["readout_norm", "umci", "lia", "lcm_quad",
                                    "cerlc", "ent_ratio", "entdiff_rot",
                                    "topspan", "eig_align", "eig_mass", "cni",
                                    "pencil_gap", "downstream_sens"]}
            probes = [unit_rows(torch.randn(n, d_model, generator=g)).squeeze()
                      for _ in range(NPROBE)]
            sens = []
            for pi in range(NPROBE):
                eps = probes[pi] * Hl.norm(dim=1, keepdim=True).mean()
                lp, _ = forward_with_patch(model, lm[layer], sel_chunks,
                                           eps.to(device), device)
                sens.append((lp - lb[keep]).norm(dim=1).cpu() / eps.norm(dim=1).cpu())
            cand["downstream_sens"] = torch.stack(sens).mean(0)
            for i, (h, q, Rq, ss, a, b, m) in enumerate(data):
                Wuq = W_U @ q
                WuRq = W_U @ Rq
                Wh = W_U @ h
                dC20 = (W_U @ (ss[0.2] * q)).double()
                dC05 = (W_U @ (ss[0.05] * q)).double()
                dR = W_U @ (Rq - q)
                cand["readout_norm"].append(Wuq.norm().item() / q.norm().clamp_min(1e-12).item())
                perp = Wuq - (Wuq @ Wh) / (Wh @ Wh + 1e-12) * Wh
                cand["umci"].append(perp.norm().item() / Wuq.norm().clamp_min(1e-12).item())
                cand["lia"].append(abs(Rq @ v_max) / Rq.norm().clamp_min(1e-12).item())
                dz2 = dC20.float()
                cand["lcm_quad"].append(
                    ((p0[keep[i]] * (1 - p0[keep[i]])) * dz2 ** 2).sum().item()
                    - (p0[keep[i]] * dz2).sum().item() ** 2)
                gradH = p0[keep[i]] * (p0[keep[i]] @ logits_base[keep[i]] - logits_base[keep[i]])
                cand["cerlc"].append(-float(dC20 @ gradH.double()))
                cand["ent_ratio"].append(
                    (entropy(out["logits_C"][i].unsqueeze(0)) / H0[keep[i]].clamp_min(1e-6)).item())
                eq = torch.softmax((W_U @ Rq).double(), dim=-1)
                eqq = torch.softmax(Wuq.double(), dim=-1)
                cand["entdiff_rot"].append(
                    (-(eq * eq.clamp_min(1e-45).log()).sum()
                     - -(eqq * eqq.clamp_min(1e-45).log()).sum()).item())
                dzR = dR
                sig = evals[-KTOP:].clamp_min(1e-12).sqrt()
                U_left = W_U @ (Uk / sig)
                wv = dzR
                proj = U_left @ (U_left.T @ wv)
                cand["topspan"].append(
                    proj.norm().item() / wv.norm().clamp_min(1e-12).item())
                cand["eig_align"].append(abs(sevec[:, 0] @ (q / q.norm().clamp_min(1e-8))).item())
                cand["eig_mass"].append(
                    (sevec[:, :k90].T @ (q / q.norm().clamp_min(1e-8))).norm().item())
                qperp = q - (q @ h) / (h @ h + 1e-12) * h
                cand["cni"].append(qperp.norm().item() / h.norm().clamp_min(1e-12).item())
                Ahh, Aqq = float(h @ (G @ h)), float(q @ (G @ q))
                Ahq = float(h @ (G @ q))
                l1 = (Ahh + Aqq) / 2 + math.sqrt(((Ahh - Aqq) / 2) ** 2 + Ahq ** 2)
                l2 = (Ahh + Aqq) / 2 - math.sqrt(((Ahh - Aqq) / 2) ** 2 + Ahq ** 2)
                cand["pencil_gap"].append((l1 - l2) / l1 if l1 > 0 else 0.0)
            means = {}
            for key in cand:
                t = torch.tensor(cand[key], dtype=torch.float64)
                t = torch.where(torch.isfinite(t), t, torch.zeros_like(t))
                means[key] = t.mean().item()
                rows.append(dict(model=name, layer=layer, n=n, candidate=key,
                                 sigma_mean=means[key], sigma_median=t.median().item()))
            kl = {}
            for dose in DOSES:
                kl[(dose, "C")] = out[(dose, "C")].mean().item()
                kl[(dose, "B")] = out[(dose, "B")].mean().item()
            per_cell[(name, layer)] = dict(means=means, kl=kl, n=n)
            print(f"{name} L{layer} n={n} | " + " ".join(
                f"{k}={v:.4g}" for k, v in means.items()))
            print(f"   KL: " + " ".join(
                f"d{d}:{kl[(d,'C')]:.3g}/{kl[(d,'B')]:.3g}" for d in DOSES))
        del model
        torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open("results/cycle_audit_sigma.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "layer", "n", "candidate",
                                          "sigma_mean", "sigma_median"])
        w.writeheader()
        w.writerows(rows)

    cells = list(per_cell.keys())
    print("\n" + "=" * 100)
    print("AUDIT: Trennung Rotation vs Attenuation bei Dosis 0.2 (0 freie Parameter)")
    print("=" * 100)
    cands = list(per_cell[cells[0]]["means"].keys())
    results = []
    for key in cands:
        vals = {c: per_cell[c]["means"][key] for c in cells}
        rot = [vals[c] for c in cells
               if (c[0], c[1]) in [("pythia", 1), ("pythia", 3), ("gpt2", 2)]]
        att = [vals[c] for c in cells
               if (c[0], c[1]) in [("pythia", 5), ("gpt2", 6), ("gpt2", 10)]]
        sep_up = min(rot) > max(att)
        sep_dn = max(rot) < min(att)
        lo = [per_cell[c]["means"][key] for c in cells]
        lratio = []
        for c in cells:
            kl = per_cell[c]["kl"]
            for dose in DOSES:
                lratio.append(math.log(max(kl[(dose, "B")], 1) / max(kl[(dose, "C")], 1)))
        sv = []
        for c in cells:
            sv.append(vals[c])
        import statistics
        def spearman(a, b):
            ra = {v: i for i, v in enumerate(sorted(range(len(a)), key=lambda i: a[i]))}
            rb = {v: i for i, v in enumerate(sorted(range(len(b)), key=lambda i: b[i]))}
            xa = [ra[i] for i in range(len(a))]
            xb = [rb[i] for i in range(len(b))]
            ma, mb = statistics.mean(xa), statistics.mean(xb)
            num = sum((x - ma) * (y - mb) for x, y in zip(xa, xb))
            den = math.sqrt(sum((x - ma) ** 2 for x in xa) * sum((y - mb) ** 2 for y in xb)) + 1e-12
            return num / den
        sp = spearman(lo, lratio)
        ok = sep_up or sep_dn
        direction = "up" if sep_up else ("down" if sep_dn else "-")
        results.append((key, ok, direction, sp))
        print(f"{key:<16} | getrennt: {'JA' if ok else 'NEIN':<4} ({direction:>4}) | "
              f"Spearman(sigma, log(KL_B/KL_C)) = {sp:+.3f}")
    with open("results/cycle_audit_verdict.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate", "separated", "direction", "spearman"])
        w.writeheader()
        for key, ok, direction, sp in results:
            w.writerow(dict(candidate=key, separated=ok, direction=direction,
                            spearman=sp))
    print("\nGespeichert: results/cycle_audit_sigma.csv, results/cycle_audit_verdict.csv")


if __name__ == "__main__":
    main()
