import csv
import math
import os
import sys

import torch

from dlora.latents import load_corpus
from dlora.verification import unit_rows, r_star_apply
from run_kausal import forward_with_patch, solve_s

DOSES = [0.05, 0.1, 0.2, 0.4]


def load_by_name(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    return model, tok


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    text = load_corpus()
    models = {
        "pythia-70m": ("EleutherAI/pythia-70m", range(6)),
        "gpt2": ("gpt2", range(12)),
        "pythia-160m": ("EleutherAI/pythia-160m", range(12)),
    }
    only = sys.argv[1] if len(sys.argv) > 1 else None
    rows = []
    for key, (hf_name, layer_range) in models.items():
        if only and key != only:
            continue
        model, tok = load_by_name(hf_name)
        model = model.to(device).eval()
        d_model = model.config.hidden_size
        W_U = (model.lm_head.weight if hasattr(model, "lm_head")
               else model.embed_out.weight).detach().float().cpu()
        evals, evecs = torch.linalg.eigh(W_U.T @ W_U)
        v_max = evecs[:, -1]
        ids = tok.encode(text)[: 150 * 128 + 1]
        chunks = torch.tensor([ids[k * 128:(k + 1) * 128] for k in range(150)],
                              device=device)
        lm = (model.transformer.h if hasattr(model, "transformer")
              else model.gpt_neox.layers)
        for layer in layer_range:
            lb, H_full = forward_with_patch(model, lm[layer], chunks, None,
                                            device, capture=True)
            H_last = H_full.reshape(-1, d_model)
            Hl = H_last[:150 * 128].reshape(150, 128, d_model)[:, -1, :]
            Hn = unit_rows(Hl)
            lb_sm = torch.log_softmax(lb.double(), dim=1)
            g = torch.Generator().manual_seed(77)
            keep, dC, dB = [], {d: [] for d in DOSES}, {d: [] for d in DOSES}
            lias = []
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
                lias.append(abs(Rq @ v_max).item() / Rq.norm().clamp_min(1e-12).item())
                for rho in DOSES:
                    dC[rho].append(ss[rho] * q)
                    dB[rho].append(ss[rho] * Rq)
            n = len(keep)
            if n < 30:
                print(f"{key} L{layer}: n={n}, uebersprungen")
                continue
            sel = chunks[keep]
            gid = torch.arange(n, device=device)
            sigma = sum(lias) / n
            row = dict(model=key, layer=layer, n=n, sigma_lia=sigma)
            for dose in DOSES:
                lc, _ = forward_with_patch(model, lm[layer], sel,
                                           torch.stack(dC[dose]).to(device), device)
                lrb, _ = forward_with_patch(model, lm[layer], sel,
                                            torch.stack(dB[dose]).to(device), device)
                lc_sm = torch.log_softmax(lc.double(), dim=1)
                lrb_sm = torch.log_softmax(lrb.double(), dim=1)
                klC = (lb_sm[gid] * (lb_sm[gid] - lc_sm)).sum(1).mean().item()
                klB = (lb_sm[gid] * (lb_sm[gid] - lrb_sm)).sum(1).mean().item()
                row[f"klC_d{dose}"] = klC
                row[f"klB_d{dose}"] = klB
                row[f"ratio_d{dose}"] = klB / max(klC, 1.0)
            rows.append(row)
            print(f"{key} L{layer:>2} n={n} sigma={sigma:.4f} | " + " ".join(
                f"d{d}: {row[f'ratio_d{d}']:.3f}" for d in DOSES))
        del model
        torch.cuda.empty_cache()
    os.makedirs("results", exist_ok=True)
    with open("results/lia_matrix.csv", "w", newline="") as f:
        keys = ["model", "layer", "n", "sigma_lia"] + [
            f"{p}_d{d}" for d in DOSES for p in ["klC", "klB", "ratio"]]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\n--- Analyse (Rotation hilfreich iff ratio<1) ---")
    for dose in DOSES:
        cells = [(r["sigma_lia"], r[f"ratio_d{dose}"] < 1.0, r["model"], r["layer"])
                 for r in rows if f"ratio_d{dose}" in r]
        if len(cells) < 4:
            continue
        att = [s for s, lab, _, _ in cells if not lab]
        rot = [s for s, lab, _, _ in cells if lab]
        if att and rot:
            thr = (max(rot) + min(att)) / 2
            acc = sum(1 for s, lab, _, _ in cells if (s < thr) == lab) / len(cells)
            print(f"Dosis {dose}: n_att={len(att)} n_rot={len(rot)} | "
                  f"sigma_c={thr:.4f} | Accuracy={acc:.3f} | "
                  f"max(rot)={max(rot):.4f} min(att)={min(att) if att else float('nan'):.4f}")
        else:
            print(f"Dosis {dose}: nur eine Klasse (att={len(att)}, rot={len(rot)})")
    print("Gespeichert: results/lia_matrix.csv")


if __name__ == "__main__":
    main()
