import csv
import math
import os
import sys

import torch

from dlora.latents import load_corpus
from dlora.verification import r_star_apply, unit_rows

DOSES = [0.05, 0.2, 0.4]
N_CHUNKS = 150


def load_by_name(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if name == "gpt2":
        model_name, layers = "gpt2", [2, 6, 10]
    else:
        model_name, layers = "EleutherAI/pythia-70m", [1, 3, 5]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tok, layers


@torch.no_grad()
def patched_logits(model, layer, chunks, deltas, device, capture=None):
    cap = {}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if capture is not None:
            cap["h"] = h.detach().float().cpu()
        if deltas is not None:
            h = h.clone()
            idx = torch.arange(h.shape[0], device=h.device)
            h[idx, -1] = h[idx, -1] + deltas.to(h.dtype)
        return h

    hand = layer_module.register_forward_hook(hook)
    try:
        logits = model(chunks, logits_to_keep=1).logits[:, -1, :].float()
    except TypeError:
        logits = model(chunks).logits[:, -1, :].float()
    hand.remove()
    return logits, cap.get("h")


def kl_stable(z0, z1):
    """KL(p0 || p1) in float64, numerisch stabil (CE - H, zwei getrennte Summen).

    Die naive Form sum(p0 * (l0 - l1)) leidet bei |logits| > ~300 unter
    katastrophaler Ausloeschung ( Vorzeichen-Artefakte bis -1e7 ).
    """
    ls0 = z0.double() - torch.logsumexp(z0.double(), dim=-1, keepdim=True)
    ls1 = z1.double() - torch.logsumexp(z1.double(), dim=-1, keepdim=True)
    p0 = ls0.exp()
    ce = -(p0 * ls1).sum(-1)
    H = -(p0 * ls0).sum(-1)
    return ce - H


def solve_s(a, b, m, rho):
    A = b ** 2 * (1 - rho ** 2)
    B = 2 * (m - rho ** 2 * a * b)
    C = a ** 2 * (1 - rho ** 2)
    disc = max(B ** 2 - 4 * A * C, 0.0) ** 0.5
    roots = [(-B - disc) / (2 * A), (-B + disc) / (2 * A)]
    pos = [r for r in roots if r > 0]
    return min(pos) if pos else None


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "pythia"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Modell: {name} | Device: {device}")
    torch.manual_seed(0)
    text = load_corpus()
    model, tok, layers = load_by_name(name)
    model = model.to(device).eval()
    d_model = model.config.hidden_size
    ids = tok.encode(text)[: N_CHUNKS * 128 + 1]
    chunks = torch.tensor([ids[k * 128:(k + 1) * 128] for k in range(N_CHUNKS)],
                          device=device)
    targets = torch.tensor([ids[(k + 1) * 128] for k in range(N_CHUNKS)], device=device)
    layer_modules = (model.transformer.h if hasattr(model, "transformer")
                     else model.gpt_neox.layers)

    g = torch.Generator().manual_seed(42)
    rows = []
    cond_names = ["raw"] + [f"{kind}{rho_t}" for rho_t in DOSES for kind in ("C", "B")]
    for layer in layers:
        lb, H_full = forward_with_patch(model, layer_modules[layer], chunks, None,
                                        device, capture=True)
        H_last = H_full.reshape(-1, d_model)
        lb_sm = torch.log_softmax(lb, dim=1)
        base_arg = lb.argmax(1)
        Hn = unit_rows(H_last)
        keep_idx, base_deltas, cos_list = [], [], []
        for k in range(N_CHUNKS):
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
            p = q
            a, b, m = h.norm().item(), p.norm().item(), float(h @ p)
            if m >= 0.0:
                continue
            deltas = {"raw": p.clone()}
            ok = True
            for rho_t in DOSES:
                s = solve_s(a, b, m, rho_t)
                if s is None:
                    ok = False
                    break
                deltas[f"C{rho_t}"] = s * p
                rot = r_star_apply(p.unsqueeze(0), h.unsqueeze(0), p.unsqueeze(0),
                                   g).squeeze(0)
                deltas[f"B{rho_t}"] = (s * b) * unit_rows(rot.unsqueeze(0)).squeeze(0)
            if not ok:
                continue
            keep_idx.append(k)
            base_deltas.append(deltas)
            cos_list.append(m / (a * b))
        n = len(keep_idx)
        if n < 30:
            print(f"Layer {layer}: nur {n} Positionen, uebersprungen")
            continue
        sel_chunks = chunks[keep_idx]
        sel_targets = targets[keep_idx]
        res = {nm: {"kl": [], "flip": [], "logp": []} for nm in cond_names}
        for start in range(0, n, 50):
            end = min(start + 50, n)
            for nm in cond_names:
                d_batch = torch.stack([base_deltas[i][nm]
                                       for i in range(start, end)]).to(device)
                lc, _ = forward_with_patch(model, layer_modules[layer],
                                           sel_chunks[start:end], d_batch, device)
                lc_sm = torch.log_softmax(lc, dim=1)
                loc = torch.arange(end - start, device=lc_sm.device)
                gidx = torch.arange(start, end, device=lc_sm.device)
                res[nm]["kl"].append(kl_stable(lb[gidx], lc).cpu())
                res[nm]["flip"].append((lc.argmax(1) != base_arg[gidx]).float().cpu())
                res[nm]["logp"].append(lc_sm[loc, sel_targets.to(lc_sm.device)[gidx]].cpu())
        for nm in cond_names:
            for key in res[nm]:
                res[nm][key] = torch.cat(res[nm][key])
        print(f"\nLayer {layer} | n={n} | mean cos(h,p) = {sum(cos_list)/n:+.3f}")
        print(f"{'Kondition':>10} | {'KL':>8} {'Flip':>7} {'logp_true':>10}")
        for nm in cond_names:
            print(f"{nm:>10} | {res[nm]['kl'].mean():>8.3f} "
                  f"{res[nm]['flip'].mean():>7.3f} {res[nm]['logp'].mean():>10.3f}")
        for rho_t in DOSES:
            kl_c, kl_b = res[f"C{rho_t}"]["kl"], res[f"B{rho_t}"]["kl"]
            dkl = kl_c - kl_b
            ci = 1.96 * dkl.std(unbiased=True).item() / math.sqrt(n)
            zf = (dkl > 0).float().mean().item()
            zscore = (zf - 0.5) / (0.5 / math.sqrt(n))
            rows.append(dict(model=name, layer=layer, dose=rho_t, n=n,
                             mean_cos=sum(cos_list) / n,
                             kl_c=kl_c.mean().item(), kl_b=kl_b.mean().item(),
                             dkl=dkl.mean().item(), ci=ci, z_sign=zscore,
                             flip_c=res[f"C{rho_t}"]["flip"].mean().item(),
                             flip_b=res[f"B{rho_t}"]["flip"].mean().item(),
                             logp_c=res[f"C{rho_t}"]["logp"].mean().item(),
                             logp_b=res[f"B{rho_t}"]["logp"].mean().item()))
            print(f"  C vs B (Dosis {rho_t}): dKL = {dkl.mean():+.3f} ± {ci:.3f} "
                  f"| Vorzeichen-z = {zscore:+.2f} | Flip C/B = "
                  f"{res[f'C{rho_t}']['flip'].mean():.3f}/{res[f'B{rho_t}']['flip'].mean():.3f}")

    os.makedirs("results", exist_ok=True)
    fname = f"results/kausal_{name}.csv"
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "layer", "dose", "n", "mean_cos",
                                          "kl_c", "kl_b", "dkl", "ci", "z_sign",
                                          "flip_c", "flip_b", "logp_c", "logp_b"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nGespeichert: {fname}")


if __name__ == "__main__":
    main()
