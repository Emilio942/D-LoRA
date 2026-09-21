import torch

from dlora.evaluate import eval_probes, fit_probes
from dlora.latents import (collect_activations, load_corpus, load_model,
                           mine_pairs, relative_noise)
from dlora.methods import DiffReference, GatedDiffAdapter, SumBaseline

LAYERS = [0, 1, 2, 3, 4, 5]
MAIN_LAYER = 3
BIN_EDGES = [-1.0, -0.4, -0.2, 0.0, 0.2, 0.4, 0.7, 1.001]


def build_split(H, generator):
    fit_pool = H[:3000]
    eval_pool = H[3000:4000]
    tr = mine_pairs(fit_pool, fit_pool, n_random=768, n_conflict=512, generator=generator)
    te = mine_pairs(eval_pool, eval_pool, n_random=384, n_conflict=384, generator=generator)
    return tr, te


def cos_stats(name, c):
    print(f"  {name}: n={len(c)} | mean={c.mean():+.3f} | min={c.min():+.3f} | max={c.max():+.3f}")


def main():
    torch.manual_seed(0)
    text = load_corpus()
    model, tok = load_model()
    acts = collect_activations(model, tok, text, LAYERS, max_tokens=4000)
    d = acts[0].shape[1]
    print(f"\nAktivierungen: Layer {LAYERS} | {acts[0].shape[0]} Token/Layer | d={d}")

    splits = {l: build_split(acts[l], torch.Generator().manual_seed(7)) for l in LAYERS}
    (x1_tr, x2_tr, c_tr), (x1_te, x2_te, c_te) = splits[MAIN_LAYER]
    noise = relative_noise(acts[MAIN_LAYER])
    print(f"Rauschen (rel. 1% der mittleren Norm, Layer {MAIN_LAYER}): {noise:.3f}")

    print("\nNatuerliche Kosinus-Verteilung der Paare (Layer 3):")
    cos_stats("Fit  (zufaellig+Konflikt)", c_tr)
    cos_stats("Eval (zufaellig+Konflikt)", c_te)

    print(f"\n{'=' * 112}")
    print(f"EXPERIMENT 2a: Interferenz-Bins auf echten Latents (Layer {MAIN_LAYER}, d={d})")
    print(f"{'=' * 112}")
    methods = [SumBaseline(), DiffReference(), GatedDiffAdapter()]
    for m in methods:
        m.fit(x1_tr, x2_tr, c_tr, noise=noise)
    y_te = torch.cat([x1_te, x2_te], dim=1)
    bin_ids = torch.bucketize(c_te, torch.tensor(BIN_EDGES[1:-1]))
    rows = []
    for b in range(len(BIN_EDGES) - 1):
        mask = bin_ids == b
        if mask.sum() < 20:
            continue
        xb1, xb2, cb, yb = x1_te[mask], x2_te[mask], c_te[mask], y_te[mask]
        line = (f"cos[{BIN_EDGES[b]:+.2f},{BIN_EDGES[b + 1]:+.2f}) n={int(mask.sum()):4d} "
                f"| cos_mean={cb.mean():+.3f}")
        for m in methods:
            zb, _ = m.forward(xb1, xb2)
            zb = zb + noise * torch.randn_like(zb)
            res = eval_probes(m.w_dec, m.c_probe, zb, yb, cb)
            rows.append((BIN_EDGES[b], m.name, res))
            line += f" | {m.short}: MSE={res['recon_mse']:.4f}, R2={res['r2_mean']:.3f}"
        z_a, c_hat = methods[2].forward(xb1, xb2)
        line += f" | cMAE(int)={(c_hat - cb).abs().mean():.3f}"
        print(line)
    za_all, cha_all = methods[2].forward(x1_te, x2_te)
    ca, cb2 = cha_all - cha_all.mean(), c_te - c_te.mean()
    print(f"\nAdapter c_hat vs. gemessener cos (gepoolt, Eval-Paare): "
          f"corr={((ca * cb2).sum() / (ca.norm() * cb2.norm() + 1e-12)).abs():.3f}, "
          f"MAE={(cha_all - c_te).abs().mean():.3f}")

    print(f"\n{'=' * 112}")
    print("EXPERIMENT 2b: Layer-Transfer (Adapter + Probes auf Layer 3 trainiert, Eval auf Layer 0-5)")
    print(f"{'=' * 112}")
    print(f"{'Layer':>5s} | {'Methode':<28s} | MSE     | R2     | cMAE(int)")
    for l in LAYERS:
        (a1, a2, ac), (e1, e2, ec) = splits[l]
        nl = relative_noise(acts[l])
        yl_tr, yl_te = torch.cat([a1, a2], 1), torch.cat([e1, e2], 1)
        for m in [methods[0], methods[1], methods[2]]:
            zt, _ = m.forward(a1, a2)
            zt = zt + nl * torch.randn_like(zt)
            w = fit_probes(zt, yl_tr)
            ze, chat = m.forward(e1, e2)
            ze = ze + nl * torch.randn_like(ze)
            res = eval_probes(w, None, ze, yl_te, ec)
            extra = ""
            if chat is not None:
                extra = f" | {(chat - ec).abs().mean():.3f}"
            print(f"{l:>5d} | {m.name:<28s} | {res['recon_mse']:.4f}  | {res['r2_mean']:.3f}  {extra}")


if __name__ == "__main__":
    main()
