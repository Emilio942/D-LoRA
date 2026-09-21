import argparse
import csv
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS = os.path.join(".cache", "corpus.txt")
CTX = 128
D = 256
NLAYERS = 4
HEADS = 4
BATCH = 32
STEPS = 6000
EVAL_EVERY = 250
GATE_TAU = 0.25
GATE_GAMMA = 0.25
BLOCK_LEVEL = False


class Block(nn.Module):
    def __init__(self, d, heads, mode):
        super().__init__()
        self.mode = mode
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)
        self.heads = heads
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self._fire_sum = 0.0
        self._fire_cnt = 0

    def reset_fire(self):
        self._fire_sum = 0.0
        self._fire_cnt = 0

    def fire_rate(self):
        return self._fire_sum / max(self._fire_cnt, 1)

    def _gate(self, h, delta):
        if self.mode == "baseline":
            return delta
        rho = ((h + delta).norm(dim=-1)
               / (h.norm(dim=-1) + delta.norm(dim=-1)).clamp_min(1e-8))
        mask = (rho < GATE_TAU).float().unsqueeze(-1)
        self._fire_sum += mask.mean().item()
        self._fire_cnt += 1
        if self.mode == "attenuate":
            return GATE_GAMMA * delta * mask + delta * (1 - mask)
        if self.mode == "rotate":
            proj = (delta * h).sum(-1, keepdim=True) / h.norm(dim=-1, keepdim=True) ** 2
            d_orth = delta - proj * h
            scale = (delta.norm(dim=-1, keepdim=True)
                     / d_orth.norm(dim=-1, keepdim=True).clamp_min(1e-8))
            d_rot = d_orth * scale
            return GATE_GAMMA * d_rot * mask + delta * (1 - mask)
        return delta

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = torch.cat([self.q(h), self.k(h), self.v(h)], dim=-1)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.heads)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(causal, float("-inf"))
        att = torch.softmax(att, dim=-1)
        a = self.proj((att @ v).transpose(1, 2).reshape(B, T, C))
        x1 = x + a
        m = self.fc2(F.gelu(self.fc1(self.ln2(x1))))
        if BLOCK_LEVEL and self.mode != "baseline":
            d = x1 + m - x
            gated = self._gate(x, d)
            return x + gated
        x1 = x1 + self._gate(x1, m)
        return x1


class TinyLM(nn.Module):
    def __init__(self, vocab, mode):
        super().__init__()
        self.emb = nn.Embedding(vocab, D)
        self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList([Block(D, HEADS, mode) for _ in range(NLAYERS)])
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.lnf(x))


def get_data():
    text = open(CORPUS, "r", encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = len(data)
    return data[: int(0.9 * n)], data[int(0.9 * n):], len(chars)


def batch(data, g, device):
    ix = torch.randint(0, len(data) - CTX - 1, (BATCH,), generator=g)
    x = torch.stack([data[i:i + CTX] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + CTX + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def evaluate(model, val, g, device, nb=100):
    model.eval()
    losses = []
    for _ in range(nb):
        x, y = batch(val, g, device)
        logits = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                      y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    global GATE_TAU, GATE_GAMMA, BLOCK_LEVEL
    BLOCK_LEVEL = os.environ.get("BLOCK_LEVEL", "0") == "1"
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--tau", type=float, default=GATE_TAU)
    ap.add_argument("--gamma", type=float, default=GATE_GAMMA)
    args = ap.parse_args()
    GATE_TAU, GATE_GAMMA = args.tau, args.gamma
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    train, val, vocab = get_data()
    model = TinyLM(vocab, args.mode).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    g = torch.Generator().manual_seed(args.seed + 100)
    log = []
    os.makedirs("results", exist_ok=True)
    tag = f"{args.mode}_s{args.seed}"
    print(f"[{tag}] device={device} params={nparams/1e6:.2f}M vocab={vocab}")
    for step in range(1, args.steps + 1):
        x, y = batch(train, g, device)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % EVAL_EVERY == 0 or step == 1:
            vl = evaluate(model, val, g, device)
            fire = sum(b.fire_rate() for b in model.blocks) / NLAYERS
            for b in model.blocks:
                b.reset_fire()
            log.append(dict(mode=args.mode, seed=args.seed, step=step,
                            train_loss=loss.item(), val_loss=vl, fire_rate=fire))
            print(f"[{tag}] step {step:5d} | train {loss.item():.4f} "
                  f"| val {vl:.4f} | fire {fire:.4f}")
    with open(f"results/train_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "seed", "step", "train_loss",
                                          "val_loss", "fire_rate"])
        w.writeheader()
        w.writerows(log)
    print(f"[{tag}] FERTIG | val final: {log[-1]['val_loss']:.4f} | "
          f"val min: {min(l['val_loss'] for l in log):.4f} | "
          f"fire final: {log[-1]['fire_rate']:.4f}")


if __name__ == "__main__":
    main()
