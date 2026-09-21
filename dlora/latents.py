import os
import urllib.request

import torch

MODEL_NAME = "EleutherAI/pythia-70m"
CACHE_DIR = ".cache"

FALLBACK_CORPUS = (
    "The sun rises over the quiet valley and the farmers begin their work in the fields. "
    "Machines process language by transforming words into vectors of numbers. "
    "She asked the question twice, but the answer remained the same. "
    "for i in range(10): print(i ** 2) and the loop terminates after the final iteration. "
    "Reality is that which, when you stop believing in it, does not go away. "
    "The committee published its report after months of deliberation and debate. "
)

GUTENBERG_URLS = [
    "https://www.gutenberg.org/files/11/11-0.txt",
    "https://www.gutenberg.org/cache/epub/84/pg84.txt",
]


def load_corpus(cache_path=os.path.join(CACHE_DIR, "corpus.txt"), max_chars=400_000):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()[:max_chars]
    parts = []
    for url in GUTENBERG_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8", errors="ignore")
            start = raw.find("*** START")
            end = raw.find("*** END")
            if start != -1 and end != -1:
                raw = raw[start:end]
            parts.append(raw)
            print(f"Korpus geladen: {url} ({len(raw)} Zeichen)")
        except Exception as e:
            print(f"Warnung: {url} nicht erreichbar ({e})")
    if not parts:
        print("Warnung: Fallback-Korpus verwendet.")
        parts = [FALLBACK_CORPUS * 50]
    text = "\n".join(parts)[:max_chars]
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tok


@torch.no_grad()
def collect_activations(model, tok, text, layers, seq_len=128, batch_size=8,
                        max_tokens=4000):
    ids = tok.encode(text)
    chunks = [ids[i:i + seq_len] for i in range(0, len(ids) - seq_len + 1, seq_len)]
    captured = {l: [] for l in layers}
    hooks = []

    def make_hook(l):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[l].append(h.reshape(-1, h.shape[-1]).float().cpu())
        return hook

    for l in layers:
        hooks.append(model.gpt_neox.layers[l].register_forward_hook(make_hook(l)))
    collected = 0
    for i in range(0, len(chunks), batch_size):
        batch = torch.tensor(chunks[i:i + batch_size])
        model(batch)
        collected += batch.numel()
        if collected >= max_tokens:
            break
    for h in hooks:
        h.remove()
    out = {}
    for l in layers:
        out[l] = torch.cat(captured[l])[:max_tokens]
    return out


def _normalize_rows(H):
    return H / H.norm(dim=1, keepdim=True).clamp_min(1e-8)


def mine_pairs(H_anchor, H_pool, n_random=768, n_conflict=512, generator=None):
    g = generator if generator is not None else torch.Generator()
    na, np_ = H_anchor.shape[0], H_pool.shape[0]
    i_r = torch.randint(0, na, (n_random,), generator=g)
    j_r = torch.randint(0, np_, (n_random,), generator=g)
    a = _normalize_rows(H_anchor)
    b = _normalize_rows(H_pool)
    cos_full = a @ b.T
    i_c = torch.randperm(na, generator=g)[:n_conflict]
    j_c = cos_full[i_c].argmin(dim=1)
    idx1 = torch.cat([i_r, i_c])
    idx2 = torch.cat([j_r, j_c])
    x1 = H_anchor[idx1]
    x2 = H_pool[idx2]
    c = (a[idx1] * b[idx2]).sum(1)
    return x1, x2, c


def relative_noise(H, rel=0.01):
    return (rel * H.norm(dim=1).mean()).item()
