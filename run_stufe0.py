import torch

from dlora.benchmark import build_methods, run_all
from dlora.pairs import sample_pairs_1d

TRAIN_B = [round(-0.9 + 0.2 * k, 1) for k in range(10)]
EVAL_B = [-0.9 + 0.1 * k for k in range(19)]


def main():
    g = torch.Generator().manual_seed(0)
    tr = [sample_pairs_1d(2048, b, g) for b in TRAIN_B]
    x1_tr = torch.cat([t[0] for t in tr])
    x2_tr = torch.cat([t[1] for t in tr])
    c_tr = torch.cat([t[2] for t in tr])

    configs = []
    for b in EVAL_B:
        x1, x2, c = sample_pairs_1d(1024, b, g)
        held = " *" if b not in TRAIN_B else ""
        configs.append((f"b={b:+.1f}{held}", x1, x2, c))

    print("=" * 100)
    print("STUFE 0: Mechanismus-Test (1D) | a ~ U(0.8, 1.2), x2 = b * a")
    print(f"Train-b: {TRAIN_B}")
    print(f"* = b war nicht im Training (Generalisationscheck)")
    print("=" * 100)
    methods = build_methods()
    run_all(methods, x1_tr, x2_tr, c_tr, configs, csv_path="results/stufe0.csv",
            compact=True)


if __name__ == "__main__":
    main()
