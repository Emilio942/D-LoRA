import torch

from dlora.benchmark import build_methods, gate_report, run_all
from dlora.pairs import sample_pairs

TRAIN_D = 16
TRAIN_ANGLES = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
EVAL_ANGLES = [-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875]
EVAL_DIMS = [16, 64, 512]


def main():
    g = torch.Generator().manual_seed(0)
    tr = [sample_pairs(2048, TRAIN_D, a, g) for a in TRAIN_ANGLES]
    x1_tr = torch.cat([t[0] for t in tr])
    x2_tr = torch.cat([t[1] for t in tr])
    c_tr = torch.cat([t[2] for t in tr])
    print(f"Training: d={TRAIN_D}, n={len(x1_tr)}, Winkel-Grid {TRAIN_ANGLES}\n")

    methods = build_methods()
    print("Adapter-Training (Summe/Differenz-Gates, gesteuerter Konfliktschaetzer):")
    methods[2].fit(x1_tr, x2_tr, c_tr, verbose=True)
    methods[0].fit(x1_tr, x2_tr, c_tr)
    methods[1].fit(x1_tr, x2_tr, c_tr)
    gate_report(methods[2])

    configs = []
    calibrations = {}
    for d in EVAL_DIMS:
        cal = [sample_pairs(512, d, a, g) for a in TRAIN_ANGLES]
        calibrations[d] = (
            torch.cat([t[0] for t in cal]),
            torch.cat([t[1] for t in cal]),
            torch.cat([t[2] for t in cal]),
        )
        for a in EVAL_ANGLES:
            x1, x2, c = sample_pairs(1024, d, a, g)
            configs.append((f"d={d:3d}, cos={a:+.3f}", x1, x2, c))

    print()
    print("=" * 118)
    print("STUFE 1: kontinuierliche Vektoren | neue Winkel (nie im Training) x neue Dimensionen")
    print("=" * 118)
    run_all(methods, x1_tr, x2_tr, c_tr, configs, csv_path="results/stufe1.csv",
            calibrations=calibrations)


if __name__ == "__main__":
    main()
