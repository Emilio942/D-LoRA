from dlora.adapter import DLoRAAdapter
from dlora.conflict import conflict_score, neuron_cosine_matrix
from dlora.dataset import CLASS_NAMES, make_xor_dataset, train_test_split
from dlora.model import BaselineNet, DLoRANet, Substrate
from dlora.train import accuracy, per_class_accuracy, train


def report(title, model, x_tr, y_tr, x_te, y_te):
    C = neuron_cosine_matrix(model.hidden(x_tr))
    print(f"\n--- {title} ---")
    print(f"Konflikt der Hidden-Neuronen: cos(h1, h2) = {C[0, 1].item():+.3f}")
    print(f"Accuracy (Test): {accuracy(model, x_te, y_te):.3f}")
    for name, a in zip(CLASS_NAMES, per_class_accuracy(model, x_te, y_te)):
        print(f"  Klasse {name}: {a:.3f}")


def main():
    x, y = make_xor_dataset()
    x_tr, y_tr, x_te, y_te = train_test_split(x, y)
    print(f"Datensatz: {len(x)} Samples | 4 Klassen | {len(x_tr)} Train / {len(x_te)} Test")
    print("Klassen:", " | ".join(CLASS_NAMES))

    substrate = Substrate()
    h = substrate(x_tr)
    print(f"\nSubstrat nach dem Bau: cos(h1, h2) = {conflict_score(h):+.3f}"
          "  <- die beiden Neuronen neutralisieren sich exakt")

    print("\n[1] Baseline: Head-Training auf kollidierendem Substrat")
    substrate.freeze()
    base = BaselineNet(substrate)
    train(base, x_tr, y_tr, base.head.parameters(), epochs=8000)
    report("Baseline (Neuronen in Interferenz)", base, x_tr, y_tr, x_te, y_te)

    print("\n[2] D-LoRA: Entflechtungs-Fine-Tuning mit Konfliktverlust")
    substrate.unfreeze()
    model = DLoRANet(substrate, DLoRAAdapter(dim=2))
    params = (list(model.substrate.parameters())
              + list(model.adapter.parameters())
              + list(model.head.parameters()))
    train(model, x_tr, y_tr, params, epochs=8000, lambda_conflict=2.0)
    report("D-LoRA (Neuronen entflochten)", model, x_tr, y_tr, x_te, y_te)


if __name__ == "__main__":
    main()
