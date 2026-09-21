# D-LoRA — Dynamic Interference Decoupling (Rotation/Attenuation Repair)

**These:** Vortrainierte Transformer enthalten ko-aktivierte Pfadpaare
$(h_A, h_B)$, deren Vektorsumme kollabiert ($\cos\theta \to -1$). Das Wissen
ist in den Komponenten vorhanden, aber im Weiterleitungspfad nicht mehr
zugreifbar. D-LoRA ist ein Inferenz-Reparaturwerkzeug: ein
Konflikt-detektierendes Gate ($\rho$, LIA) plus normerhaltende Rotation
(isotrope Layer) bzw. Dämpfung (Kegel-Layer).

**Zentraler Befund:** Das Phänomen ist ein *Inferenz*-Problem gefrorener
Modelle. Beim Training tritt destruktive Kancellation nicht auf (gemessen,
siehe §5) — das Gate feuert dort korrekt nie.

---

## 1. Konfliktgrößen

Auslöschungs-Ratio (0 = totale Kancellation, 1 = keine Überlappung):

$$
\rho(h, q) \;=\; \frac{\|h + q\|}{\|h\| + \|q\|}
\;=\; \frac{\sqrt{1 + r^2 + 2rc}}{1 + r},
\qquad c = \frac{h^\top q}{\|h\|\,\|q\|},\;\; r = \frac{\|q\|}{\|h\|}
$$

Konflikt-Detektor mit kalibrierter Fehlalarmrate: Schwelle $\tau_\rho$ =
$\alpha$-Quantil der $\rho$-Nullverteilung (isotrope Paare, gleiche Normen);
$\alpha = 0.01$ empirisch exakt getroffen. Reine Winkel-Schwellen $c < \tau_c$
scheitern bei Norm-Ungleichgewicht (Präzision 1.000 bei $r{=}1$, 0.000 bei
$r{=}4$) — $\rho(c,r)$ ist notwendig.

## 2. Reparatur-Operator

Planare Givens-Rotation, die $q$ um den Anteil $\beta$ Richtung Orthogonalität
zu $h$ dreht (exakt normerhaltend, Rest $< 10^{-6}$):

$$
h' \;=\; h + \gamma\, R_\beta(q), \qquad
h^\top R_\beta(q) \xrightarrow{\beta \to 1} 0, \qquad
\big|\|R_\beta q\| - \|q\|\big| < 10^{-6}.
$$

$\gamma$ = Dämpfungsanteil. Beide Operationen sind closed-form (Givens/Ebene
$\mathrm{span}\{h, q\}$), **null trainierte Parameter**.

## 3. Kausaler Schaden (Injektion in echte Modelle)

In Pythia-70m / GPT-2 (Residual-Pfad, letzte Position, echte Textaktivierungen):
erzwungene Auslöschung $\rho^\ast$ schädigt die Vorhersage signifikant mehr als
die **norm-gleiche orthogonale** Störung — Signifikanz $z = +6.7$ bis $+10.9$,
dosis-monoton. Bei $\rho^\ast{=}0.2$ schlägt Rotation die Baseline dort, wo die
geometrische Entkopplung vor der Summation wirkt; Kapazität *nach* der
Interferenz (12.5k-Parameter-MLP) recovert nichts (0.507 vs. Baseline 0.498,
Oracle 0.961).

## 4. LIA — die Schalt-Größe (volle Matrix, 36 Zellen)

$$
\mathrm{LIA}(l) \;=\; \Big|\Big\langle \widehat{R_1 q},\; v_{\max}\!\big(W_U^\top W_U\big)\Big\rangle\Big|
$$

(Alignment des voll rotierten Partners mit der dominierenden
Logit-Richtung). Messung über alle Layer × Dosen $\rho^\ast \in \{0.05, 0.1,
0.2, 0.4\}$:

| Modell | $\sigma_c$ | Accuracy | Attenuation-Layer | Dosis-Kippung |
|---|---|---|---|---|
| pythia-70m | **0.0898** | 1.000 (Dosis 0.2/0.4) | L5 (letzte) | ratio 0.25 → 8.87 |
| pythia-160m | **0.0891** | 1.000 (Dosis 0.2/0.4) | L11 (letzte) | ratio 0.30 → 6.02 |
| gpt2 | trennt nicht | — | keine signifikante | max ratio 1.09 |

- $\sigma_c \approx 0.089$ ist **skalenstabil** innerhalb der Pythia-Familie
  (70m/160m innerhalb 1 %); das Attenuation-Regime ist exakt die letzte Layer
  (direkt vor dem Unembedding).
- Dosis-Flip: in der Kegel-Layer kehrt sich der Vorteil zwischen
  $\rho^\ast \approx 0.1$ und $0.2$ um (Rotation hilfreich bei tiefer
  Auslöschung, katastrophal bei partieller).
- **Falsifiziert:** globale Spektralkonzentration ($d_{\text{eff}}$, stable
  rank, Spektral-Entropie), Partner-Mittelkosinus, Fisher-Krümmungs- und
  Entropie-Taylor-Formen — keine trennt die Layer (Cycle-Audit, 13 Kandidaten,
  `results/cycle_audit_*.csv`).

### Zwei-Modes-Politik (empirisch)

$$
\beta^\ast \;=\;
\begin{cases}
1 & \mathrm{LIA} < \sigma_c \ \text{(isotrop: rotieren)}\\[2pt]
0,\ \gamma = 0.25 & \mathrm{LIA} \geq \sigma_c \ \text{(Kegel: dämpfen)}
\end{cases}
$$

In der Kegel-Layer (Pythia L5/L11) ist Dämpfung die einzige Intervention mit
$KL \approx 0$; Rotation ist dort 4× schlimmer als das Problem selbst.

## 5. Trainings-Deployment: refuted

Char-Level-Transformer (4 Layer, d=256, 3.2M Parameter), 10 Läufe, Baseline
vs. ρ-gegate Dämpfung vs. Rotation (Update- und Block-Ebene, $\tau \in
\{0.25, 0.4\}$):

| Bedingung | val final (Seed 0/1) | Feuerrate |
|---|---|---|
| Baseline | 1.589 / 1.559 | – |
| Dämpfung | 1.596 / 1.555 | 0.0000 |
| Rotation | 1.586 / 1.557 | 0.0000 |

**Gemessen:** Die Kancellations-$\rho$ der Residual-Updates liegt während des
Trainings bei ≈ 0.90 (p05 = 0.81) — destruktive Auslöschung ($\rho < 0.4$)
tritt **nie** auf. Das Gate feuert korrekt nie; Trainingsergebnisse sind
statistisch identisch. D-LoRA ist ein Inferenz-Werkzeug, kein
Trainings-Beschleuniger.

## 6. Weitere belegte Ergebnisse

- **Negativ-Abgrenzung:** Kapazität nach der Interferenz nutzlos;
  Mehrpfad-Rotation rettet nur katastrophale Fälle (ortho-aler Floor wird
  erreicht, Funktionale bleiben orientierungsgebunden); LIA als
  Universalschwelle über alle Modelle refuted (GPT-2-Ausnahme dokumentiert).
- **Strukturierte Antipodie:** Pythia L4 zeigt signifikanten Exzess
  ($\Delta P = +0.030 \pm 0.006$ vs. gematchter Isotropie-Null);
  Mean-Shift-Bias allein stirbt bei $d \to \infty$ (ΔP: +0.147 bei d=16 →
  0.000 bei d ≥ 64) — der Exzess ist also strukturell, nicht geometrisch.
- **XOR-Smoke-Test (Stufe 0):** antipodales Substrat $\cos(h_1,h_2) = -1.000$
  → Baseline 0.695, D-LoRA-Entflechtung 1.000 (Test-Accuracy).

## 7. Struktur & Ausführung

```text
dlora/            Kern-Bibliothek
  pairs.py          kontrollierte Paar-Generatoren (Winkel, Normen, 1D)
  methods.py        Baseline / Referenz / GatedDiffAdapter
  evaluate.py       Ridge-Probes, R², Konflikt-Probes
  benchmark.py      harness (alle Methoden × alle Zellen)
  verification.py   Q1–Q5-Mathematik (ρ, Ψ, R*, Intervention)
  latents.py        Pythia/GPT-2-Hooks, Partner-Mining
  model.py, train.py, adapter.py   XOR-Smoke-Test (Stufe 0)
run_stufe0.py     1D-Mechanismus-Test (b-Sweep)
run_stufe1.py     R^d: neue Winkel × Dimensionen 16→512
run_stufe2.py     echte Latents (Pythia Layer 0–5)
run_verification.py  Q1–Q5-Protokoll (Null-Matching, R*, A/B)
run_lia_matrix.py    LIA-Gesetz: alle Layer × 4 Dosen × 3 Modelle
run_kausal.py        kausaler Brückentest (Injektion, Pythia/GPT-2)
run_beta_sweep.py    β-Kurve (Rotation anteilig)
run_training_ab.py   Trainings-A/B (Baseline/Dämpfung/Rotation)
run_cycle_audit.py   Audit der Pipeline-Hypothesen
results/             alle Messdaten (CSV) + Berichte
```

```bash
python -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install transformers numpy
.venv/bin/python run_verification.py     # Q1–Q5
.venv/bin/python run_lia_matrix.py pythia-70m
.venv/bin/python run_training_ab.py --mode baseline --seed 0
```

GPU (CUDA) optional; alle Skripte laufen auch auf CPU.

## 8. Offene Punkte

1. GPT-2-L11-Anomalie (negative KLs = Numerik oder echtes Verhalten).
2. Pythia-410m/1b: verschwindet die letzte-Layer-Kegel-Layer mit Skalierung?
3. Inferenz-Deployment: Gate in einen echten Serve-Pfad, Task-Metrik statt KL.
4. Ursprung des Layer-4-Exzess ($+4.8\sigma$) — Token-Klassen, Köpfe, Skalierung.

*Daten: `results/*.csv`. Kern-Skripte deterministisch (Seeds fixiert).*
