# D-LoRA — Dynamic Interference Decoupling (Rotation/Attenuation Repair)

**These:** Vortrainierte Transformer enthalten ko-aktivierte Pfadpaare
$(h_A, h_B)$, deren Vektorsumme kollabiert ($\cos\theta \to -1$). Das Wissen
ist in den Komponenten vorhanden, aber im Weiterleitungspfad nicht mehr
zugreifbar. D-LoRA untersucht Messung, Kausalanalyse und Reparatur dieser
Kollabiere auf gefrorenen Modellen.

> **⚠ Korrektur v1.1:** Die ursprüngliche KL-Berechnung
> $\sum_i p^0_i(\log p^0_i - \log p^1_i)$ litt bei großen Logits
> ($|z| \approx 330$ in GPT-2) unter katastrophaler Auslöschung — absolute
> Werte „im Millionenbereich" und negative KLs waren numerische Artefakte.
> Alle KL-Messungen wurden auf die stabile Form $CE - H$ (zwei getrennte
> Summen, float64) umgestellt und neu gemessen. Korrigierte Befunde siehe
> §3–§4; die Trainings- und Synthetik-Ergebnisse (§5, §6) nutzen
> Cross-Entropy/MSE und sind nicht betroffen.

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

## 3. Kausaler Schaden (Injektion in echte Modelle — korrigierte Numerik)

Stabile KL (float64, $CE-H$), paired sign test, $n = 149$, Pythia-70m:

| Layer | Dosis | $KL$(Kancellation) | $KL$(Rotation) | sign-z | Fazit |
|---|---|---|---|---|---|
| L3 (isotrop-mitte) | 0.05 | 2.31 | 1.16 | **+10.6** | Kancellation schädigt mehr |
| L3 | 0.20 | 2.78 | 1.31 | **+10.1** | Kancellation schädigt mehr |
| L3 | 0.40 | 2.05 | 1.22 | **+7.8** | Kancellation schädigt mehr |
| L5 (Kegel) | 0.05 | 23.5 | 14.3 | **+12.2** | Kancellation schädigt mehr |
| L5 | 0.20 | 4.79 | 10.62 | **−11.6** | Rotation schädigt mehr |
| L5 | 0.40 | 0.99 | 6.38 | **−12.2** | Rotation schädigt mehr |

**Befunde (überleben die Numerik-Korrektur):**
- In isotropen Mittellayern schädigt die destruktive Addition signifikant
  mehr als die norm-gleiche orthogonale — bei allen Dosen.
- In der Kegel-Layer (L5) kehrt sich der Effekt dosisabhängig um: tiefe
  Auslöschung ist katastrophal, partielle Auslöschung ist *schonender* als
  ihre Rotation.
- Kapazität *nach* der Interferenz recovert nichts (12.5k-Parameter-MLP auf
  der Summe ≈ Baseline; 0-Parameter-$R^\ast$-Refit +0.53 der Lücke —
  Accuracy-Metrik, von der KL-Korrektur unberührt).

## 4. LIA & die Dosis-Abhängigkeit (korrigierte Matrix, 30 Zellen)

$$
\mathrm{LIA}(l) \;=\; \Big|\Big\langle \widehat{R_1 q},\; v_{\max}\!\big(W_U^\top W_U\big)\Big\rangle\Big|
$$

Stabile Ratios $KL(R_1q)/KL(sq)$ bei Dosis 0.2: **> 1 in allen 30 Zellen**
(alle drei Modelle) — die ungedrehte Kancellation liegt näher an der Baseline
als ihre Vollrotation. Rotation als Reparatur ist nur in den
**Kegel-Layern bei kleinen Dosen** vorteilhaft:

| Zelle | ratio bei $\rho^\ast$ = 0.05 / 0.1 / 0.2 / 0.4 |
|---|---|
| pythia-70m L5 (σ=0.102) | **0.77** / 0.90 / 1.41 / 1.78 |
| pythia-160m L11 (σ=0.111) | **0.79** / 0.89 / 1.32 / 1.67 |
| gpt2 L11 (σ=0.089) | **0.83** / 0.85 / 0.95 / 1.13 |

- Die **Kegel-Layer (höchste LIA, letzte Layer)** sind die einzigen Zellen
  mit rotations-Freundlichkeit — und kippen mit steigender Dosis.
- Die ursprüngliche Schwellen-Geschichte („$\sigma_c \approx 0.089$,
  Accuracy 1.000") war ein Artefakt der Naiv-KL und ist **zurückgezogen**;
  σ_c-Accuracy fiel bei stabiler Numerik auf 0.17–0.58.
- LIA bleibt ein **Schwere-Indikator**: die höchste LIA-Layer hat stets die
  stärkste Rotations-Strafe (pythia-70m L5: ratio 1.41 vs. 1.13–1.23
  übrige) und das größte Dosis-Flip.

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
Trainings-Beschleuniger. (Metrik: Cross-Entropy — von der KL-Korrektur
unbetroffen.)

## 6. Weitere belegte Ergebnisse

- **Negativ-Abgrenzung:** Kapazität nach der Interferenz nutzlos
  (Accuracy-Metrik); Mehrpfad-Rotation rettet nur katastrophale Fälle.
- **Strukturierte Antipodie:** Pythia L4 zeigt signifikanten Exzess
  ($\Delta P = +0.030 \pm 0.006$ vs. gematchter Isotropie-Null);
  Mean-Shift-Bias allein stirbt bei $d \to \infty$ (ΔP: +0.147 bei d=16 →
  0.000 bei d ≥ 64).
- **XOR-Smoke-Test (Stufe 0):** antipodales Substrat $\cos(h_1,h_2) = -1.000$
  → Baseline 0.695, D-LoRA-Entflechtung 1.000 (Test-Accuracy).
- **Stufen 1–2 & Verifikation (Q1–Q5):** Ridge-Probes, Accuracy und
  MSE-Metriken — von der KL-Korrektur unbetroffen; korrigiert wurden
  ausschließlich die KL-basierten Kausal-/β-Sweep-Auswertungen.

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
run_lia_matrix.py    LIA × Dosis-Matrix, stabile KL (alle Layer/Modelle)
run_kausal.py        kausaler Brückentest (stabile KL, Pythia/GPT-2)
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

1. GPT-2-L4/L6/L10-Marginalia: mit stabiler KL alle im Rotations-Regime —
   ein echtes GPT-2-Kegel-Regime wurde nicht gefunden (L11-Daten nun stabil).
2. Pythia-410m/1b: bleibt die letzte-Layer-Kegel-Layer mit Skalierung?
3. Reparatur-Metrik jenseits „KL zur (möglicherweise gestörten) Baseline":
   Counterfactual-Vergleich (Partner abwesend) oder Task-Metrik.
4. Ursprung des Layer-4-Exzess ($+4.8\sigma$) — Token-Klassen, Köpfe, Skalierung.

*Changelog: v1.1 — stabile KL (CE−H) überall; LIA-Schwellenbehauptung
zurückgezogen; gpt2-L6/L10-Labels korrigiert; kausale Vorzeichen bestätigt.*
*Daten: `results/*.csv`. Kern-Skripte deterministisch (Seeds fixiert).*
