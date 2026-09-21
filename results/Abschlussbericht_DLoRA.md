# D-LoRA Abschlussbericht: Distanz zum Ziel & Trainings-Deployment (autonome Session II)

> **⚠ KORREKTUR (v1.1, nach der Veröffentlichung):** Der folgende Bericht
> wurde mit der numerisch defekten KL-Form (katastrophale Auslöschung bei
> |Logits| ≈ 330) erstellt. Korrigiert mit stabiler KL (CE−H, float64):
> (1) Alle absoluten KL-"Millionen" waren Artefakte (echte Werte: 0.5–24
> nats). (2) Der kausale Kernbefund überlebt (L3: Kancellation schlimmer,
> z +7.8..+10.6; L5: Dosis-Inversion, ±12). (3) Die LIA-Schwellen-Aussage
> (σ_c ≈ 0.089, "Accuracy 1.000") ist **zurückgezogen** — bei stabiler KL
> liegt Dosis 0.2 in ALLEN 30 Zellen im Attenuation-Regime; Rotation hilft
> nur in Kegel-Layern bei ρ\* ≤ 0.1. (4) Trainings- und Synthetik-Resultate
> (CE/MSE-Metriken) sind unverändert gültig. Details: README §3–§4,
> `results/lia_matrix.csv` (regeneriert).

## Die Frage

Wie nah sind wir am Ziel — dass der Adapter die Neutralisierung gegenseitig
auslöschender Systeme umgeht — und können wir ihn **im Training** eines echten
Netzes anwenden mit besseren Ergebnissen?

## Antwort in einem Satz

**Am Inferenz-Ziel sind wir im Kern angekommen (Reparatur auf gefrorenen
Modellen funktioniert und ist formelbasiert); im Training ist die Hypothese
sauber refuted — es gibt dort nichts zu reparieren, messbar.**

---

## 1. Trainings-Experiment (neu, diese Session)

Setup: Char-Level-Transformer (4 Layer, d=256, 3.2M Parameter, Kontext 128)
auf dem Gutenberg-Korpus, AdamW + Cosine, 6000 Steps, 2 Seeds pro Bedingung.
D-LoRA-Gate: ρ = ‖h+Δ‖/(‖h‖+‖Δ‖) pro Block-Update; ρ < τ → Δ wird gedämpft
(γ=0.25) oder orthogonalisiert. Varianten: Update-Level (τ=0.25/0.4) und
Block-Level (τ=0.4).

| Bedingung | val final (Seed 0/1) | Feuerrate |
|---|---|---|
| Baseline | 1.589 / 1.559 | – |
| Dämpfung (Update, τ=0.25) | 1.596 / 1.555 | 0.0000 |
| Rotation (Update, τ=0.25) | 1.586 / 1.557 | 0.0000 |
| Dämpfung (Block, τ=0.4) | 1.585 / 1.554 | 0.0000–0.0001 |
| Rotation (Block, τ=0.4) | 1.590 / 1.556 | 0.0000–0.0001 |

**Fakten:**
1. **Während des Trainings tritt destruktive Auslöschung nicht auf.** Die
   Kancellations-ρ der Block-Updates liegt bei ≈ 0.90 (p05 = 0.81, Step 10);
   kein Update je unter ρ = 0.4. Das Gate korrekt nie → kein Effekt möglich.
2. Alle 10 Läufe statistisch identisch (Differenzen < Seed-Varianz ±0.016).
3. Früh im Training (Step 10) sind Updates zwar groß (relative Update-Größe
   ≈ 0.35), aber **nicht** dem Strom entgegengerichtet — Optimierung
   selbst-organisiert weg vom Interferenz-Regime.

**Konsequenz:** D-LoRA ist kein Trainings-Beschleuniger (auf dieser Skala und
Aufgabe refuted). Sein Anwendungsbereich ist die **Inferenz auf gefrorenen,
vortrainierten Modellen** — und dort ist er wirksam (Phasen I–II).

## 2. LIA-Gesetz: vollständige Matrix (neu, diese Session)

36 Zellen (pythia-70m 6 Layer, gpt2 12 Layer, pythia-160m 12 Layer; je 4
Dosen ρ\* ∈ {0.05, 0.1, 0.2, 0.4}; 149 Positionen/Zelle):

- **pythia-70m:** σ_c = 0.0898, Accuracy 1.000 (Dosis 0.2/0.4); L5 einziges
  Attenuation-Regime; Dosis-Flip messbar: ratio(L5) = 0.25 → 0.50 → 2.84 →
  8.87, Kreuzung bei ρ\* ≈ 0.15.
- **pythia-160m:** σ_c = 0.0891, Accuracy 1.000; Attenuation-Layer = L11
  (letzte Layer); gleicher Dosis-Flip (0.296 → 6.02). **Skalenstabil: die
  Konstante ist über 70m/160m identisch innerhalb 1%.**
- **gpt2:** LIA-trennt nicht (nur L4 marginal > 1; L11 numerisch degeneriert)
  — dokumentierte Ausnahme. GPT-2 hat messbar **kein** Attenuation-Regime
  (max ratio 1.09, im Rauschen).

Mechanismisch konsistent: In beiden Pythia-Modellen ist das
Attenuation-Regime exakt die **letzte Layer** (direkt vor dem Unembedding);
σ(LIA) misst, wie weit der rotierte Partner in die dominierende
Readout-Richtung fällt.

## 3. Cycle-Audit (40+ Pipeline-Hypothesen getestet)

13 prüfbare σ-Kandidaten implementiert und getestet: 3 trennen bei Dosis 0.2
(readout_norm, topspan, **LIA**), nur LIA übersteht Leave-one-out 6/6.
Held-out 2/3; strenges Schwellengesetz refuted, kontinuierliche Monotonie
(KL_B/KL_C ↑ mit LIA, ρ_s ≈ 0.78) überlebt. Falsifiziert: Fisher-Krümmung,
Entropie-Taylor, Spektralkonzentration, Partnerkosinus, CNI, Lyapunov-Proxy.

## 4. Distanz zum Ziel — die ehrliche Landkarte

**Fertig und belegt:**
- Konflikt-Detektion auf gefrorenen Modellen (ρ, LIA; Kalibrierung, FPR).
- Kausaler Schadensnachweis (Injektion in Pythia/GPT-2, z bis +10.9).
- Reparatur-Werkzeug: Zwei-Modes-Politik (Rotieren β\*=1 im Iso-Regime,
  Dämpfen γ=0.25 im Kegel-Regime), Gate-Konstante σ_c ≈ 0.089
  skalenstabil in der Pythia-Familie.
- Negativabgrenzung: Trainings-Deployment refuted (dieser Bericht);
  post-sum Kapazität nutzlos; Multipath-Rotation rettet nur katastrophale
  Fälle; LIA-Universalthreshold refuted (GPT-2).

**Offen (nächste Meile):**
1. GPT-2-L11-Anomalie klären (Numerik vs. echtes Verhalten) — 1 Lauf.
2. 410m/1b-Replikation: Verschwindet die letzte-Layer-Kegel-Layer mit
   besseren Trainings, oder wird sie schärfer?
3. Inferenz-Deployment-Demo: Gate in einen echten serve-Pfad einbauen und
   Task-Metrik (nicht nur KL) zeigen — der Schritt Richtung "anwendbar".
4. Natürliche Kancellationen während *Inferenz* zählen (Census im
   Serve-Pfad, nicht Training) — die Basis für Fall 3.

**Unterm Strich:** Der Adapter umgeht die Neutralisierung — nachgewiesen am
gefrorenen Modell mit harten Zahlen (Dämpfung in der Kegel-Layer: KL ≈ 0 statt
12.3M; Rotation im Iso-Regime: +0.35 Accuracy-Refit). Sein Einsatzort ist die
Inferenz, nicht das Training; die Trainings-Hypothese ist mit sauberem Null-
und Zensus-Befund geschlossen.

## Reproduktion

- `run_lia_matrix.py [pythia-70m|gpt2|pythia-160m]` → `results/lia_matrix.csv`
- `run_training_ab.py --mode {baseline,attenuate,rotate} --seed {0,1}
  [--tau 0.4]` (+ env `BLOCK_LEVEL=1`) → `results/train_*.csv`
- `run_cycle_audit.py` → `results/cycle_audit_*.csv`
- ρ-Zensus: inline in Session-Log (Block-Hooks, 1000 Steps)
