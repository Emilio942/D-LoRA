# D-LoRA-II Cycle-Audit Report (autonomous session)

Eingang: 40+ Hypothesen-Zyklen der Pipeline (Verdacht: σ(l) bestimmt Rotation vs. Attenuation).
Methode: Jede testbare σ-Kandidat-Größe wurde für alle 6 gelabelten Layer aus demselben
Mining-Protokoll berechnet (n=149/150, Dosis ρ\*=0.2, GPU) und gegen die Labels getestet
(0 freie Parameter, perfekte Trennung gefordert), plus Leave-one-out und Held-out-Replikation.

## Mining-Korrektur (wichtig)

Erste Version degenerierte in 5/6 Zellen: der argmax-cos-Partner traf Positions-Dubletten
(identische Kontexte) → q ≈ −h → die "Dosis" war nur eine Norm-Skalierung von h. Fix:
Dubletten-Maske (|cos|>0.995) + Near-Parallel-Verwerfung (‖q⊥‖/‖q‖ < 0.05). Alle Zahlen
dieses Reports aus der korrigierten Pipeline.

## Audit-Ergebnis (13 Kandidaten, Dosis 0.2, 6 Layer)

| Kandidat | Quelle (Cycle) | Trennung 6/6 | LOO | Bemerkung |
|---|---|---|---|---|
| readout_norm = ‖W_U q‖/‖q‖ | 28/72-Familie | JA (eng: 114.9 vs 123.4) | **FAIL** (gL2) | Pythia L5: 545× Logit-Gain |
| **LIA = \|⟨R₁q̂, v_max(W_UᵀW_U)⟩\|** | 850 (repariert) | **JA** (0.068 vs 0.102) | **6/6** | **einziger LOO-stabiler Kandidat** |
| topspan | 101/33-Familie | JA (haarengenau: 0.9651 vs 0.9691) | FAIL (gL2) | |
| lcm_quad (Fisher-Krümmung) | 2385/1 | NEIN | – | |
| cerlc (lineare Entropie) | 5198 | NEIN | – | |
| ent_ratio / entdiff_rot | 4861/12/26 | NEIN | – | |
| eig_align / eig_mass | 1 | NEIN | – | |
| umci / cni / pencil_gap | 307/2631/13 | NEIN | – | |
| downstream_sens (Lyapunov-Proxy) | 30/425/2 | NEIN | – | |

## LIA: Validierung

σ(l) = |⟨R₁q̂, v_max(W_UᵀW_U)⟩| — Alignment des voll rotierten Partners mit der
dominierenden Logit-Richtung des Unembeddings.

Training (6 Layer): Rotation {pL1 0.038, pL3 0.058, gL2 0.068} < Attenuation
{pL5 0.102, gL6 0.105, gL10 0.112}. LOO 6/6, σ_c ≈ 0.085 (Mittelpunkt, 1 Parameter).

Held-out (3 neue Layer, Dosis 0.2):
- pythia L2: σ=0.036 → rotation, Wahrheit rotation (KL_B/KL_C = 0.56) — TREFFER
- pythia L4: σ=0.077 → rotation, Wahrheit rotation (0.59) — TREFFER
- gpt2 L8: σ=0.117 → attenuation, Wahrheit rotation (0.86, nur 1.17× Abstand) — **FEHLSCHLAG**

**Strenges Schwellen-Gesetz: refuted** (eigene Klausel: "misclassifies any").

**Überlebende kontinuierliche Form:** KL(R₁q)/KL(sq) steigt monoton in σ(LIA) über alle
9 Layer × 2 Modelle (Spearman ρ_s ≈ 0.78, p ≈ 0.02). Interpretation: Rotation ist genau
dann relativ vorteilhaft, wenn sie den Partner *aus* der dominierenden Logit-Richtung
hinausdreht; σ(LIA) misst, wie weit der rotierte Partner in dieser Richtung landet.

## Mechanismus (konsistent mit Phase I)

Pythia L5: Partner-Logit-Gain ‖W_U q‖/‖q‖ = 545 (vs. 23–158 sonst) — der Partner lebt in
der dominierenden Richtung; jede Rotation bleibt im Kegel dieser Richtung und trifft die
Logits mit massivem Gain. Dämpfung (Att25, KL≈0) bleibt daher die korrekte Reparatur dort;
Rotation ist die korrekte Reparatur in Layern mit niedrigem σ(LIA).

## Falsifizierte Klassen (dieses Audit)

- Fisher-Krümmungs-Quadratform (Cycle 2385): trennt nicht.
- Linear-Entropie-Taylor (Cycle 5198): trennt nicht.
- Globale Spektralkonzentration (bereits v2): erneut bestätigt als nicht-trennend.
- Partner-Mittelkosinus, CNI, Bleistift-Gap, Downstream-Empfindlichkeit: trennen nicht.
- LIA als universelles Schwellengesetz: refuted auf Held-out (gL8); kontinuierliche
  Monotonieform überlebt.

## Reproduktion

`run_cycle_audit.py` (Audit + Batterie), Held-out-Skript inline im Session-Log;
CSVs: `results/cycle_audit_sigma.csv`, `results/cycle_audit_verdict.csv`.
Protokoll: Seed 77, Gutenberg-Korpus, n=149–150, Dosis 0.2, float64-KL.
