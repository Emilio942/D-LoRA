# RESEARCH START (v2): Switching Variable of the Rotation-vs-Attenuation Repair in Transformer Residual Streams (D-LoRA-II, Single-Question Program — spectral family falsified, data complete)

## PROBLEM SETUP

A pretrained transformer residual stream $h \in \mathbb{R}^d$ receives an additive conflicting pathway vector $q$ (mined partner activation). The pinned repair operator is

$$
\mathcal{I}(h, q, \beta, \gamma) = h + \gamma\, R_\beta(q),
$$

with $R_\beta$ the planar Givens rotation by $\beta \cdot \arccos\langle \hat h, \hat q\rangle$ ($\beta \in [0,1]$, norm-preserving to $6\times10^{-7}$). Phase I measured that full rotation ($\beta = 1$) beats leaving the cancellation in place in *some* layers and loses catastrophically in *others* (table T below). The **single question** of this program: identify the switching variable $\sigma(l)$ that classifies the six measured layers, and derive the closed-form decision law $\beta^\ast(\sigma)$ with its critical constant.

The candidate family "global covariance concentration" ($d_{\text{eff}}/d$, stable rank, spectral entropy) was **pre-registered and falsified** by the complete data in table T — it is closed for this program and may only appear as a control.

## MODEL (PINNED, NOT NEGOTIABLE)

- Models: `EleutherAI/pythia-70m` ($d = 512$, layers $\{1,3,5\}$), `gpt2` ($d = 768$, layers $\{2,6,10\}$), frozen, exact forward hooks.
- Corpus: cached Gutenberg pair, fixed seed; contexts of length 128; intervention at the last position; readout = KL of next-token distribution vs. unperturbed baseline, computed in float64; $n = 149$–$150$ positions per cell; paired sign test resolution $\pm 0.16$ z.
- Conflict partner per position: argmin-cos if negative, else negated argmax-cos (neighborhood $\pm 130$ excluded); cancellation dose solved from $\|h + s q\| = \rho^\ast(\|h\| + s\|q\|)$, $\rho^\ast = 0.2$.
- All quantities below are measured with this exact protocol; the KL table is complete — no cell is missing.

## TABLE T (measured, complete, binding — the data this program must explain)

| Layer | $\kappa_{d_{\text{eff}}}$ | $\kappa_{\text{sr}}$ | $\kappa_{\text{ent}}$ | $KL(sq)$ (cancellation) | $KL(R_1 q)$ (rotation) | label |
|---|---|---|---|---|---|---|
| pythia L1 | 0.071 | 0.003 | 0.792 | 2.09e6 | 1.50e6 | **rotation** |
| pythia L3 | 0.003 | 0.002 | 0.282 | 2.28e6 | 1.63e6 | **rotation** |
| pythia L5 | 0.006 | 0.002 | 0.490 | 0.70e6 | 16.9e6 | **attenuation** (24x worse) |
| gpt2 L2 | 0.001 | 0.001 | 0.043 | 3.14e6 | 1.96e6 | **rotation** |
| gpt2 L6 | 0.001 | 0.001 | 0.078 | 1.86e6 | 2.13e6 | **attenuation** (mild, 1.15x) |
| gpt2 L10 | 0.002 | 0.001 | 0.229 | 0.94e6 | 2.22e6 | **attenuation** (2.4x) |

Sanity: $KL(R_0 q) = KL(sq)$ verified to $< 0.1\%$ in every cell. Note the spectral columns do **not** separate the labels (pythia L3 has lower $\kappa_{d_{\text{eff}}}$ than L5 yet label *rotation*; gpt2 L2 has the lowest $\kappa_{\text{ent}}$ of all yet label *rotation*) — this falsifies candidate family (i)/(ii) of version 1 and is binding.

Additional measured partner statistics (Phase I): mean mined alignment $\langle \hat h, \hat q\rangle$ = $-0.91$ (L1), $-0.87$ (L3), $-0.99$ (L5), $-0.96$ (gpt2 L2), $-0.90$ (L6), $-0.91$ (L10) — likewise non-separating.

## PAIN POINT

Six labeled layers, three falsified order parameters, and one surviving candidate class: the damage of rotating $q$ must be carried by the **readout geometry** — the unembedding $W_U$ — or by a partner-geometry statistic not yet written down. Until $\sigma(l)$ is derived, the D-LoRA gate has no formula.

## OPERATIVE TASK (single question)

**Derive the closed-form switching variable and law**

$$
\boxed{\;\text{label}(l) = \text{rotation} \iff \sigma(l) > \sigma_c, \qquad \sigma: \text{layer} \to \mathbb{R} \text{ closed-form from pinned quantities},\;}
$$

consistent with all six rows of table T, and give $\sigma_c$.

### Candidate mechanisms (pre-ranked; (i)/(ii) are falsified, kept as controls)

(i) *Control (falsified):* global covariance concentration ($d_{\text{eff}}$, stable rank, spectral entropy) — table T excludes it.
(ii) *Control (falsified):* mean partner cosine — non-separating (see above).
(iii) **Logit-readout law (leading candidate):** the damage of $R_1 q$ is proportional to the unembedding mass carried by the rotated-out component,
$$
\sigma(l) = \frac{\|P_{\hat h^{\perp}} q\; \text{routed through } W_U\|}{\|W_U q\|}
\quad\text{e.g.}\quad
\sigma(l) = 1 - \frac{\|W_U R_1 q - W_U q\|_2 \text{-weighted true-logit share}}{\dots}
$$
— to be made precise by the derivation; it predicts model-dependence beyond geometry and must reproduce the pythia-L5 catastrophe ($16.9\times$) and the gpt2-L2 benefit simultaneously.
(iv) *Partner-depth law:* $\sigma$ = function of the mined alignment distribution (e.g., $|\langle \hat h, \hat q\rangle|$ quantiles) — the derivation must beat (iii) to claim it.
(v) Refutations welcome: any alternative closed form consistent with all six rows and falsifiable under the pinned protocol.

## TECHNICAL LEVERAGE

(i) The full β-sweep pipeline is implemented and GPU-fast ($\approx 2$ min per (model, layer) cell); $W_U$-projected statistics are one matrix product per cell.
(ii) float64 log-softmax mandatory; $KL$ values in table T span $7\times10^5$–$1.7\times10^7$ — order-of-magnitude separation, robust to numerics.
(iii) $W_U$ is available for both pinned models; logit-lens projections per layer are one hook away.
(iv) Extension to pythia-160m/410m replicates the law at zero new code.

## VERDICT + FALSIFIABILITY

**Verdict slot:** `valid / invalid / unknown / untestable-at-current-oracle` + novelty score.

**Refuted if:** under the pinned protocol the derived $\sigma$ misclassifies any of the six rows of table T, or requires per-layer free parameters (a law with six fitted constants is invalid — at most two: $\sigma_c$ and one slope), or predicts a *new* measurement inconsistent with a fresh 150-position cell on pythia-160m.

**Formal constraint:** all constants depend only on (layer, $d$, $\rho^\ast$, dose, $n$, $W_U$). Restating table T is not novel evidence. Spectral-concentration variables are excluded by table T unless the derivation *explains the six-row pattern* through a second-order correction and predicts the pythia-160m replication quantitatively. If the derivation needs quantities outside the pinned protocol, mark `untestable-at-current-oracle`.

**Core sharpened question:** *why does the identical intervention ($\rho^\ast = 0.2$ cancellation vs. its full rotation) flip from beneficial to catastrophic between pythia L3 and L5 — and between gpt2 L2 and L6 — while every global spectrum statistic stays flat?* Derive $\sigma$ and $\sigma_c$.

---

*Messumgebung: D-LoRA-Repo (`/home/emilio/Documents/ai/D-LoRA`), Skripte `run_kausal.py`, `run_beta_sweep.py`, `dlora/verification.py` (`r_star_apply` mit `beta`-Parameter); Tabelle T vollständig aus `run_kausal`-Protokoll + Kovarianz-Spektren (2e4 Aktivierungen/Layer), Seed 77.*
