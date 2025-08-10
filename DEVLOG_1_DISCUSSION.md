# Development Log 1 — Discussion Format

Project: Stop Match (NB-based YouTube View Recovery Detection)
Period: Recent sprint summary

## Session 1. Problem Reframing
- PM: Hardware optimization is on hold due to budget and time constraints. Return to the origin and focus on YouTube view time series.
- Researcher: In YouTube data, NB_MIN consistently appears strong. With GPT‑5, we checked down to hardware signals and NB_MIN still dominates. What does this counter‑indicate?
- Algorithm: Two hypotheses. (1) Normalization/smoothing/clipping may press down the ceiling, causing a structural inversion. (2) The real‑world distribution may have a thick floor and short, clipped peaks. Either way, decision distortion occurs, so we need a recovery signal.
- Conclusion: Lock v1 goal to early detection of NB value recovery points.

## Session 2. Metrics and Signals
- Algorithm: Keep metrics simple. NB_MAX, NB_MIN, R = MAX/MIN, S = MAX − MIN. These four are sufficient.
- Data: Use `log1p` for long‑tail correction; robust per‑window scaling (Q05 ~ Q95 to [0, 1]) as defaults.
- Analyst: How do we define the recovery signal?
- Algorithm (minimal rules):
  - A. NB_MIN drops against its 3h rolling median (thinner floor)
  - B. R turns upward (increase over the last 1h average)
  - C. NB_MAX 20‑min moving average crosses above its 60‑min moving average
- PM: If 2 of 3 are satisfied → recovery candidate; if all 3 → recovery confirmed.
- Action: Implement metric computation and signal decision as separate modules.

## Session 3. Title Clustering and Ranking
- Data: View by title clusters, not single videos; variations on the same topic interact.
- Engineer: Text preprocessing (remove stopwords/emojis; keep numbers/season tokens) → TF‑IDF → cosine similarity ~0.4 for clustering; re‑merge daily.
- Researcher: Cluster cards include NB_MAX, NB_MIN, R, S, recovery score, state (cooling/observing/candidate/confirmed/overheated).
- PM: Dashboard top ranking is by recovery score; users look at the top cards.

## Session 4. Guardrails and Safe Mode
- Algorithm: Defense against balance collapse: if R < 1.2 or S < 10 → hold publishing; on violation, auto‑recalibrate (expand window, widen percentile band, toggle log1p).
- Engineer: If recalibration fails, switch to Safe Mode: conservative raw percentiles (1–99) and signals only.
- PM: Show mode badge (normal/recalibrating/safe) and failure counter for transparency.

## Session 5. Validation and Learning Loop
- Analyst: After recovery confirmation, evaluate over next 24h. KPIs: Precision@N, median growth rate, lead time (signal → peak).
- Data: Store publishing outcomes as labels (success/failure/void). Weekly auto‑retrain thresholds and window length. Add weekday/time‑of‑day dummies for baseline correction.
- PM: Separate Shorts and Longform. Exclude bot‑like signals (exposure spike + CTR drop) from recovery.

## Session 6. Observations and Interpretation
- Researcher: Repeatedly seeing cards where NB values recover gradually; clearer at cluster level.
- Algorithm: Empirical sequence observed: floor thins → ratio improves → top recovers. Regardless of hypothesis (1) or (2), this is a valid practical trigger.

## Session 7. Open Issues
- Data: Sparse clusters; few samples cause false positives.
- Algorithm: Introduce minimum samples and priors. If n < 30, relax thresholds and cap score.
- PM: Mixed‑language titles; branch pipeline via language detection.

## Session 8. Next Sprint Plan
1. Apply recovery score scaled 0–100 with thresholds at 60/75/90
2. Stabilize cluster re‑merge logic (auto‑tune similarity)
3. Auto‑collect failure logs and generate weekly reports
4. Define overheated policy (pace uploads, pivot to derivative content)
5. Public docs: state hypothesis and limits of why NB_MIN looks strong

## Appendix. Agreed Minimal Spec
- Preprocessing: `log1p` → per‑window Q05–Q95 normalization
- Metrics: `NB_MAX = pct95 × 100`, `NB_MIN = pct05 × 100`, `R = MAX/MIN`, `S = MAX − MIN`
- Recovery signal: 2 of A/B/C = candidate; all 3 = confirmed
- Guardrails: if `R < 1.2` or `S < 10` → recalibrate → on failure, Safe Mode
- Clustering: TF‑IDF cosine ~0.4, re‑merge daily
- Evaluation: Precision@N, 24h median improvement, lead time

## Notes
This project started from YouTube views—not hardware—and continues to learn from field data. We avoid premature certainty, capture recovery signals with consistent rules, and validate with real outcomes.


