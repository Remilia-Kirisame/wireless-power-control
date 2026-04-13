# WirelessScaler and Two-Stage Data Generation

## Why Two Separate Dataset Generation Processes?

The training set and evaluation set are generated independently — the raw physics (layouts, path loss, fading) does not depend on the other set. The reason for the two-stage process is **normalization consistency**: the neural network learns to interpret inputs normalized by training-set statistics, so evaluation inputs must be normalized by the exact same statistics.

## What WirelessScaler Does

`WirelessScaler` performs **z-score normalization** (subtract mean, divide by std), with one critical design choice: **separate statistics for diagonal vs off-diagonal entries**.

- **Diagonal** (direct links): strong signals, short distance, high channel gain.
- **Off-diagonal** (interference links): weaker signals, longer distance, lower channel gain.

These two populations have very different distributions. Mixing them into a single mean/variance would produce meaningless z-scores that help neither the GNN nor the DNN.

### API

| Method | What it does |
|---|---|
| `fit(X)` | Compute `diag_mean`, `diag_var`, `off_mean`, `off_var` from `X` [B, K, K] |
| `transform(X)` | Apply stored statistics to normalize any `X` of shape [B, K, K] |

### Two Scalers in Practice

In `main.py`, two independent scalers are fitted:

| Scaler | Fitted on | Input transformation |
|---|---|---|
| `dist_scaler` | `1.0 / train_dists` (inverse distance) | Reciprocal distance matrix |
| `loss_scaler` | `np.sqrt(train_ch_final)` (sqrt channel loss) | Square-root of channel gain matrix |

## The Flow

```
Phase 1 (Training Data):
  generate_layouts() → path_loss → shadowing → fading → train_ch_final
  dist_scaler.fit(1/train_dists)        ← learn distribution statistics
  loss_scaler.fit(sqrt(train_ch_final)) ← learn distribution statistics
  dist_scaler.transform(1/train_dists)  → normalized training input (distances)
  loss_scaler.transform(sqrt(train_ch))  → normalized training input (losses)
  → Build graph → Train GNN on these normalized inputs

Phase 3 (Evaluation Data, per K):
  generate_evaluation_dataset(K, ..., dist_scaler, loss_scaler)
    generate_layouts() → path_loss → shadowing → fading → eval_ch
    dist_scaler.transform(1/eval_dists)  ← same scaler, training statistics
    loss_scaler.transform(sqrt(eval_ch))  ← same scaler, training statistics
    → Build graph → Feed into trained GNN
```

## Why Never Fit on Eval Data

If you re-fit the scaler on the evaluation set:
- The normalized values would center around the *eval set's own* distribution.
- The NN was trained to map *training-distribution-normalized* inputs to power allocations.
- Distribution mismatch → garbage predictions.

This is standard ML practice: **fit on train, transform both**.

## Additional Nuance: Different Fading Treatment

The training and evaluation pipelines also differ in what channel data they use:

| | Training (Phase 1) | Evaluation (Phase 3) |
|---|---|---|
| Channel matrix | `proc_train_losses()`: diagonal = full fading, off-diagonal = path-loss only | Raw `channel_losses`: full fading everywhere |
| Scaler fitted on | `sqrt(train_ch_final)` — the full-fading version | N/A (uses training scaler) |
| Scaler applied to | Same full-fading version | `sqrt(eval_channel_losses)` — also full fading |

The `proc_train_losses()` step is used only to build the GNN graph (which needs path-loss-only interference for stable training), not for the scaler fitting.

## Q&A: Smoke Test and the Unmodified Scaler

### Did using the unmodified WirelessScaler in the Step 4 smoke test cause problems?

No runtime error — but the normalization was **technically wrong** for JSAC.

**Why it still ran fine:** The diagonal mask `np.eye(K)` still correctly identifies direct links, because `channel_losses[b, i, i]` is always the direct link for link `i`, in both D2D and JSAC. So the diagonal statistics are correct.

**Where it goes wrong:** The off-diagonal statistics. In D2D, every off-diagonal entry is a real interference path. In JSAC, most off-diagonal entries are **structurally zero** (links on different channels don't interfere). The current `fit()` counts all K(K-1) off-diagonal entries, so:
- `off_mean` gets diluted towards zero by the many zero entries.
- `off_var` is distorted — it reflects the zero/nonzero mix, not the distribution of actual interference values.

And in `transform()`, zero entries become `(0 - off_mean) / off_var` instead of staying zero, injecting false signal into non-interfering link pairs.

**Bottom line:** The smoke test passed mechanically, but normalized values fed to the GNN were of degraded quality. This is what Step 5 fixes.

### What is `fit()` — a Python built-in?

No. `fit()` is a **custom method** defined in our `WirelessScaler` class (`main_supporter.py:31`). The naming convention follows scikit-learn's `StandardScaler.fit()` / `.transform()` API pattern, but there is no inheritance — it's entirely our own implementation.

### Current `fit()` vs JSAC-adapted `fit()` — what changes?

| Aspect | Current `fit()` (D2D) | Needed `fit()` (JSAC) |
|---|---|---|
| Diagonal mask | `np.eye(K)` | Same — still correct for direct links |
| Off-diag mask | Implicit: everything not on diagonal | Explicit: `interf_mask` — only same-channel, different-group pairs |
| Off-diag count | `L * K * (K-1)` — all off-diag entries | `L * nnz` where `nnz = np.count_nonzero(interf_mask)` |
| Off-diag mean | `sum(off_diag) / (L * K * (K-1))` | `sum(X * interf_mask) / (L * nnz)` |
| Off-diag var | Same denominator issue | Same fix — count only real interference entries |

The `transform()` similarly needs the mask: zero entries (non-interfering pairs) must stay zero after normalization, not become `(0 - off_mean) / off_var`.

**Signature change:** `fit(X)` → `fit(X, interf_mask, yellow_mask, green_mask)` (and similarly for `transform`). The masks are generated by `CS.jsac_metadata()` and are the same for all layouts with the same topology.

## JSAC Adaptation: 3-Category Normalization

### Why upgrade from 2 to 3 categories?

The D2D scaler uses 2 categories (diagonal / off-diagonal) because the only meaningful statistical boundary is direct vs interference links. In JSAC, we adopt **3 categories**:

| Category | What entries | Mask logic | Physical regime |
|---|---|---|---|
| **Direct-sensing** | Diagonal entries where link is Yellow | `np.diag(yellow_mask)` | Blue→Yellow, 2-20m |
| **Direct-communication** | Diagonal entries where link is Green | `np.diag(green_mask)` | Blue→Green, 2-20m |
| **Interference** | Off-diagonal where `interf_mask == 1` | `interf_mask` (already excludes diagonal) | Cross-Blue same-channel, ~50-250m |

All other off-diagonal entries (intra-Blue or different-channel) are structural zeros — excluded from statistics and left as zero after normalization.

### Why do this now when all links use the same channel model?

Right now, direct-sensing and direct-communication use identical path-loss, shadowing, and fading models with the same 2-20m distance range. Their statistics will be nearly identical, so the 3-way split doesn't improve normalization quality today.

But FUTURE_WORKPLAN Section 1 introduces a **triple channel model**:
- **Sensing**: d⁴ radar-equation path loss + compound fading + RCS parameter
- **Communication**: d² one-way path loss + Rician/Rayleigh fading
- **Interference**: d² NLOS path loss + Rayleigh fading

At that point, sensing and communication distributions diverge massively, and a 3-way split becomes essential. Implementing it now costs almost nothing (one extra mask in `fit()`/`transform()`) and avoids a refactor later.

### Sample count check

With B=10, M_y=2, M_g=3 and 1000 training layouts:
- Direct-sensing: 20 links × 1000 layouts = 20,000 samples
- Direct-communication: 30 links × 1000 layouts = 30,000 samples
- Interference: M × B × (B-1) = 5 × 10 × 9 = 450 entries × 1000 = 450,000 samples

All categories have plenty of samples for stable mean/variance estimation.
