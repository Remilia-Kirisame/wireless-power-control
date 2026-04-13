# WMMSE — JSAC vs D2D: A Detailed Comparison

This note compares the WMMSE algorithm as used in D2D (Framework_ver_2.3) vs the JSAC scenario (Framework_ver_4.0). The goal is to have a clear picture of what changes and what stays the same before coding.

---

## 1. Problem Formulation Side-by-Side

### D2D (Framework_ver_2.3)

$$
\max_{\mathbf{p}} \sum_{k=1}^{K} \alpha_k \log_2 \left(1 + \frac{|h_{kk}|^2 p_k}{\sum_{j \neq k} |h_{kj}|^2 p_j + \sigma^2}\right)
$$

Subject to:
- Per-link power constraint: $0 \leq p_k \leq P_{\max}$, for all $k$
- (QoS variant) Minimum rate: $r_k \geq r_{\min}$, for all $k$

**Key properties:**
- All K links are in the objective (alpha_k = 1 for all k)
- All K links mutually interfere (H is dense)
- Each link has its own independent power budget

### JSAC (Framework_ver_4.0)

$$
\max_{\mathbf{p}} \sum_{k \in \mathcal{G}} \alpha_k \log_2 \left(1 + \frac{|h_{kk}|^2 p_k}{\sum_{j \neq k} |h_{kj}|^2 p_j + \sigma^2}\right)
$$

Subject to:
- Per-group (Blue car) power constraint: $\sum_{k \in \mathcal{B}_b} p_k \leq P_{\max}$, for each Blue car $b$
- Sensing SINR constraint: $\text{SINR}_k \geq \gamma_{\min}$, for all $k \in \mathcal{Y}$

Where:
- $\mathcal{G}$ = set of Green (communication) links
- $\mathcal{Y}$ = set of Yellow (sensing) links
- $\mathcal{B}_b$ = set of links belonging to Blue car $b$

**Key properties:**
- Only Green links appear in the objective
- Constraint is on Yellow links' SINR (not rate)
- H is sparse: only links on the same orthogonal channel AND in different Blue cars interfere
- Power budget is shared within each Blue car cluster

---

## 2. What Changes in WMMSE — A Dimension-by-Dimension Analysis

### 2.1 Channel Matrix H — No WMMSE code change needed

In D2D, $H$ is a dense $K \times K$ matrix (all links interfere). In JSAC, after applying the orthogonal channel mask (Step 1C), $H$ is sparse:

```
H[i,j] > 0  only if:
  - i == j  (direct link), OR
  - group_ids[i] != group_ids[j]  AND  channel_ids[i] == channel_ids[j]  (cross-Blue, same channel)
```

**Impact on WMMSE:** None on the internal math. The WMMSE update equations (u, w, f, b) all operate on H via matrix multiplications. Zero entries in H simply contribute zero to the interference sums. The algorithm doesn't "know" which entries are zero or why — it just computes with whatever H it receives.

This is the key insight: **the channel masking in Step 1C does ALL the work of encoding the orthogonal channel structure. WMMSE sees a regular K x K matrix and operates as usual.**

### 2.2 Objective Weights (alpha) — Small change

In D2D: `alpha = ones([Batch, K])` — all links contribute equally to the objective.

In JSAC: `alpha[k] = 1 if k is Green, 0 if k is Yellow` — only Green links matter for the sum-rate objective.

**Impact on WMMSE:** The `alpha` vector is already a parameter of `batch_WMMSE2()`. Setting `alpha[yellow_links] = 0` makes WMMSE ignore Yellow links in the objective. Yellow links still receive power (they affect interference), but their individual rates don't contribute to the weighted sum being maximized.

This is already supported by the existing interface — we just pass a different alpha vector. However, see the nuance in Section 3 about whether alpha=0 for Yellow is the right choice.

### 2.3 Power Constraint — The main structural change

**D2D (per-link):**
```python
b = np.clip(btmp, 0, np.sqrt(Pmax))
```
Each link is independently constrained to [0, Pmax]. This is a simple box projection.

**JSAC (per-group):**
```
For each Blue car b:
    total_power = sum(p[k] for k in group_b)
    if total_power > Pmax:
        p[group_b] *= Pmax / total_power   # proportional scaling
```

This is a projection onto a simplex-like constraint (actually onto an L1-ball). The links within a Blue car compete for a shared power budget.

**Impact on WMMSE:** This is the single biggest code change. The projection step in the inner loop of WMMSE must be replaced. See Section 4 for the detailed implementation.

### 2.4 QoS / Sensing Constraint — Structural change (for QoS variant)

**D2D QoS:** Minimum rate $r_k \geq r_{\min}$ for ALL links, enforced via Lagrange multiplier $\mu_k$:

```python
mu = max(0, mu + lr * (r_min - rate_k))
effective_weight = alpha + mu
```

**JSAC QoS:** Minimum SINR $\gamma_k \geq \gamma_{\min}$ for YELLOW links only:

```python
mu[k] = max(0, mu[k] + lr * (sinr_min - sinr_k))   # only for k in Yellow
mu[k] = 0   # for k in Green (no constraint)
effective_weight = alpha + mu
```

**Impact:** Same Lagrangian dual mechanism, but:
1. The constraint is on SINR (not rate). Since $r = \log_2(1 + \text{SINR})$, constraining SINR is equivalent to constraining rate — just with a different threshold value: $\gamma_{\min}$ maps to $r_{\min} = \log_2(1 + \gamma_{\min})$.
2. Only Yellow links have the constraint, so $\mu_k = 0$ for Green links (they are never penalized).

---

## 3. Theoretical Nuance: What Should alpha Be for Yellow Links?

Setting `alpha[yellow] = 0` tells WMMSE: "I don't care about Yellow rates at all in the objective." This means WMMSE will allocate power primarily to help Green links, potentially starving Yellow links of power entirely (since they don't contribute to the thing being maximized).

Since we always want the sensing QoS constraint in JSAC, the Lagrange multiplier $\mu_k$ will push power towards Yellow links that fall below $\gamma_{\min}$. However, with `alpha[yellow] = 0`, the only reason WMMSE allocates power to Yellow links is via $\mu$. If $\mu$ starts at 0, initial iterations completely ignore Yellow, which may slow convergence.

- Alternative: set `alpha[yellow]` to a small positive value (e.g., 0.1) to give WMMSE a "hint" that Yellow links matter, while still prioritizing Green. The Lagrange multiplier then fine-tunes the balance.
- In practice, both approaches converge (WMMSE is robust), but the latter may converge faster.

**Recommendation:** Start with `alpha[yellow] = small_positive` (e.g., 0.1). Test `alpha[yellow] = 0` later to compare convergence speed.

---

## 4. Implementation Details

### 4.1 Pseudocode: `batch_WMMSE2_JSAC`

Since JSAC always requires the sensing QoS constraint, we implement a single function that includes the Lagrange multiplier mechanism. There is no separate unconstrained variant.

```
Input:
    p_int:       Initial power [Batch, K, 1]
    alpha:       Weights [Batch, K] — 1 for Green, small positive for Yellow
    H:           Channel magnitude [Batch, K, K] — already masked (sparse)
    Pmax:        Max power per Blue car (scalar)
    var_noise:   Noise variance (scalar)
    group_ids:   [K] — which Blue car each link belongs to
    sinr_min:    Minimum SINR for Yellow links (scalar)
    yellow_mask: [K] boolean — True for Yellow links

Output:
    p_opt: Optimized power [Batch, K]

Algorithm:
    v = sqrt(clip(p_int, 0, Pmax))
    mu = zeros([Batch, K, 1])               # Lagrange multipliers (only Yellow will be nonzero)
    lr_mu = 0.5

    FOR iter = 1 to 50:
        # === Compute SINR for all links ===
        h_diag = diagonal(H)
        total_rx_p = H^2 @ p + var_noise
        signal = p * h_diag^2
        interf_plus_noise = max(total_rx_p - signal, var_noise)

        # === MMSE receiver and weight ===
        u = (h_diag * v) / total_rx_p
        w = total_rx_p / interf_plus_noise

        # === Effective weight: alpha + mu (QoS boost for Yellow) ===
        effective_weight = alpha + mu

        # === Update v (precoder) ===
        A = H_T^2 @ (u^2 * w * effective_weight)
        B = effective_weight * w * u * h_diag
        v_new = B / A

        # === KEY DIFFERENCE: Per-group power projection ===
        p = v_new^2
        FOR each Blue car group b:
            links = where group_ids == b
            group_power = sum(p[:, links])
            IF group_power > Pmax:
                p[:, links] *= Pmax / group_power
        v = sqrt(p)

        # === Dual update: only for Yellow links ===
        sinr = signal / interf_plus_noise
        mu_update = lr_mu * (sinr_min - sinr)
        mu = max(0, mu + mu_update * yellow_mask)   # only Yellow links get mu > 0

    RETURN squeeze(p)
```

**Key differences from D2D marked:** Per-group power projection (replacing per-link clip) and Yellow-only dual variable update.

### 4.2 Pseudocode: `naive_equal_power`

```
Input:
    general_para: config (for n_blue)
    group_ids: [K]

Output:
    allocs: [K] or [Batch, K]

Algorithm:
    FOR each Blue car b:
        links = where group_ids == b
        allocs[links] = 1.0 / len(links)   # normalized: sums to 1 per group
```

This is the simplest possible allocation: divide the power budget equally among all links in each Blue car. No channel awareness, no optimization. Pure lower bound.

---

## 5. Coding Implementation Map

### 5.1 What to reuse from D2D (no changes needed in the equations)

| Component | D2D code | JSAC status |
|---|---|---|
| MMSE receiver `f = valid_rx / interference` | `batch_WMMSE2` line ~137 | Same — H sparsity handles interference automatically |
| MMSE weight `w = 1 / (1 - f*valid_rx)` | `batch_WMMSE2` line ~138 | Same |
| Precoder update numerator/denominator | `batch_WMMSE2` lines ~156-170 | Same |
| Matrix multiplications against H | Throughout | Same — zero H entries contribute zero |

### 5.2 What changes

| Change | D2D code | JSAC replacement |
|---|---|---|
| Power projection | `b = np.clip(btmp, 0, np.sqrt(Pmax))` | Per-group proportional scaling (see 4.1) |
| alpha vector | `np.ones([Batch, K])` | 1 for Green, small positive for Yellow |
| QoS constraint target | Rate: `r_min` | SINR: `sinr_min` (equivalent via log transform) |
| QoS dual variable scope | All K links | Yellow links only: `mu *= yellow_mask` |

### 5.3 New function signatures

Since JSAC always requires the sensing QoS constraint, we only need one WMMSE function (not two):

```python
def batch_WMMSE2_JSAC(p_int, alpha, H, Pmax, var_noise, group_ids, sinr_min, yellow_mask):
    """
    WMMSE with per-group (Blue car) power constraint + Yellow SINR constraint.
    alpha should be 1 for Green, small positive for Yellow.
    H should already have non-interfering entries zeroed (from Step 1C).
    Uses Lagrange multipliers (mu) for Yellow links below sinr_min.
    """

def naive_equal_power(general_para, group_ids):
    """
    Equal power split within each Blue car. Returns [K] allocation.
    """
```

---

## 6. Summary of Changes

```
D2D WMMSE                              JSAC WMMSE (single function)
---------                              ----------
H: dense K x K                    →    H: sparse (masked in Step 1C)
alpha: all ones                   →    alpha: 1 for Green, small positive for Yellow
clip(b, 0, sqrt(Pmax)) per link  →    proportional scaling per Blue car group
QoS on all links (rate)          →    QoS on Yellow only (SINR)
Two variants (with/without QoS)  →    One function (QoS always included)

Internal u/w/f/b update equations: IDENTICAL
Matrix multiplication logic:       IDENTICAL
Convergence properties:            SAME (WMMSE guarantees still hold)
```

The sparse H structure means fewer non-zero terms in each sum, which actually makes each iteration slightly cheaper. The per-group projection is a marginally more complex operation than per-link clipping, but still O(K) per iteration.
