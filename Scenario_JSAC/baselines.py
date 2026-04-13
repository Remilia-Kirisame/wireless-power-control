# ==============================================================================
# Filename: baselines.py
# Description: Baseline algorithms for JSAC wireless power control.
# Algorithms included:
#   - WMMSE_JSAC (per-group power constraint + Yellow SINR constraint)
#   - Naive equal-power (lower bound)
# ==============================================================================

import numpy as np


# =============================================================================
# WMMSE for JSAC
# =============================================================================
# Adapted from D2D batch_WMMSE2_QoS. Changes (see "WMMSE - JSAC vs D2D.md"):
#   1. Per-group (Blue car) power projection replaces per-link clip
#   2. Yellow-only SINR constraint replaces all-link rate constraint
#   3. alpha: 1 for Green, small positive for Yellow
def batch_WMMSE2_JSAC(N_batch, alpha, H, Pmax, var_noise, group_ids, sinr_min, yellow_mask):
    """
    WMMSE with per-group (Blue car) power constraint + Yellow SINR constraint.

    Initializes at equal power within each group (Pmax/M per link). This is
    critical for convergence: the per-group projection (unlike the D2D per-link
    clip) has no mechanism to push power upward, so starting from low power
    leads to the algorithm getting stuck at a suboptimal low-power solution.

    Args:
        N_batch:     Number of layouts in the batch
        alpha:       Weights [Batch, K] — 1 for Green, small positive for Yellow
        H:           Channel magnitude [Batch, K, K] (Rx, Tx) — already masked (sparse)
        Pmax:        Max power per Blue car (scalar, in normalized units)
        var_noise:   Noise variance (scalar)
        group_ids:   [K] int array — which Blue car each link belongs to (0..B-1)
        sinr_min:    Minimum SINR for Yellow links (scalar, linear scale)
        yellow_mask: [K] boolean — True for Yellow links

    Returns:
        p_opt: Optimized power [Batch, K]
    """
    K = len(group_ids)
    n_blue = int(np.max(group_ids)) + 1

    # --- Initialization: equal power within each group ---
    # Each link gets Pmax / (links_per_group) so each group starts at full budget.
    # D2D WMMSE relied on per-link clip(v, 0, sqrt(Pmax)) to push power up to Pmax,
    # but per-group projection only scales DOWN — so we must start near full power.
    p = np.zeros((N_batch, K, 1))
    group_indices = [np.where(group_ids == b)[0] for b in range(n_blue)]
    for links in group_indices:
        p[:, links, :] = Pmax / len(links)
    v = np.sqrt(p)
    w_alpha = alpha.reshape(N_batch, K, 1).astype(float)

    # Lagrange multipliers for Yellow SINR constraint (only Yellow will be nonzero)
    mu = np.zeros((N_batch, K, 1))
    lr_mu = 0.5

    # Reshape yellow_mask for broadcasting: [1, K, 1]
    ymask = yellow_mask.reshape(1, K, 1).astype(float)

    EPS = 1e-12

    for iteration in range(100):
        # --- Extract diagonal (direct-link) channel ---
        h_diag = np.diagonal(H, axis1=1, axis2=2).reshape(N_batch, K, 1)

        # --- Total received power and SINR components ---
        total_rx_p = np.matmul(np.square(H), p) + var_noise
        signal = p * np.square(h_diag)
        interference_plus_noise = np.maximum(total_rx_p - signal, var_noise)

        # --- MMSE receiver (u) and weight (w) ---
        u = (h_diag * v) / (total_rx_p + EPS)
        w = total_rx_p / (interference_plus_noise + EPS)

        # --- Effective weight: alpha + mu (QoS boost for Yellow) ---
        effective_weight = w_alpha + mu
        H_T_sq = np.transpose(np.square(H), (0, 2, 1))
        u_sq_w_eff = np.square(u) * w * effective_weight

        # --- Update v (precoder) ---
        A = np.matmul(H_T_sq, u_sq_w_eff)
        B = effective_weight * w * u * h_diag
        v_new = B / (A + EPS)

        # Non-negativity
        v = np.maximum(v_new, 0.0)
        p = np.square(v)

        # --- Per-group power projection (replaces per-link clip) ---
        for links in group_indices:
            group_power = np.sum(p[:, links, :], axis=1, keepdims=True)  # [Batch, 1, 1]
            excess = group_power > Pmax
            scale = np.where(excess, Pmax / group_power, 1.0)
            p[:, links, :] *= scale
        v = np.sqrt(p)

        # --- Dual update: Yellow SINR constraint ---
        signal_new = p * np.square(h_diag)
        total_rx_p_new = np.matmul(np.square(H), p) + var_noise
        interference_plus_noise_new = np.maximum(total_rx_p_new - signal_new, var_noise)

        sinr = signal_new / (interference_plus_noise_new + EPS)
        mu = np.maximum(0.0, mu + lr_mu * (sinr_min - sinr) * ymask)

    return np.squeeze(p, axis=-1)


# =============================================================================
# Naive Equal-Power Baseline
# =============================================================================
def naive_equal_power(general_para, group_ids):
    """
    Equal power split within each Blue car. Returns [K] normalized allocation
    (sums to 1.0 per group, so each link gets 1/M of the budget).

    Args:
        general_para: Config object (uses n_blue)
        group_ids:    [K] int array — which Blue car each link belongs to

    Returns:
        allocs: [K] power allocation (normalized: multiply by Pmax for absolute power)
    """
    K = len(group_ids)
    allocs = np.zeros(K)
    for b in range(general_para.n_blue):
        links = np.where(group_ids == b)[0]
        allocs[links] = 1.0 / len(links)
    return allocs
