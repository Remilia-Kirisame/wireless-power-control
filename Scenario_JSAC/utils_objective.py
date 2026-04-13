# =============================================================================
# Filename: utils_objective.py
# Description: Utility functions for objective calculations in wireless power control.
# Includes:
#   - PyTorch loss function for GNN training
#   - Numpy functions for evaluation metrics (Sum Rate, SINR)
# =============================================================================

import numpy as np

# =============================================================================
# Evaluation Metrics (Numpy / Standard Calculation)
# =============================================================================

def get_directLink_channel_losses(channel_losses):
    """
    Extracts the diagonal elements (Direct Links) from the channel matrix.
    Args:
        channel_losses: [Batch, N, N]
    Returns:
        [Batch, N]
    """
    return np.diagonal(channel_losses, axis1=1, axis2=2)


def get_crossLink_channel_losses(channel_losses):
    """
    Extracts the off-diagonal elements (Interference Links) by masking diagonals.
    Args:
        channel_losses: [Batch, N, N]
    Returns:
        [Batch, N, N] (Diagonals set to 0)
    """
    N = np.shape(channel_losses)[-1]
    return channel_losses * ((np.identity(N) < 1).astype(float))


# Calculates Signal-to-Interference-plus-Noise Ratio.
def compute_SINRs(general_para, allocs, directlink_channel_losses, crosslink_channel_losses):
    """
    Computes SINR for a batch of layouts.
    
    Args:
        general_para: Config object containing noise/power params.
        allocs: Power allocation matrix [Batch, N] (binary or continuous).
        directlink_channel_losses: Direct channel gains [Batch, N].
        crosslink_channel_losses: Interference channel gains [Batch, N, N].
        
    Returns:
        SINRs: [Batch, N]
    """
    assert np.shape(directlink_channel_losses) == np.shape(allocs), \
        "Mismatch shapes: {} VS {}".format(np.shape(directlink_channel_losses), np.shape(allocs))
        
    # Numerator: P_i * h_ii
    SINRs_numerators = allocs * directlink_channel_losses
    
    # Denominator: Sum(P_j * h_ij) + Noise/Tx_Power
    # Note: allocs is expanded to [Batch, N, 1] for broadcasting against [Batch, N, N]
    interference = np.matmul(crosslink_channel_losses, np.expand_dims(allocs, axis=-1))
    interference = np.squeeze(interference, axis=-1)
    
    noise_term = general_para.output_noise_power / general_para.tx_power
    SINRs_denominators = interference + noise_term
    
    # SINRs = SINRs_numerators / (SINRs_denominators)
    SINRs = SINRs_numerators / (SINRs_denominators + 1e-12)
    return SINRs


def compute_rates(general_para, allocs, directlink_channel_losses, crosslink_channel_losses):
    """
    Computes Sum Rate (Shannon Capacity) based on SINR.
    Returns:
        rates: [Batch, N] (bits/s/Hz)
    """
    SINRs = compute_SINRs(general_para, allocs, directlink_channel_losses, crosslink_channel_losses)
    # Using log2(1 + SINR)
    rates = np.log2(1 + SINRs)
    return rates


# =============================================================================
# JSAC Evaluation Helpers
# =============================================================================

def compute_green_sumrate(general_para, allocs, directlink_channel_losses, crosslink_channel_losses, green_mask):
    """
    Computes mean sum-rate over Green (communication) links only.

    Args:
        general_para: Config object.
        allocs: [Batch, K] power allocations.
        directlink_channel_losses: [Batch, K] direct channel gains.
        crosslink_channel_losses: [Batch, K, K] interference channel gains.
        green_mask: (K,) bool — True for Green links.

    Returns:
        mean_green_sumrate: scalar — average Green sum-rate across the batch.
    """
    rates = compute_rates(general_para, allocs, directlink_channel_losses, crosslink_channel_losses)
    green_rates = rates[:, green_mask]
    return np.mean(np.sum(green_rates, axis=1))


def compute_yellow_sinr_violation(general_para, allocs, directlink_channel_losses, crosslink_channel_losses, yellow_mask, sinr_min):
    """
    Computes Yellow (sensing) SINR violation rate.

    Args:
        general_para: Config object.
        allocs: [Batch, K] power allocations.
        directlink_channel_losses: [Batch, K] direct channel gains.
        crosslink_channel_losses: [Batch, K, K] interference channel gains.
        yellow_mask: (K,) bool — True for Yellow links.
        sinr_min: float — minimum SINR threshold (linear scale).

    Returns:
        violation_rate: float — percentage of Yellow links below sinr_min.
        yellow_sinrs: [Batch, n_yellow] — SINR values for Yellow links.
    """
    sinrs = compute_SINRs(general_para, allocs, directlink_channel_losses, crosslink_channel_losses)
    yellow_sinrs = sinrs[:, yellow_mask]
    violation_rate = np.mean(yellow_sinrs < sinr_min) * 100
    return violation_rate, yellow_sinrs