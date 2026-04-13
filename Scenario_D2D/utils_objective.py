# =============================================================================
# Filename: utils_objective.py
# Description: Utility functions for objective calculations in wireless power control.
# Includes:
#   - PyTorch loss function for GNN training
#   - Numpy functions for evaluation metrics (Sum Rate, SINR)
# =============================================================================

import torch
import numpy as np

# =============================================================================
# PyTorch Loss Function (Differentiable)
# =============================================================================

# Custom loss function (negative sum rate) used for backpropagation.
def sumrate_loss(data, out, K, var_noise):
    """
    Calculates the negative Sum Rate loss for training the GNN.
    
    Args:
        data: PyG Data batch. data.y contains channel magnitudes [Batch, K, K].
        out: Model output [Batch*K, 3]. out[:, 2] is the power/probability p.
        K: Number of links (users).
        var_noise: Noise variance (normalized).
        
    Returns:
        loss: Negative mean sum rate (scalar tensor).
    """
    # 1. Extract power coefficients p from the 3rd output dimension
    # out[:, 2] is expected to be in (0,1) via Sigmoid
    p = out[:, 2]
    
    # 2. Reshape to [Batch, K, 1] to match channel dimensions
    # Note: data.num_graphs is the batch size
    # batch_size = data.num_graphs
    # p = torch.reshape(p, (batch_size, K, 1))
    p = torch.reshape(p, (-1, K, 1))  # (1,K,1) 在 DataLoader batch=1 的情形
    
    # 3. Prepare Channel Matrix H
    # data.y was stored as [1, K, K] or [Batch, K, K] in build_graph
    absH2 = data.y 
    # Ensure dimension alignment if necessary (PyG batching stacks graphs)
    # if absH2.dim() == 2: # Handling potential flattening by PyG
    #     absH2 = absH2.view(batch_size, K, K)
        
    # Transpose to align with [Receiver, Transmitter] convention if needed
    # (Assuming data.y is [Batch, Tx, Rx], we align to [Batch, Rx, Tx] for matmul)
    absH2 = absH2.permute(0, 2, 1)

    # 4. Calculate Received Power
    # rx_power[b, i, j] = Power at Receiver i from Transmitter j
    rx_power = torch.mul(absH2, p) # (1,K,K)
    
    # 5. Separate Signal (Diagonal) and Interference (Off-Diagonal)
    mask = torch.eye(K, device=out.device)
    valid_rx_power = torch.sum(rx_power * mask, dim=1)  # Signal
    
    # Interference = Sum of all powers - Signal + Noise
    # (1 - mask) selects off-diagonals
    interference = torch.sum(rx_power * (1 - mask), dim=1) + var_noise 
    
    # 6. Calculate Rate
    # Rate = log2(1 + S / (I + N))
    rate = torch.log2(1 + valid_rx_power / (interference + 1e-12))  #标准SINR
    
    # 7. Compute Mean Sum Rate across the batch
    sum_rate = torch.mean(torch.sum(rate, dim=1))   #均质，得到lossfunction
    
    # Return negative sum rate for minimization
    return -sum_rate


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