# ==============================================================================
# Filename: baselines.py
# Description: Implementation of baseline algorithms for wireless power control.
# Algorithms included:
#   - FPLinQ (Fractional Programming)
#   - WMMSE (Weighted Minimum Mean Square Error)
#   - Simple Greedy Heuristic
# ==============================================================================

import numpy as np
import utils_objective as UO

# =============================================================================
# FPLinQ (Fractional Programming)
# =============================================================================
# From FPLinQ.py:
#   FP: A variant of FP_optimize, but not called.
def FP_optimize(general_para, g, weights):
    """
    Runs the FPLinQ algorithm for link scheduling/power control.
    
    Args:
        general_para: Config object (for power/noise params).
        g: Channel gains (magnitudes or losses) [Batch, N, N].
        weights: Weights for weighted sum rate [Batch, N].
        
    Returns:
        x_final: Optimized power allocation [Batch, N] in range [0, 1].
    """
    number_of_samples, N, _ = np.shape(g)
    assert np.shape(g)==(number_of_samples, N, N)
    assert np.shape(weights)==(number_of_samples, N)
    
    # Extract Direct and Cross links using utility functions
    g_diag = UO.get_directLink_channel_losses(g)      # [Batch, N]
    g_nondiag = UO.get_crossLink_channel_losses(g)    # [Batch, N, N]
    
    # Reshape for matrix broadcasting
    #   For matrix multiplication and dimension matching requirement
    #   reshape into column vectors
    weights_exp = np.expand_dims(weights, axis=-1)       # [Batch, N, 1]
    g_diag_exp = np.expand_dims(g_diag, axis=-1)         # [Batch, N, 1]
    
    # Initialization
    x = np.ones([number_of_samples, N, 1])
    tx_power = general_para.tx_power
    output_noise_power = general_para.output_noise_power
    
    # Tx powers assume max power for calculation context
    tx_powers = np.ones([number_of_samples, N, 1]) * tx_power
    
    # Iterative Update (100 iterations)
    # Each step's output shape: [Batch, N, 1]?
    for i in range(100):
        # 1. Compute auxiliary variable z
        # p_x_prod represents actual transmit power P*x
        p_x_prod = x * tx_powers
        
        # Denominator: Interference + Noise
        z_denominator = np.matmul(g_nondiag, p_x_prod) + output_noise_power
        z_numerator = g_diag_exp * p_x_prod
        z = z_numerator / (z_denominator + 1e-12)
        
        # 2. Compute auxiliary variable y
        # Denominator: Total Received Power (Signal + Interference) + Noise
        y_denominator = np.matmul(g, p_x_prod) + output_noise_power
        y_numerator = np.sqrt(z_numerator * weights_exp * (z + 1))
        y = y_numerator / (y_denominator + 1e-12)
        
        # 3. Update primal variable x (Power strategy)
        # Gradient-like step based on Lagrangian
        # g transpose (0,2,1) aligns receiver-to-transmitter interference
        g_transpose = np.transpose(g, (0, 2, 1))
        
        x_denominator = np.matmul(g_transpose, np.power(y, 2)) * tx_powers
        x_numerator = y * np.sqrt(weights_exp * (z + 1) * g_diag_exp * tx_powers)
        
        x_new = np.power(x_numerator / (x_denominator + 1e-12), 2)
        
        # Projection/Clipping to [0, 1]
        x_new[x_new > 1] = 1
        x = x_new
        
    assert np.shape(x) == (number_of_samples, N, 1)
    x_final = np.squeeze(x, axis=-1)

    return x_final


# =============================================================================
# WMMSE (Weighted Minimum Mean Square Error)
# =============================================================================
# Note: function_wmmse_powercontrol.py contains many unused versions of WMMSE;
#   batch_WMMSE2 is the specific one used in main.py).
# This function has been revised since the original one "handles dimensions inconsistently between the initialization phase and the loop phase."
# Refer 【说明1】.md for details.
def batch_WMMSE2(p_int, alpha, H, Pmax, var_noise):
    """
    Weighted MMSE Power Control.
    
    Inputs:
    - p_int: Initial power [Batch, K, 1]
    - alpha: Weights [Batch, K]
    - H: Channel Magnitude [Batch, K, K] (Rx, Tx)
    - Pmax: Max Power (scalar)
    - var_noise: Noise Variance (scalar)
    """
    N_batch, K, _ = p_int.shape
    
    # 1. Align 'b' (Precoder) to Transmitter Dimension: (Batch, 1, Tx)
    # Original code treated it as (Batch, Rx, 1) initially, which was inconsistent.
    b = np.sqrt(p_int) # (Batch, K, 1)
    b = np.transpose(b, (0, 2, 1)) # (Batch, 1, K)
    
    mask = np.eye(K) # (K, K)
    
    # Prepare Weights for Broadcasting
    # alpha: (Batch, K, 1) aligned with Users
    alpha = np.expand_dims(alpha, axis=2) 

    # --- Initialization Step ---
    # Signal: H * b. b broadcasts over Rows (Rx). 
    # H: (Batch, Rx, Tx), b: (Batch, 1, Tx) -> Result: (Batch, Rx, Tx)
    rx_power = H * b 
    rx_power_s = np.square(rx_power)
    
    # Signal Power (Diagonal elements summed over Tx dimension)
    # Note: element-wise mult with identity mask keeps only i=j terms
    valid_rx_power = np.sum(rx_power * mask, axis=2, keepdims=True) # (Batch, K, 1)
    
    # Interference + Noise (Sum over Tx dimension)
    # I_i = Sum_j (H_ij * b_j)^2 + sigma
    interference = np.sum(rx_power_s, axis=2, keepdims=True) + var_noise
    
    # Receiver Filter (f) and MMSE Weight (w)
    # Shapes: (Batch, K, 1)
    f = valid_rx_power / (interference + 1e-12)
    w = 1 / (1 - f * valid_rx_power + 1e-12)

    # --- Iterative Updates ---
    for _ in range(100):
        # 1. Update Transmit Power (b) using Receiver (f)
        # We start with the Dual Channel: H_trans (Tx, Rx)
        # f aligned with Rx: (Batch, 1, K)
        f_row = np.transpose(f, (0, 2, 1)) 
        
        # Dual Received Signal
        # H.mT is matrix transpose (Tx, Rx). f_row broadcasts over Tx.
        # rx_power_dual[b, j, i] = H[b, i, j] * f[b, i]
        rx_power_dual = np.transpose(H, (0, 2, 1)) * f_row
        rx_power_s_dual = np.square(rx_power_dual)
        
        # Dual Signal (Diagonal)
        valid_rx_power_dual = np.sum(rx_power_dual * mask, axis=2, keepdims=True) # (Batch, K, 1)
        
        # Calculate Numerator (bup)
        # bup_j = alpha_j * w_j * (f_j * H_jj)
        bup = alpha * w * valid_rx_power_dual
        
        # Calculate Denominator (bdown)
        # bdown_j = Sum_i ( alpha_i * w_i * (H_ij * f_i)^2 )
        # We need to broadcast alpha/w over the Tx dimension j
        alpha_row = np.transpose(alpha, (0, 2, 1)) # (Batch, 1, K)
        w_row = np.transpose(w, (0, 2, 1))         # (Batch, 1, K)
        
        # Broadcasting: 
        # rx_power_s_dual is (Batch, Tx, Rx).
        # alpha_row/w_row are (Batch, 1, Rx). 
        # Sum over Rx (axis 2).
        bdown = np.sum(alpha_row * w_row * rx_power_s_dual, axis=2, keepdims=True) # (Batch, K, 1)
        
        # Update b (Unconstrained)
        btmp = bup / (bdown + 1e-12)
        
        # Project to Power Constraint
        # b = np.minimum(btmp, np.sqrt(Pmax)) + np.maximum(btmp, 0) - btmp # equivalent to clip
        b = np.clip(btmp, 0, np.sqrt(Pmax))
        
        # 2. Update Receiver (f) and Weight (w) using new Transmit Power (b)
        # Align b to Tx (Columns): (Batch, 1, K)
        b_col = np.transpose(b, (0, 2, 1))
        
        # Forward Received Signal
        rx_power = H * b_col
        rx_power_s = np.square(rx_power)
        
        valid_rx_power = np.sum(rx_power * mask, axis=2, keepdims=True)
        interference = np.sum(rx_power_s, axis=2, keepdims=True) + var_noise
        
        f = valid_rx_power / (interference + 1e-12)
        w = 1 / (1 - f * valid_rx_power + 1e-12)

    # Output Power: Square the precoder b
    p_opt = np.square(b) 
    return np.squeeze(p_opt, axis=2) # Return (Batch, K)



# =============================================================================
# Greedy Baseline
# =============================================================================

# Binary Greedy Heuristic
def simple_greedy(X, AAA, label):
    """
    Binary Greedy Heuristic.
    Selects top-k strongest links, where k is estimated from the FPLinQ solution.
    
    Args:
        X: Channel Gains [Batch, K, K].
        AAA: Weights [Batch, K].
        label: Reference solution (e.g., FPLinQ output) to estimate sparsity [Batch, K].
        
    Returns:
        Y: Binary power allocation [Batch, K].
    """
    n, K, _ = X.shape
    
    # Heuristic: Estimate how many links should be active (threshold)
    # based on the average activity level of the reference label
    thd = int(np.sum(label) / n)
    thd = max(0, min(K, thd)) # Clamp between 0 and K
    
    Y = np.zeros((n, K))
    
    for ii in range(n):
        alpha = AAA[ii, :]
        
        # Calculate weighted signal strength (diagonal)
        # X[ii] is channel matrix, diag extracts h_kk
        H_diag = alpha * np.square(np.diag(X[ii, :, :]))
        
        # Sort indices by strength descending
        xx = np.argsort(H_diag)[::-1]
        
        # Activate the top 'thd' links
        if thd > 0:
            Y[ii, xx[:thd]] = 1
            
    return Y

# =============================================================================
# WMMSE Baseline with QoS Constraints (minimum data rate)
# =============================================================================
# Note: This is a more complex baseline that incorporates QoS constraints into the WMMSE
def batch_WMMSE2_QoS(p_int, alpha, H, Pmax, var_noise, R_min=0.5):
    """
    基于论文 eWMMSE 的纯 D2D 实现（无 cell_mask）：
    利用拉格朗日乘子 (mu) 动态保障最低速率 R_min (QoS)。
    H 为信道幅度矩阵 [Batch, K, K] (Rx, Tx)
    """
    N_batch, K, _ = p_int.shape
    
    # 初始化
    p = np.clip(p_int, 0.0, Pmax)
    v = np.sqrt(p)
    w_alpha = alpha.reshape(N_batch, K, 1).astype(float)
    
    # QoS 乘子 mu 初始化
    mu = np.zeros((N_batch, K, 1))
    lr_mu = 0.5  # 更新步长
    
    EPS = 1e-12 

    for iteration in range(50):
        # 提取对角线直连信道
        h_diag = np.diagonal(H, axis1=1, axis2=2).reshape(N_batch, K, 1)
        
        # 矩阵相乘计算全图接收总功率
        total_rx_p = np.matmul(np.square(H), p) + var_noise
        signal = p * np.square(h_diag)
        
        # 干扰+噪声 (托底防止浮点数越界)
        interference_plus_noise = np.maximum(total_rx_p - signal, var_noise)
        
        u = (h_diag * v) / (total_rx_p + EPS) 
        w = total_rx_p / (interference_plus_noise + EPS)
        
        # 核心：融合 QoS 乘子
        effective_weight = w_alpha + mu
        H_T_sq = np.transpose(np.square(H), (0, 2, 1))
        u_sq_w_eff = np.square(u) * w * effective_weight      
        
        A = np.matmul(H_T_sq, u_sq_w_eff)
        B = effective_weight * w * u * h_diag
        
        v_new = B / (A + EPS)
        v = np.clip(v_new, 0.0, np.sqrt(Pmax))
        p = np.square(v)
        
        # 速率检验与对偶更新
        signal_new = p * np.square(h_diag)
        total_rx_p_new = np.matmul(np.square(H), p) + var_noise
        interference_plus_noise_new = np.maximum(total_rx_p_new - signal_new, var_noise)
        
        rate = np.log2(1.0 + signal_new / (interference_plus_noise_new + EPS))
        mu = np.maximum(0.0, mu + lr_mu * (R_min - rate))

    return np.squeeze(p, axis=-1)