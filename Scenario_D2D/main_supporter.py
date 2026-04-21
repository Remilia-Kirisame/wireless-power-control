# ==============================================================================
# Filename: main_supporter.py
# Description: Helper functions and classes for main execution.
#              Includes data generation, normalization, and plotting.
# ==============================================================================

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

# Import project modules
import config_system as CS
import utils_objective as UO
import baselines as BL
import model_gnn as MG
import model_dnn as MD

# =============================================================================
# 1. Helper Class: Wireless Scaler
# =============================================================================
class WirelessScaler:
    """
    Handles Z-score normalization. 
    Fits on training data statistics and applies them to test data.
    """
    def __init__(self):
        self.diag_mean = None
        self.diag_var = None
        self.off_mean = None
        self.off_var = None

    def fit(self, X):
        """Calculates mean/var from Training Data (X)."""
        L, K, _ = X.shape
        mask = np.eye(K)
        
        # Direct Links (Diagonal)
        diag_H = X * mask
        self.diag_mean = np.sum(diag_H) / (L * K)
        self.diag_var  = np.sqrt(np.sum(np.square(diag_H)) / (L * K))

        # Interference Links (Off-Diagonal)
        off_diag = X - diag_H
        self.off_mean = np.sum(off_diag) / (L * K * (K - 1))
        self.off_var  = np.sqrt(np.sum(np.square(off_diag)) / (L * K * (K - 1)))

    def transform(self, X):
        """Applies the calculated stats to X."""
        if self.diag_mean is None:
            raise ValueError("Scaler not fitted! Call fit() with training data first.")
            
        L, K, _ = X.shape
        mask = np.eye(K)
        
        # Normalize Diagonal
        diag_H = X * mask
        norm_diag = (diag_H - self.diag_mean) / (self.diag_var + 1e-12)
        
        # Normalize Off-Diagonal
        off_diag = X - diag_H
        norm_off = (off_diag - self.off_mean) / (self.off_var + 1e-12)
        
        # Combine (Diagonal + Off-Diagonal)
        return (norm_diag * mask) + (norm_off - norm_off * mask)


# =============================================================================
# 2. Helper Function: Evaluation Data Generator
# =============================================================================
def generate_evaluation_dataset(K_size, num_layouts, RULER=False, dist_scaler=None, loss_scaler=None, density=None):
    """
    Generates a complete dataset (channel losses and graph objects) for testing.
    Uses pre-fitted scalers to ensure valid normalization.
    """
    # A. Init Config
    conf = CS.init_parameters(n_links=K_size)
    if density is not None:
        conf.field_length = int(np.sqrt(density * K_size))
        
    # B. Physics: Generate Geometry and Fading
    _, dists = CS.generate_layouts(conf, num_layouts)
    path_losses = CS.compute_path_losses(conf, dists)
    channel_losses = CS.add_shadowing(path_losses)
    channel_losses = CS.add_fast_fading(channel_losses)
    
    # C. Normalization (Using passed Scalers)
    # Note: We must preprocess inputs (1/dist and sqrt(loss)) to match training logic
    norm_dists = dist_scaler.transform(1.0 / dists)
    norm_losses = loss_scaler.transform(np.sqrt(channel_losses))

    # D. Graph Construction for GNN
    graph_data_list = MG.proc_data(channel_losses, dists, norm_dists, norm_losses, K_size)
    
    # E. Extract Evaluation Helpers (Diagonal/Cross for Rates)
    d_losses = UO.get_directLink_channel_losses(channel_losses)
    c_losses = UO.get_crossLink_channel_losses(channel_losses)
    
    return channel_losses, graph_data_list, d_losses, c_losses, conf


# =============================================================================
# 3. Helper Function: Specialized DNN Trainer
# =============================================================================
def train_specialized_dnn(K, train_layouts, loss_scaler, dnn_config, device, density=None, lr=1e-3):
    """
    Trains a DNN specifically for a given problem size K (as per original logic).
    """
    print(f'   [DNN] Training specialized model for K={K}...')
    
    # Init Config for this K
    cfg_K = CS.init_parameters(n_links=K)
    if density is not None:
        cfg_K.field_length = int(np.sqrt(density * K))
    var_K = cfg_K.output_noise_power / cfg_K.tx_power
    
    # Generate Training Data for this DNN
    _, tr_dists = CS.generate_layouts(cfg_K, train_layouts)
    tr_path = CS.compute_path_losses(cfg_K, tr_dists)
    tr_ch = CS.add_shadowing(tr_path)
    tr_ch = CS.add_fast_fading(tr_ch)
    train_losses_proc = CS.proc_train_losses(tr_path, tr_ch)
    
    # # OLD WAY (Log Scale) --- IGNORE ---
    # # Preprocess Input (Log scale for DNN)
    # X_train = 10 * np.log10(tr_ch + 1e-12).reshape(tr_ch.shape[0], -1)
    
    # NEW WAY (Z-score Normalisation)
    X_norm = loss_scaler.transform(np.sqrt(train_losses_proc))
    X_train = X_norm.reshape(X_norm.shape[0], -1)

    # Generate Targets using WMMSE
    Pini = np.random.rand(train_layouts, K, 1)
    Y_train = BL.batch_WMMSE2(Pini, np.ones([train_layouts, K]), np.sqrt(train_losses_proc), 1, var_K)
    
    # Train
    # model = MD.train_dnn_torch(
    #     X_train=X_train, Y_train=Y_train,
    #     epochs=dnn_config['epochs'],
    #     batch_size=dnn_config['batch_size'],
    #     hidden_layers=dnn_config['shape'],
    #     lr=lr,
    #     device=device, cfg=cfg_K
    # )
    # Using Unsupervised Training
    model = MD.train_dnn_unsup(
        X_train=X_train, H_train=train_losses_proc,
        epochs=dnn_config['epochs'],
        batch_size=dnn_config['batch_size'],
        hidden_layers=dnn_config['shape'],
        lr=lr,
        device=device, cfg=cfg_K    
    )
    return model


# =============================================================================
# 4. Helper Function: Plotting
# =============================================================================
def plot_results(res, save_dir=None):
    """Helper to plot results at the end. If save_dir is given, writes PNGs there."""
    # Sum Rate Plot
    fig_sr = plt.figure()
    plt.plot(res['K'], res['FPLinQ'], marker='o', label='FPLinQ')
    plt.plot(res['K'], res['WMMSE'],  marker='^', label='WMMSE')
    plt.plot(res['K'], res['Greedy'], marker='s', label='Greedy')
    plt.plot(res['K'], res['GNN'],    marker='x', label='GNN (Ours)')
    plt.plot(res['K'], res['DNN'],    marker='v', label='DNN (Ours)')
    plt.xlabel('Number of links K')
    plt.ylabel('Average sum-rate')
    plt.title('Algorithm Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    if save_dir is not None:
        fig_sr.savefig(os.path.join(save_dir, 'd2d_sumrate.png'), dpi=130, bbox_inches='tight')

    # Time Plot
    fig_time = plt.figure()
    plt.semilogy(res['K'], res['Time_FPL'], marker='o', label='FPLinQ')
    plt.semilogy(res['K'], res['Time_WMM'], marker='^', label='WMMSE')
    plt.semilogy(res['K'], res['Time_Grd'], marker='s', label='Greedy')
    plt.semilogy(res['K'], res['Time_GNN'], marker='x', label='GNN')
    plt.semilogy(res['K'], res['Time_DNN'], marker='v', label='DNN')
    plt.xlabel('Number of links K')
    plt.ylabel('Runtime (s)')
    plt.title('Computation Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    if save_dir is not None:
        fig_time.savefig(os.path.join(save_dir, 'd2d_time.png'), dpi=130, bbox_inches='tight')
    plt.show()


# =============================================================================
# 5. Helper Functions: QoS Reporting and Plotting
# =============================================================================

def print_qos_metrics(results):
    """Prints a tabular comparison of QoS metrics."""
    metrics_wm = results['metrics_wm']
    metrics_wm_qos = results['metrics_wm_qos']
    metrics_old = results['metrics_old']
    metrics_QoS = results['metrics_QoS']
    metrics_QoS_anneal = results['metrics_QoS_anneal']

    print("\n" + "="*100)
    print(f"{'Metric':<25} | {'orig WMMSE':<10} | {'WMMSE(QoS)':<10} | {'Old GNN':<10} | {'QoS GNN':<10} | {'QoS GNN (Anneal)':<15}")
    print("-" * 100)

    print(f"{'Violation Rate (< r_min)':<25} | {metrics_wm['violation_rate']:>9.2f}% | {metrics_wm_qos['violation_rate']:>9.2f}% | {metrics_old['violation_rate']:>9.2f}% | {metrics_QoS['violation_rate']:>9.2f}% | {metrics_QoS_anneal['violation_rate']:>9.2f}%")
    print(f"{'Sum-Rate':<25} | {metrics_wm['sum_rate']:>10.4f} | {metrics_wm_qos['sum_rate']:>10.4f} | {metrics_old['sum_rate']:>10.4f} | {metrics_QoS['sum_rate']:>10.4f} | {metrics_QoS_anneal['sum_rate']:>10.4f}")
    print(f"{'Jain Fairness Index':<25} | {metrics_wm['jains']:>10.4f} | {metrics_wm_qos['jains']:>10.4f} | {metrics_old['jains']:>10.4f} | {metrics_QoS['jains']:>10.4f} | {metrics_QoS_anneal['jains']:>10.4f}")
    print(f"{'Rate Variance':<25} | {metrics_wm['variance']:>10.4f} | {metrics_wm_qos['variance']:>10.4f} | {metrics_old['variance']:>10.4f} | {metrics_QoS['variance']:>10.4f} | {metrics_QoS_anneal['variance']:>10.4f}")
    print(f"{'5th Percentile Rate':<25} | {metrics_wm['p05']:>10.4f} | {metrics_wm_qos['p05']:>10.4f} | {metrics_old['p05']:>10.4f} | {metrics_QoS['p05']:>10.4f} | {metrics_QoS_anneal['p05']:>10.4f}")
    print(f"{'20th Percentile Rate':<25} | {metrics_wm['p20']:>10.4f} | {metrics_wm_qos['p20']:>10.4f} | {metrics_old['p20']:>10.4f} | {metrics_QoS['p20']:>10.4f} | {metrics_QoS_anneal['p20']:>10.4f}")
    print(f"{'40th Percentile Rate':<25} | {metrics_wm['p40']:>10.4f} | {metrics_wm_qos['p40']:>10.4f} | {metrics_old['p40']:>10.4f} | {metrics_QoS['p40']:>10.4f} | {metrics_QoS_anneal['p40']:>10.4f}")
    print("="*100)


def plot_qos_cdf(results, save_dir=None):
    """Plots the CDF of user data rates from QoS results. If save_dir is given, writes PNG there."""
    metrics_wm = results['metrics_wm']
    metrics_wm_qos = results['metrics_wm_qos']
    metrics_old = results['metrics_old']
    metrics_QoS = results['metrics_QoS']
    metrics_QoS_anneal = results['metrics_QoS_anneal']
    R_MIN = results['R_MIN']

    plt.figure(figsize=(8, 5))
    
    # Sort data for CDF
    sort_wm = np.sort(metrics_wm['rates_flat'])
    sort_wm_qos = np.sort(metrics_wm_qos['rates_flat'])
    sort_old = np.sort(metrics_old['rates_flat'])
    sort_QoS = np.sort(metrics_QoS['rates_flat'])
    sort_QoS_anneal = np.sort(metrics_QoS_anneal['rates_flat'])
    p = np.arange(len(sort_wm)) / float(len(sort_wm) - 1)
    
    plt.plot(sort_wm, p, label='WMMSE (Baseline)', linestyle=':')
    plt.plot(sort_wm_qos, p, label='WMMSE (QoS Constraint)', linestyle='-.')
    plt.plot(sort_old, p, label='Original GNN', linestyle='--')
    plt.plot(sort_QoS, p, label='Modified GNN (Constraint)', linewidth=2)
    plt.plot(sort_QoS_anneal, p, label='Modified GNN with Penalty Annealing', linestyle='-.', linewidth=2)
    
    plt.axvline(x=R_MIN, color='r', linestyle='-.', label=f'r_min = {R_MIN}')
    
    plt.xlim(0, max(np.percentile(sort_wm, 95), R_MIN*3)) # Focus on the lower/middle end
    plt.xlabel('Data Rate (bits/s/Hz)')
    plt.ylabel('Cumulative Probability (CDF)')
    plt.title('CDF of User Data Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'qos_cdf.png'), dpi=130, bbox_inches='tight')
    plt.show()