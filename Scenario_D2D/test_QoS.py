# ==============================================================================
# Filename: test_QoS.py
# Description: This script evaluates the performance of the original GNN and the modified GNN with QoS constraints on a test set of wireless layouts. 
#   It compares their performance against WMMSE baselines (with and without QoS constraints) using key metrics such as constraint violation rate, sum-rate, rate distribution percentiles, and network fairness. 
#   The results are printed in a tabular format and visualized using CDF plots of user data rates.
# ==============================================================================

import os
import pickle

import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

# Import Core Modules from your project 
import config_system as CS
import utils_objective as UO
import baselines as BL
import model_gnn as MG
import main_supporter as Sapo

def calculate_metrics(rates, r_min):
    """
    Calculates the 4 key metrics for a given set of data rates.
    rates shape: [Batch, K]
    """
    K = rates.shape[1]
    rates_flat = rates.flatten()
    
    # 1. Constraint Violation Rate (Outage Probability)
    violation_rate = np.mean(rates_flat < r_min) * 100
    
    # 2. Sum-Rate (Average across layouts)
    sum_rate = np.mean(np.sum(rates, axis=1))
    
    # 3. Rate Distribution (Percentiles)
    p05 = np.percentile(rates_flat, 5)
    p10 = np.percentile(rates_flat, 10)
    p20 = np.percentile(rates_flat, 20)
    p40 = np.percentile(rates_flat, 40)
    
    # 4. Network Fairness
    # Variance
    rate_variance = np.var(rates_flat)
    # Jain's Fairness Index (computed per layout, then averaged)
    # Formula: (Sum x_i)^2 / (n * Sum x_i^2)
    jains_index = np.mean(
        np.square(np.sum(rates, axis=1)) / (K * np.sum(np.square(rates), axis=1) + 1e-12)
    )
    
    return {
        'violation_rate': violation_rate,
        'sum_rate': sum_rate,
        'p05': p05, 'p10': p10, 'p20': p20, 'p40': p40,
        'variance': rate_variance,
        'jains': jains_index,
        'rates_flat': rates_flat # Keep for CDF plotting
    }

def run_test():
    # --- Save Feature Toggle ---
    ENABLE_SAVING = True  
    if ENABLE_SAVING:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        SAVE_DIR = os.path.join(SCRIPT_DIR, 'saves_QoS')
        os.makedirs(SAVE_DIR, exist_ok=True) 
        print(f"Save feature ENABLED. Files will be saved to: {SAVE_DIR}")
    else:
        print("Save feature DISABLED.")
        
    # --- Configuration ---
    # GNN Hyperparameters
    lr_GNN = 3e-3
    epochs_GNN = 40
    TRAIN_K_GNN = 50
    TRAIN_LAYOUTS = 5000 
    TEST_LAYOUTS = 1000 
    # QoS Constraint
    R_MIN = 0.5 # Minimum required data rate (bits/s/Hz)
    penalty_weight = 10 # For modified GNN

    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(f"Device: {DEVICE}")

    # --- Phase 1: Data Generation & Scalers ---
    print("\nGenerating Data & Fitting Scalers...")
    train_config = CS.init_parameters(n_links=TRAIN_K_GNN) 
    var_noise = train_config.output_noise_power / train_config.tx_power 
    
    _, train_dists = CS.generate_layouts(train_config, TRAIN_LAYOUTS)
    train_path = CS.compute_path_losses(train_config, train_dists)
    train_ch_final = CS.add_fast_fading(CS.add_shadowing(train_path)) 
    train_losses_proc = CS.proc_train_losses(train_path, train_ch_final)
    
    dist_scaler = Sapo.WirelessScaler() 
    loss_scaler = Sapo.WirelessScaler() 
    dist_scaler.fit(1.0 / train_dists)
    loss_scaler.fit(np.sqrt(train_ch_final))

    # --- Save Scalers ---
    if ENABLE_SAVING:
        print("Saving fitted scalers and R_MIN...")
        with open(os.path.join(SAVE_DIR, f'scalers_R_MIN.pkl'), 'wb') as f:
            pickle.dump({'dist': dist_scaler, 'loss': loss_scaler, 'R_MIN': R_MIN}, f)

    norm_tr_dists = dist_scaler.transform(1.0 / train_dists)
    norm_tr_losses = loss_scaler.transform(np.sqrt(train_ch_final))
    
    train_data_list = MG.proc_data(train_losses_proc, train_dists, norm_tr_dists, norm_tr_losses, TRAIN_K_GNN) 
    train_loader = DataLoader(train_data_list, batch_size=64, shuffle=True) 

    # Generate Test Data
    ch_test, graph_test_list, dLoss_test, cLoss_test, test_cfg = Sapo.generate_evaluation_dataset(
        K_size=TRAIN_K_GNN, num_layouts=TEST_LAYOUTS, dist_scaler=dist_scaler, loss_scaler=loss_scaler
    )
    test_loader = DataLoader(graph_test_list, batch_size=TEST_LAYOUTS, shuffle=False) 

    # --- Phase 2: Train Both GNN Models ---
    print("\nTraining Original GNN...")
    gnn_old = MG.IGCNet().to(DEVICE) 
    opt_old = torch.optim.Adam(gnn_old.parameters(), lr=lr_GNN)
    for epoch in range(epochs_GNN):
        MG.train_epoch(gnn_old, train_loader, opt_old, DEVICE, TRAIN_K_GNN, var_noise) 
        
    print("Training Modified GNN (Constraint-Aware)...")
    gnn_QoS = MG.IGCNet().to(DEVICE) 
    opt_QoS = torch.optim.Adam(gnn_QoS.parameters(), lr=lr_GNN)
    for epoch in range(epochs_GNN):
        MG.train_epoch_QoS(gnn_QoS, train_loader, opt_QoS, DEVICE, TRAIN_K_GNN, var_noise, R_MIN, penalty_weight=penalty_weight)

    print("Training Modified GNN (Constraint-Aware) with penalty annealing...")
    gnn_QoS_anneal = MG.IGCNet().to(DEVICE)
    opt_QoS_anneal = torch.optim.Adam(gnn_QoS_anneal.parameters(), lr=lr_GNN)
    for epoch in range(epochs_GNN):
        MG.train_epoch_QoS(gnn_QoS_anneal, train_loader, opt_QoS_anneal, DEVICE, TRAIN_K_GNN, var_noise, R_MIN, penalty_weight=penalty_weight*epoch/epochs_GNN)

    # --- Save Trained Models ---
    if ENABLE_SAVING:
        print("\nSaving trained GNN models...")
        torch.save(gnn_old.state_dict(), os.path.join(SAVE_DIR, f'gnn_old_K{TRAIN_K_GNN}.pth'))
        torch.save(gnn_QoS.state_dict(), os.path.join(SAVE_DIR, f'gnn_QoS_K{TRAIN_K_GNN}.pth'))
        torch.save(gnn_QoS_anneal.state_dict(), os.path.join(SAVE_DIR, f'gnn_QoS_anneal_K{TRAIN_K_GNN}.pth'))

    # --- Phase 3: Evaluation ---
    print(f"\nEvaluating on Test Set (K={TRAIN_K_GNN}, r_min={R_MIN})...")
    
    K = TRAIN_K_GNN

    # 0. WMMSE Baseline
    Pini = np.random.rand(TEST_LAYOUTS, K, 1) 
    Y_wm = BL.batch_WMMSE2(Pini, np.ones([TEST_LAYOUTS, K]), np.sqrt(ch_test), 1, var_noise) 
    rates_wm = UO.compute_rates(test_cfg, Y_wm, dLoss_test, cLoss_test)
    metrics_wm = calculate_metrics(rates_wm, R_MIN)

    # 1. WMMSE with QoS Constraints
    Pini_qos = np.random.rand(TEST_LAYOUTS, K, 1)
    Y_wm_qos = BL.batch_WMMSE2_QoS(Pini_qos, np.ones([TEST_LAYOUTS, K]), np.sqrt(ch_test), 1, var_noise, R_MIN)
    rates_wm_qos = UO.compute_rates(test_cfg, Y_wm_qos, dLoss_test, cLoss_test)
    metrics_wm_qos = calculate_metrics(rates_wm_qos, R_MIN)

    # 2. Original GNN
    gnn_old.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out_old = gnn_old(batch) 
        Y_old = out_old[:, 2].reshape(-1, K).cpu().numpy() 
    rates_old = UO.compute_rates(test_cfg, Y_old, dLoss_test, cLoss_test)
    metrics_old = calculate_metrics(rates_old, R_MIN)

    # 3. Modified GNN
    gnn_QoS.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out_QoS = gnn_QoS(batch)
        Y_QoS = out_QoS[:, 2].reshape(-1, K).cpu().numpy()
    rates_QoS = UO.compute_rates(test_cfg, Y_QoS, dLoss_test, cLoss_test)
    metrics_QoS = calculate_metrics(rates_QoS, R_MIN)

    # 4. Modified GNN with penalty annealing
    gnn_QoS_anneal.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out_QoS_anneal = gnn_QoS_anneal(batch)
        Y_QoS_anneal = out_QoS_anneal[:, 2].reshape(-1, K).cpu().numpy()
    rates_QoS_anneal = UO.compute_rates(test_cfg, Y_QoS_anneal, dLoss_test, cLoss_test)
    metrics_QoS_anneal = calculate_metrics(rates_QoS_anneal, R_MIN)

    # --- Build Results Dictionary ---
    qos_results = {
            'K': TRAIN_K_GNN,
            'R_MIN': R_MIN,
            'metrics_wm': metrics_wm,
            'metrics_wm_qos': metrics_wm_qos,
            'metrics_old': metrics_old,
            'metrics_QoS': metrics_QoS,
            'metrics_QoS_anneal': metrics_QoS_anneal
        }
    # --- Save Simulation Results ---
    if ENABLE_SAVING:
        print("\nSaving QoS simulation results...")
        with open(os.path.join(SAVE_DIR, f'qos_results_K{TRAIN_K_GNN}.pkl'), 'wb') as f:
            pickle.dump(qos_results, f)

    # --- Phase 4: Reporting ---
    Sapo.print_qos_metrics(qos_results)
    # --- Plotting CDF ---
    Sapo.plot_qos_cdf(qos_results)

if __name__ == "__main__":
    run_test()

