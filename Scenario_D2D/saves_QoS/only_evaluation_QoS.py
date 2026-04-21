import os
import sys
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

# Import core modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import baselines as BL
import model_gnn as MG
import main_supporter as Sapo
import utils_objective as UO
from test_QoS import calculate_metrics  # Reusing your metrics function

def evaluate_and_plot(k_size=50, test_layouts=1000, R_MIN=0.5):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # SAVE_DIR = os.path.join(SCRIPT_DIR, 'saves')
    SAVE_DIR = SCRIPT_DIR
    
    # 1. Load Scalers
    print("Loading scalers...")
    with open(os.path.join(SAVE_DIR, 'scalers_R_MIN.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    dist_scaler = scalers['dist']
    loss_scaler = scalers['loss']
    R_MIN = scalers['R_MIN']

    # 2. Generate Brand New Evaluation Data
    print(f"Generating {test_layouts} new test layouts...")
    ch_test, graph_test_list, dLoss_test, cLoss_test, test_cfg = Sapo.generate_evaluation_dataset(
        K_size=k_size, num_layouts=test_layouts, 
        dist_scaler=dist_scaler, loss_scaler=loss_scaler
    )
    test_loader = DataLoader(graph_test_list, batch_size=test_layouts, shuffle=False)
    var_noise = test_cfg.output_noise_power / test_cfg.tx_power

    # 3. Initialize and Load Models
    print("Loading trained models...")
    gnn_old = MG.IGCNet().to(DEVICE)
    gnn_old.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'gnn_old_K{k_size}.pth')))
    gnn_old.eval()

    gnn_QoS = MG.IGCNet().to(DEVICE)
    gnn_QoS.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'gnn_QoS_K{k_size}.pth')))
    gnn_QoS.eval()

    gnn_QoS_anneal = MG.IGCNet().to(DEVICE)
    gnn_QoS_anneal.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'gnn_QoS_anneal_K{k_size}.pth')))
    gnn_QoS_anneal.eval()

    # 4. Run Baselines
    print("Running WMMSE Baselines...")
    Pini = np.random.rand(test_layouts, k_size, 1)
    Y_wm = BL.batch_WMMSE2(Pini, np.ones([test_layouts, k_size]), np.sqrt(ch_test), 1, var_noise)
    rates_wm = UO.compute_rates(test_cfg, Y_wm, dLoss_test, cLoss_test)
    metrics_wm = calculate_metrics(rates_wm, R_MIN)

    Pini_qos = np.random.rand(test_layouts, k_size, 1)
    Y_wm_qos = BL.batch_WMMSE2_QoS(Pini_qos, np.ones([test_layouts, k_size]), np.sqrt(ch_test), 1, var_noise, R_MIN)
    rates_wm_qos = UO.compute_rates(test_cfg, Y_wm_qos, dLoss_test, cLoss_test)
    metrics_wm_qos = calculate_metrics(rates_wm_qos, R_MIN)

    # 5. Run GNN Inference
    print("Running GNN Inference...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            
            # Old GNN
            out_old = gnn_old(batch)
            Y_old = out_old[:, 2].reshape(-1, k_size).cpu().numpy()
            
            # QoS GNN
            out_QoS = gnn_QoS(batch)
            Y_QoS = out_QoS[:, 2].reshape(-1, k_size).cpu().numpy()
            
            # QoS Anneal GNN
            out_QoS_anneal = gnn_QoS_anneal(batch)
            Y_QoS_anneal = out_QoS_anneal[:, 2].reshape(-1, k_size).cpu().numpy()

    # Calculate Rates and Metrics
    rates_old = UO.compute_rates(test_cfg, Y_old, dLoss_test, cLoss_test)
    metrics_old = calculate_metrics(rates_old, R_MIN)

    rates_QoS = UO.compute_rates(test_cfg, Y_QoS, dLoss_test, cLoss_test)
    metrics_QoS = calculate_metrics(rates_QoS, R_MIN)

    rates_QoS_anneal = UO.compute_rates(test_cfg, Y_QoS_anneal, dLoss_test, cLoss_test)
    metrics_QoS_anneal = calculate_metrics(rates_QoS_anneal, R_MIN)

    # 6. Plot the results
    # (Insert the exact same plotting block from the load_and_plot_only script here)
    # --- Reporting ---
    Sapo.print_qos_metrics({
        'metrics_wm': metrics_wm,
        'metrics_wm_qos': metrics_wm_qos,
        'metrics_old': metrics_old,
        'metrics_QoS': metrics_QoS,
        'metrics_QoS_anneal': metrics_QoS_anneal
    })

    # --- Plotting CDF ---
    Sapo.plot_qos_cdf({
        'metrics_wm': metrics_wm,
        'metrics_wm_qos': metrics_wm_qos,
        'metrics_old': metrics_old,
        'metrics_QoS': metrics_QoS,
        'metrics_QoS_anneal': metrics_QoS_anneal,
        'R_MIN': R_MIN
    }, save_dir=SAVE_DIR)

if __name__ == "__main__":
    evaluate_and_plot(k_size=50, test_layouts=500, R_MIN=0.5)