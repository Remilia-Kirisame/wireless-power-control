# ==============================================================================
# Filename: resume_evaluation.py
# Description: Load trained GNN model + scalers, generate fresh evaluation data,
#              run all methods (Naive / WMMSE / GNN), print and plot results.
#              Reproduces Phase 3 + 4 of main.py without retraining.
#
# Usage:  cd Framework_ver_4.0/save_main && python resume_evaluation.py
# ==============================================================================

import os
import sys
import pickle
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Add parent directory (Framework_ver_4.0) to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

import config_system as CS
import utils_objective as UO
import baselines as BL
import model_gnn as MG
from main import _print_scenario, print_sweep_tables, plot_sweep_results


def main():
    # =========================================================================
    #  Load saved artifacts
    # =========================================================================
    with open(os.path.join(SCRIPT_DIR, 'hyperparams.pkl'), 'rb') as f:
        hp = pickle.load(f)
    with open(os.path.join(SCRIPT_DIR, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    dist_scaler = scalers['dist']
    loss_scaler = scalers['loss']

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load GNN model
    gnn_model = MG.IGCNet().to(DEVICE)
    gnn_model.load_state_dict(torch.load(
        os.path.join(SCRIPT_DIR, 'gnn_model.pth'), map_location=DEVICE, weights_only=True))
    gnn_model.eval()

    # Unpack hyperparameters
    TRAIN_N_BLUE   = hp['TRAIN_N_BLUE']
    TRAIN_N_YELLOW = hp['TRAIN_N_YELLOW']
    TRAIN_N_GREEN  = hp['TRAIN_N_GREEN']
    FIELD_LENGTH   = hp['FIELD_LENGTH']
    TEST_LAYOUTS   = hp['TEST_LAYOUTS']
    SHUFFLE_CHANNELS = hp['SHUFFLE_CHANNELS']
    SINR_MIN       = hp['SINR_MIN']
    SINR_MIN_DB    = hp['SINR_MIN_DB']
    PENALTY_WEIGHT = hp['PENALTY_WEIGHT']
    WMMSE_ALPHA_YELLOW = hp['WMMSE_ALPHA_YELLOW']
    SWEEP_B_LIST   = hp['SWEEP_B_LIST']
    SWEEP_B_MY     = hp['SWEEP_B_MY']
    SWEEP_B_MG     = hp['SWEEP_B_MG']
    SWEEP_M_LIST   = hp['SWEEP_M_LIST']
    SWEEP_M_B      = hp['SWEEP_M_B']

    M_train = TRAIN_N_YELLOW + TRAIN_N_GREEN
    K_train = TRAIN_N_BLUE * M_train

    # Banner
    print(f"\n{'='*60}")
    print(f"  [Resume Evaluation] JSAC Wireless Power Control")
    print(f"  Device: {DEVICE}")
    print(f"  Training topology: B={TRAIN_N_BLUE}, M_y={TRAIN_N_YELLOW}, "
          f"M_g={TRAIN_N_GREEN}  ->  K={K_train} links")
    print(f"  Shuffle channels: {SHUFFLE_CHANNELS}")
    print(f"  SINR_min = {SINR_MIN_DB} dB ({SINR_MIN:.2f} linear), "
          f"penalty = {PENALTY_WEIGHT}")
    print(f"{'='*60}")

    # =========================================================================
    #  Evaluate scenario (same logic as main.py Phase 3)
    # =========================================================================
    def evaluate_scenario(config):
        K = config.n_links
        B = config.n_blue
        M = config.n_links_per_blue

        # Generate fresh evaluation data
        _, dists, group_ids, link_types = CS.generate_layouts_jsac(config, TEST_LAYOUTS)
        _, interf_mask, green_mask, yellow_mask = CS.jsac_metadata(
            group_ids, link_types, M, shuffle_channels=SHUFFLE_CHANNELS)
        _, channel_losses = CS.compute_channel_losses_jsac(config, dists, interf_mask)

        # Normalize using pre-fitted scalers
        norm_dists  = dist_scaler.transform(1.0 / dists, interf_mask, yellow_mask, green_mask)
        norm_losses = loss_scaler.transform(
            np.sqrt(channel_losses), interf_mask, yellow_mask, green_mask)
        graph_data = MG.proc_data(
            channel_losses, dists, norm_dists, norm_losses, K,
            interf_mask, group_ids, green_mask, yellow_mask,
        )

        # Evaluation arrays
        dLoss = UO.get_directLink_channel_losses(channel_losses)
        cLoss = UO.get_crossLink_channel_losses(channel_losses)
        var_eval = config.output_noise_power / config.tx_power

        # 1. Naive Equal Power
        t0 = time.time()
        Y_naive = np.tile(BL.naive_equal_power(config, group_ids), (TEST_LAYOUTS, 1))
        t_naive = time.time() - t0

        # 2. WMMSE
        t0 = time.time()
        alpha = np.ones((TEST_LAYOUTS, K))
        alpha[:, yellow_mask] = WMMSE_ALPHA_YELLOW
        H = np.sqrt(channel_losses)
        Y_wmmse = BL.batch_WMMSE2_JSAC(
            TEST_LAYOUTS, alpha, H, 1.0, var_eval,
            group_ids, SINR_MIN, yellow_mask,
        )
        t_wmmse = time.time() - t0

        # 3. GNN
        loader = DataLoader(graph_data, batch_size=TEST_LAYOUTS, shuffle=False)
        t0 = time.time()
        Y_gnn = None
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                out = gnn_model(batch)
                raw_p = out[:, 3]
                p = MG.apply_group_softmax(raw_p, batch.group_ids, batch.batch, B)
                Y_gnn = p.reshape(-1, K).cpu().numpy()
        t_gnn = time.time() - t0

        # Compute metrics
        scenario = {}
        for name, Y in [('Naive', Y_naive), ('WMMSE', Y_wmmse), ('GNN', Y_gnn)]:
            green_sr = UO.compute_green_sumrate(config, Y, dLoss, cLoss, green_mask)
            viol_rate, _ = UO.compute_yellow_sinr_violation(
                config, Y, dLoss, cLoss, yellow_mask, SINR_MIN)
            scenario[name] = {'green_sr': green_sr, 'viol_rate': viol_rate}
        scenario['Naive']['time']  = t_naive
        scenario['WMMSE']['time']  = t_wmmse
        scenario['GNN']['time']    = t_gnn
        return scenario

    # =========================================================================
    #  Run sweeps (same structure as main.py)
    # =========================================================================
    print("\n=== Evaluation ===")

    # ---- Axis 1: Vary B ----
    print("\n--- Sweep: Vary B (Blue cars) ---")
    print(f"    Fixed M_y={SWEEP_B_MY}, M_g={SWEEP_B_MG}")

    results_B = {'B': [], 'K': []}
    for method in ['Naive', 'WMMSE', 'GNN']:
        results_B[f'{method}_sr']   = []
        results_B[f'{method}_viol'] = []
        results_B[f'{method}_time'] = []

    for B_val in SWEEP_B_LIST:
        cfg = CS.init_parameters_JSAC(
            n_blue=B_val,
            n_yellow_per_blue=SWEEP_B_MY,
            n_green_per_blue=SWEEP_B_MG,
            field_length=FIELD_LENGTH,
        )
        scenario = evaluate_scenario(cfg)
        _print_scenario(f"B={B_val} (K={cfg.n_links})", scenario)

        results_B['B'].append(B_val)
        results_B['K'].append(cfg.n_links)
        for method in ['Naive', 'WMMSE', 'GNN']:
            results_B[f'{method}_sr'].append(scenario[method]['green_sr'])
            results_B[f'{method}_viol'].append(scenario[method]['viol_rate'])
            results_B[f'{method}_time'].append(scenario[method]['time'])

    # ---- Axis 2: Vary M ----
    print(f"\n--- Sweep: Vary (M_y, M_g) per Blue car ---")
    print(f"    Fixed B={SWEEP_M_B}")

    results_M = {'M_y': [], 'M_g': [], 'M': [], 'K': []}
    for method in ['Naive', 'WMMSE', 'GNN']:
        results_M[f'{method}_sr']   = []
        results_M[f'{method}_viol'] = []
        results_M[f'{method}_time'] = []

    for (my_val, mg_val) in SWEEP_M_LIST:
        cfg = CS.init_parameters_JSAC(
            n_blue=SWEEP_M_B,
            n_yellow_per_blue=my_val,
            n_green_per_blue=mg_val,
            field_length=FIELD_LENGTH,
        )
        scenario = evaluate_scenario(cfg)
        _print_scenario(f"M_y={my_val}, M_g={mg_val} (K={cfg.n_links})", scenario)

        results_M['M_y'].append(my_val)
        results_M['M_g'].append(mg_val)
        results_M['M'].append(my_val + mg_val)
        results_M['K'].append(cfg.n_links)
        for method in ['Naive', 'WMMSE', 'GNN']:
            results_M[f'{method}_sr'].append(scenario[method]['green_sr'])
            results_M[f'{method}_viol'].append(scenario[method]['viol_rate'])
            results_M[f'{method}_time'].append(scenario[method]['time'])

    # Save fresh results (overwrite previous)
    with open(os.path.join(SCRIPT_DIR, 'results_sweep_B.pkl'), 'wb') as f:
        pickle.dump(results_B, f)
    with open(os.path.join(SCRIPT_DIR, 'results_sweep_M.pkl'), 'wb') as f:
        pickle.dump(results_M, f)

    # =========================================================================
    #  Plot
    # =========================================================================
    print("\n=== Plotting ===")
    plot_sweep_results(results_B, results_M, save_dir=SCRIPT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
