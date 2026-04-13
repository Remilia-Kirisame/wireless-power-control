# ==============================================================================
# Filename: resume_evaluation.py
# Description: Load trained GNN model + scalers, generate fresh evaluation data,
#              run all methods (Equal Power / WMMSE / GNN), print and plot results.
#              Reproduces Phase 3–5 of test_JSAC.py without retraining.
#
# Usage:  cd Framework_ver_4.0/save_test && python resume_evaluation.py
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
import main_supporter as Sapo
from test_JSAC import (calculate_metrics_jsac, build_results_summary,
                       print_metrics_summary, plot_dashboard, plot_deep_dive,
                       plot_layout_snapshots)


def main():
    # --- Plot controls ---
    SHOW_PLOTS          = False   # True = display, False = save only
    N_SAMPLE_LAYOUTS    = 2      # snapshots: how many random layouts to draw
    SNAPSHOT_NODE_SIZE  = 100    # snapshots: base marker size for cars

    # =========================================================================
    #  Load saved artifacts
    # =========================================================================
    with open(os.path.join(SCRIPT_DIR, 'hyperparams.pkl'), 'rb') as f:
        hp = pickle.load(f)
    with open(os.path.join(SCRIPT_DIR, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    with open(os.path.join(SCRIPT_DIR, 'train_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)

    dist_scaler = scalers['dist']
    loss_scaler = scalers['loss']

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load GNN model
    gnn_model = MG.IGCNet().to(DEVICE)
    gnn_model.load_state_dict(torch.load(
        os.path.join(SCRIPT_DIR, 'gnn_model.pth'), map_location=DEVICE, weights_only=True))
    gnn_model.eval()

    # Unpack hyperparameters
    N_BLUE     = hp['N_BLUE']
    N_YELLOW   = hp['N_YELLOW']
    N_GREEN    = hp['N_GREEN']
    FIELD_LENGTH = hp['FIELD_LENGTH']
    TEST_LAYOUTS = hp['TEST_LAYOUTS']
    SINR_MIN   = hp['SINR_MIN']
    SINR_MIN_DB = hp['SINR_MIN_DB']
    PENALTY_WEIGHT = hp['PENALTY_WEIGHT']
    WMMSE_ALPHA_YELLOW = hp['WMMSE_ALPHA_YELLOW']

    # Reconstruct config
    para = CS.init_parameters_JSAC(
        n_blue=N_BLUE,
        n_yellow_per_blue=N_YELLOW,
        n_green_per_blue=N_GREEN,
        field_length=FIELD_LENGTH,
    )
    K = para.n_links
    M = para.n_links_per_blue

    # Reuse training metadata (masks are topology-dependent, not data-dependent)
    green_mask  = meta['green_mask']
    yellow_mask = meta['yellow_mask']
    group_ids   = meta['group_ids']
    interf_mask = meta['interf_mask']

    # Banner
    print(f"\n{'='*60}")
    print(f"  [Resume Evaluation] JSAC Single-Scenario")
    print(f"  Device: {DEVICE}")
    print(f"  Topology: B={N_BLUE}, M_y={N_YELLOW}, "
          f"M_g={N_GREEN}  ->  K={K} links")
    print(f"  SINR_min = {SINR_MIN_DB} dB ({SINR_MIN:.2f} linear), "
          f"penalty = {PENALTY_WEIGHT}")
    print(f"{'='*60}")

    # =========================================================================
    #  Generate fresh evaluation data
    # =========================================================================
    print("\nGenerating fresh evaluation data...")
    ch_test, test_graph, dLoss_test, cLoss_test, layouts_test, _ = \
        Sapo.generate_evaluation_dataset(
            para, TEST_LAYOUTS, dist_scaler, loss_scaler,
            interf_mask, group_ids, green_mask, yellow_mask,
        )
    test_loader = DataLoader(test_graph, batch_size=TEST_LAYOUTS, shuffle=False)

    # =========================================================================
    #  Evaluate all methods
    # =========================================================================
    print("Evaluating...")
    allocs_dict = {}

    # 1. Naive Equal Power
    t0 = time.time()
    allocs_eq = np.tile(BL.naive_equal_power(para, group_ids), (TEST_LAYOUTS, 1))
    t_eq = time.time() - t0
    allocs_dict['Equal Power'] = allocs_eq

    # 2. WMMSE
    t0 = time.time()
    alpha = np.ones((TEST_LAYOUTS, K))
    alpha[:, yellow_mask] = WMMSE_ALPHA_YELLOW
    H_test = np.sqrt(ch_test)
    allocs_wm = BL.batch_WMMSE2_JSAC(
        TEST_LAYOUTS, alpha, H_test, 1.0,
        para.output_noise_power / para.tx_power,
        group_ids, SINR_MIN, yellow_mask,
    )
    t_wm = time.time() - t0
    allocs_dict['WMMSE'] = allocs_wm

    # 3. GNN
    t0 = time.time()
    gnn_powers = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out = gnn_model(batch)
            raw_p = out[:, 3]
            p = MG.apply_group_softmax(raw_p, batch.group_ids, batch.batch, N_BLUE)
            gnn_powers.append(p.reshape(-1, K).cpu().numpy())
    allocs_gnn = np.concatenate(gnn_powers, axis=0)
    t_gnn = time.time() - t0
    allocs_dict['GNN'] = allocs_gnn

    # Compute metrics
    all_metrics = {}
    timings = {'Equal Power': t_eq, 'WMMSE': t_wm, 'GNN': t_gnn}
    for name, allocs in allocs_dict.items():
        all_metrics[name] = calculate_metrics_jsac(
            para, allocs, dLoss_test, cLoss_test,
            green_mask, yellow_mask, group_ids, SINR_MIN,
        )
        all_metrics[name]['time'] = timings[name]

    # =========================================================================
    #  Build summary, report, save, plot
    # =========================================================================
    # Pick random sample layouts for the snapshot plot
    sample_idx = np.random.choice(TEST_LAYOUTS, N_SAMPLE_LAYOUTS, replace=False)
    sample_layouts = layouts_test[sample_idx]

    summary = build_results_summary(
        all_metrics, allocs_dict, sample_idx, sample_layouts,
        group_ids, yellow_mask, green_mask, N_BLUE,
    )

    # Save compact summary (overwrite previous)
    with open(os.path.join(SCRIPT_DIR, 'test_jsac_results.pkl'), 'wb') as f:
        pickle.dump(summary, f)

    # Free the bulky in-memory arrays
    del all_metrics, allocs_dict

    print_metrics_summary(summary, SINR_MIN_DB)
    print(f"  Inference time:  EqPow {t_eq:.3f}s  |  WMMSE {t_wm:.3f}s  |  GNN {t_gnn:.3f}s")

    print("\n=== Plotting ===")
    plot_dashboard(summary, SINR_MIN_DB, para,
                   save_dir=SCRIPT_DIR, show=SHOW_PLOTS)
    plot_deep_dive(summary, para, group_ids, yellow_mask, green_mask,
                   SINR_MIN, save_dir=SCRIPT_DIR, show=SHOW_PLOTS)
    plot_layout_snapshots(summary, para, group_ids, yellow_mask, green_mask,
                          SINR_MIN, save_dir=SCRIPT_DIR, show=SHOW_PLOTS,
                          node_size=SNAPSHOT_NODE_SIZE)
    print("Done.")


if __name__ == "__main__":
    main()
