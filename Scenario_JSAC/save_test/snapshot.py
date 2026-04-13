# ==============================================================================
# Filename: snapshot.py
# Description: Quick snapshot generator. Loads the trained GNN model + scalers,
#              generates a fresh small batch of evaluation layouts, runs all
#              three methods, and displays a fresh set of layout snapshots.
#              Use this to grab a different/nicer snapshot without re-running
#              the full evaluation pipeline.
#
# Usage:  cd Framework_ver_4.0/save_test && python snapshot.py
# ==============================================================================

import os
import sys
import pickle
import tempfile

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Add parent directory (Framework_ver_4.0) to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

import config_system as CS
import baselines as BL
import model_gnn as MG
import main_supporter as Sapo
from test_JSAC import (calculate_metrics_jsac, build_results_summary,
                       plot_layout_snapshots)


def main():
    # --- Snapshot controls (tune here) ---
    N_SAMPLE_LAYOUTS    = 3      # how many random layouts to draw
    SNAPSHOT_NODE_SIZE  = 35    # base marker size for cars
    EVAL_LAYOUTS        = 50     # how many fresh layouts to generate (sample drawn from these)
    SHOW_PLOTS          = True   # display the figure
    SAVE_PLOT           = True  # also save PNG to SCRIPT_DIR (overwrites jsac_layout_snapshots.png)

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

    # Unpack hyperparameters + masks
    N_BLUE       = hp['N_BLUE']
    N_YELLOW     = hp['N_YELLOW']
    N_GREEN      = hp['N_GREEN']
    FIELD_LENGTH = hp['FIELD_LENGTH']
    SINR_MIN     = hp['SINR_MIN']
    WMMSE_ALPHA_YELLOW = hp['WMMSE_ALPHA_YELLOW']

    para = CS.init_parameters_JSAC(
        n_blue=N_BLUE,
        n_yellow_per_blue=N_YELLOW,
        n_green_per_blue=N_GREEN,
        field_length=FIELD_LENGTH,
    )
    K = para.n_links
    var = para.output_noise_power / para.tx_power

    green_mask  = meta['green_mask']
    yellow_mask = meta['yellow_mask']
    group_ids   = meta['group_ids']
    interf_mask = meta['interf_mask']

    print(f"\n[snapshot] B={N_BLUE} M_y={N_YELLOW} M_g={N_GREEN} K={K}  "
          f"|  drawing {N_SAMPLE_LAYOUTS} of {EVAL_LAYOUTS} fresh layouts")

    # =========================================================================
    #  Generate a fresh evaluation batch (one call gives ch + graphs + layouts)
    # =========================================================================
    ch, graph_data, dLoss, cLoss, layouts, _ = Sapo.generate_evaluation_dataset(
        para, EVAL_LAYOUTS, dist_scaler, loss_scaler,
        interf_mask, group_ids, green_mask, yellow_mask,
    )

    # =========================================================================
    #  Run all three methods on this batch
    # =========================================================================
    allocs_dict = {}

    # 1. Equal Power
    allocs_dict['Equal Power'] = np.tile(
        BL.naive_equal_power(para, group_ids), (EVAL_LAYOUTS, 1))

    # 2. WMMSE
    alpha = np.ones((EVAL_LAYOUTS, K))
    alpha[:, yellow_mask] = WMMSE_ALPHA_YELLOW
    allocs_dict['WMMSE'] = BL.batch_WMMSE2_JSAC(
        EVAL_LAYOUTS, alpha, np.sqrt(ch), 1.0, var,
        group_ids, SINR_MIN, yellow_mask,
    )

    # 3. GNN
    loader = DataLoader(graph_data, batch_size=EVAL_LAYOUTS, shuffle=False)
    gnn_powers = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = gnn_model(batch)
            raw_p = out[:, 3]
            p = MG.apply_group_softmax(raw_p, batch.group_ids, batch.batch, N_BLUE)
            gnn_powers.append(p.reshape(-1, K).cpu().numpy())
    allocs_dict['GNN'] = np.concatenate(gnn_powers, axis=0)

    # =========================================================================
    #  Per-layout metrics (needed for the snapshot title annotations)
    # =========================================================================
    all_metrics = {}
    for name, allocs in allocs_dict.items():
        all_metrics[name] = calculate_metrics_jsac(
            para, allocs, dLoss, cLoss,
            green_mask, yellow_mask, group_ids, SINR_MIN,
        )

    # =========================================================================
    #  Pick fresh sample layouts and plot
    # =========================================================================
    sample_idx = np.random.choice(EVAL_LAYOUTS, N_SAMPLE_LAYOUTS, replace=False)
    sample_layouts = layouts[sample_idx]

    # Build the compact summary the plot function now consumes.  Downsample
    # sizes don't matter here — only the snapshot-indexed fields are read —
    # but the builder is cheap on a 50-layout batch so we just use defaults.
    summary = build_results_summary(
        all_metrics, allocs_dict, sample_idx, sample_layouts,
        group_ids, yellow_mask, green_mask, N_BLUE,
    )

    # plot_layout_snapshots unconditionally writes a PNG; if SAVE_PLOT is False,
    # send it to the system temp dir so the canonical save_test PNG is preserved.
    save_dir = SCRIPT_DIR if SAVE_PLOT else tempfile.gettempdir()

    plot_layout_snapshots(
        summary, para, group_ids, yellow_mask, green_mask, SINR_MIN,
        save_dir=save_dir, show=SHOW_PLOTS, node_size=SNAPSHOT_NODE_SIZE,
    )

    print(f"\n[snapshot] picked layouts: {sample_idx.tolist()}")
    if SAVE_PLOT:
        print(f"[snapshot] saved PNG → {save_dir}/jsac_layout_snapshots.png")
    print("Done.")


if __name__ == "__main__":
    main()
