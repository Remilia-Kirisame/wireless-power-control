# ==============================================================================
# Filename: main.py (JSAC — Framework_ver_4.0)
# Description: Full pipeline for JSAC wireless power control.
#   Phase 1: Generate training data + fit scalers
#   Phase 2: Train GNN (IGCNet)
#   Phase 3: Evaluate across scenarios (Naive Equal-Power, WMMSE, GNN)
#   Phase 4: Plot results
#
# Methods compared:
#   - Naive Equal-Power: lower bound (Pmax / M per link)
#   - WMMSE_JSAC: near-optimal iterative baseline
#   - GNN (IGCNet): ML approximation, K-independent
# ==============================================================================

import os
import pickle
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

import config_system as CS
import utils_objective as UO
import baselines as BL
import model_gnn as MG
import main_supporter as Sapo


def run_main():
    # =========================================================================
    #  HYPERPARAMETERS — All tunable knobs are here
    # =========================================================================

    # ----- JSAC Topology (Training Configuration) -----
    TRAIN_N_BLUE   = 10     # B:   Number of Blue cars (Tx groups)
    TRAIN_N_YELLOW = 2      # M_y: Yellow (sensing) Rx per Blue car
    TRAIN_N_GREEN  = 3      # M_g: Green (communication) Rx per Blue car
    FIELD_LENGTH   = 225    # Field side length (meters); Blue cars need ≥50m apart

    # ----- Data Generation -----
    TRAIN_LAYOUTS  = 5000   # Training layouts (more → better scaler stats & GNN)
    VAL_LAYOUTS    = 500    # Validation layouts (for monitoring during training)
    TEST_LAYOUTS   = 500    # Test layouts (final evaluation per scenario)
    SHUFFLE_CHANNELS = False # Randomize channel-to-link assignment per Blue car?
    #   True  → Yellow/Green from different Blue cars can share channels
    #           → richer cross-type interference → harder problem
    #   False → fixed slot order (k mod M) → same-type channels only
    #           → validated baseline matching smoke test results

    # ----- GNN Architecture & Training -----
    GNN_EPOCHS     = 30     # Training epochs
    GNN_LR         = 2e-3   # Initial learning rate
    GNN_BATCH_SIZE = 64     # Mini-batch size
    GNN_SCHED_STEP = 10     # LR scheduler: decay every N epochs
    GNN_SCHED_GAMMA = 0.5   # LR scheduler: multiply by gamma
    EVAL_EVERY     = 5      # Print validation metrics every N epochs

    # ----- Constraint & Loss -----
    SINR_MIN_DB     = 0.0   # Minimum Yellow SINR threshold (dB)
    SINR_MIN        = 10 ** (SINR_MIN_DB / 10)  # → linear scale (1.0 at 0 dB)
    PENALTY_WEIGHT  = 2.0   # Yellow SINR violation penalty in GNN loss
    #   0   → unconstrained Green-only optimization
    #   >0  → penalize Yellow links below SINR_MIN

    # ----- WMMSE Baseline -----
    WMMSE_ALPHA_YELLOW = 0.1  # Weight for Yellow links in WMMSE objective
    #   Green always gets alpha=1. Small positive for Yellow keeps WMMSE
    #   aware of sensing links without dominating the objective.

    # ----- Evaluation: Scenario Sweep -----
    # Axis 1 — Vary B (number of Blue cars), fixed M_y and M_g
    SWEEP_B_LIST = [3, 5, 7, 10, 13]
    SWEEP_B_MY   = TRAIN_N_YELLOW   # M_y held constant
    SWEEP_B_MG   = TRAIN_N_GREEN    # M_g held constant

    # Axis 2 — Vary (M_y, M_g) per Blue car, fixed B
    SWEEP_M_LIST = [(1, 2), (2, 3), (2, 4), (3, 5), (4, 6)]  # (M_y, M_g) pairs
    SWEEP_M_B    = TRAIN_N_BLUE     # B held constant

    # ----- Save / Device -----
    ENABLE_SAVING = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(SCRIPT_DIR, 'save_main')
    if ENABLE_SAVING:
        os.makedirs(SAVE_DIR, exist_ok=True)

    # =========================================================================
    #  Banner
    # =========================================================================
    M_train = TRAIN_N_YELLOW + TRAIN_N_GREEN
    K_train = TRAIN_N_BLUE * M_train
    print(f"\n{'='*60}")
    print(f"  JSAC Wireless Power Control")
    print(f"  Device: {DEVICE}")
    print(f"  Training topology: B={TRAIN_N_BLUE}, M_y={TRAIN_N_YELLOW}, "
          f"M_g={TRAIN_N_GREEN}  →  K={K_train} links, M={M_train} ch/Blue")
    print(f"  Shuffle channels: {SHUFFLE_CHANNELS}")
    print(f"  SINR_min = {SINR_MIN_DB} dB ({SINR_MIN:.2f} linear), "
          f"penalty = {PENALTY_WEIGHT}")
    print(f"{'='*60}")

    # =========================================================================
    #  Phase 1: Generate Training Data & Fit Scalers
    # =========================================================================
    print("\n=== Phase 1: Data Generation ===")

    train_config = CS.init_parameters_JSAC(
        n_blue=TRAIN_N_BLUE,
        n_yellow_per_blue=TRAIN_N_YELLOW,
        n_green_per_blue=TRAIN_N_GREEN,
        field_length=FIELD_LENGTH,
    )
    var = train_config.output_noise_power / train_config.tx_power

    # Training data + scaler fitting (all-in-one via Sapo)
    train_data_list, dist_scaler, loss_scaler, train_meta = \
        Sapo.generate_training_dataset(train_config, TRAIN_LAYOUTS,
                                       shuffle_channels=SHUFFLE_CHANNELS)
    train_loader = DataLoader(train_data_list, batch_size=GNN_BATCH_SIZE, shuffle=True)

    # Validation set (uses pre-fitted scalers)
    print("Generating validation set...")
    _, val_graph_data, val_dLoss, val_cLoss, _, _ = Sapo.generate_evaluation_dataset(
        train_config, VAL_LAYOUTS, dist_scaler, loss_scaler,
        train_meta['interf_mask'], train_meta['group_ids'],
        train_meta['green_mask'], train_meta['yellow_mask'],
        shuffle_channels=SHUFFLE_CHANNELS,
    )
    val_loader = DataLoader(val_graph_data, batch_size=VAL_LAYOUTS, shuffle=False)

    if ENABLE_SAVING:
        with open(os.path.join(SAVE_DIR, 'scalers.pkl'), 'wb') as f:
            pickle.dump({'dist': dist_scaler, 'loss': loss_scaler}, f)
        with open(os.path.join(SAVE_DIR, 'train_meta.pkl'), 'wb') as f:
            pickle.dump(train_meta, f)
        # Save all hyperparameters needed by resume scripts
        hyperparams = {
            'TRAIN_N_BLUE': TRAIN_N_BLUE, 'TRAIN_N_YELLOW': TRAIN_N_YELLOW,
            'TRAIN_N_GREEN': TRAIN_N_GREEN, 'FIELD_LENGTH': FIELD_LENGTH,
            'TEST_LAYOUTS': TEST_LAYOUTS, 'SHUFFLE_CHANNELS': SHUFFLE_CHANNELS,
            'SINR_MIN_DB': SINR_MIN_DB, 'SINR_MIN': SINR_MIN,
            'PENALTY_WEIGHT': PENALTY_WEIGHT, 'WMMSE_ALPHA_YELLOW': WMMSE_ALPHA_YELLOW,
            'SWEEP_B_LIST': SWEEP_B_LIST, 'SWEEP_B_MY': SWEEP_B_MY,
            'SWEEP_B_MG': SWEEP_B_MG,
            'SWEEP_M_LIST': SWEEP_M_LIST, 'SWEEP_M_B': SWEEP_M_B,
        }
        with open(os.path.join(SAVE_DIR, 'hyperparams.pkl'), 'wb') as f:
            pickle.dump(hyperparams, f)

    # =========================================================================
    #  Phase 2: Train GNN
    # =========================================================================
    print("\n=== Phase 2: GNN Training ===")
    print(f"  {GNN_EPOCHS} epochs, lr={GNN_LR}, batch={GNN_BATCH_SIZE}")

    gnn_model = MG.IGCNet().to(DEVICE)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=GNN_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, GNN_SCHED_STEP, GNN_SCHED_GAMMA)

    best_green_sr = -1
    for epoch in range(1, GNN_EPOCHS + 1):
        train_loss = MG.train_epoch(
            gnn_model, train_loader, optimizer, DEVICE,
            K_train, var, TRAIN_N_BLUE, SINR_MIN, PENALTY_WEIGHT,
        )
        if epoch % EVAL_EVERY == 0 or epoch == GNN_EPOCHS:
            _, green_sr, viol_rate = MG.eval_epoch(
                gnn_model, val_loader, DEVICE, K_train, train_config,
                val_dLoss, val_cLoss,
                train_meta['green_mask'], train_meta['yellow_mask'],
                TRAIN_N_BLUE, SINR_MIN, PENALTY_WEIGHT,
            )
            marker = ""
            if green_sr > best_green_sr:
                best_green_sr = green_sr
                marker = " *best"
            print(f"  Epoch {epoch:03d} | TrLoss {train_loss:.4f} | "
                  f"GreenSR {green_sr:.2f} | YellowViol {viol_rate:.1f}%{marker}")
        scheduler.step()

    print(f"  Best Green sum-rate on validation: {best_green_sr:.2f}")

    if ENABLE_SAVING:
        torch.save(gnn_model.state_dict(), os.path.join(SAVE_DIR, 'gnn_model.pth'))

    # =========================================================================
    #  Phase 3: Multi-Scenario Evaluation
    # =========================================================================
    print("\n=== Phase 3: Evaluation ===")

    def evaluate_scenario(config):
        """
        Run Naive / WMMSE / GNN on a given JSAC config.
        Returns a dict with metrics and timing for each method.
        """
        K = config.n_links
        B = config.n_blue
        M = config.n_links_per_blue

        # --- Generate evaluation data with its own topology ---
        _, dists, group_ids, link_types = CS.generate_layouts_jsac(config, TEST_LAYOUTS)
        _, interf_mask, green_mask, yellow_mask = CS.jsac_metadata(
            group_ids, link_types, M, shuffle_channels=SHUFFLE_CHANNELS)
        _, channel_losses = CS.compute_channel_losses_jsac(config, dists, interf_mask)

        # --- Normalize using pre-fitted scalers (with this topology's masks) ---
        norm_dists  = dist_scaler.transform(1.0 / dists, interf_mask, yellow_mask, green_mask)
        norm_losses = loss_scaler.transform(
            np.sqrt(channel_losses), interf_mask, yellow_mask, green_mask)
        graph_data = MG.proc_data(
            channel_losses, dists, norm_dists, norm_losses, K,
            interf_mask, group_ids, green_mask, yellow_mask,
        )

        # --- Evaluation arrays ---
        dLoss = UO.get_directLink_channel_losses(channel_losses)
        cLoss = UO.get_crossLink_channel_losses(channel_losses)
        var_eval = config.output_noise_power / config.tx_power

        # --- 1. Naive Equal Power ---
        t0 = time.time()
        Y_naive = np.tile(BL.naive_equal_power(config, group_ids), (TEST_LAYOUTS, 1))
        t_naive = time.time() - t0

        # --- 2. WMMSE ---
        t0 = time.time()
        alpha = np.ones((TEST_LAYOUTS, K))
        alpha[:, yellow_mask] = WMMSE_ALPHA_YELLOW
        H = np.sqrt(channel_losses)
        Y_wmmse = BL.batch_WMMSE2_JSAC(
            TEST_LAYOUTS, alpha, H, 1.0, var_eval,
            group_ids, SINR_MIN, yellow_mask,
        )
        t_wmmse = time.time() - t0

        # --- 3. GNN ---
        loader = DataLoader(graph_data, batch_size=TEST_LAYOUTS, shuffle=False)
        t0 = time.time()
        gnn_model.eval()
        Y_gnn = None
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                out = gnn_model(batch)
                raw_p = out[:, 3]
                p = MG.apply_group_softmax(raw_p, batch.group_ids, batch.batch, B)
                Y_gnn = p.reshape(-1, K).cpu().numpy()
        t_gnn = time.time() - t0

        # --- Compute metrics ---
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

    def print_scenario(label, scenario):
        _print_scenario(label, scenario)

    # ---- Axis 1: Vary B (number of Blue cars) ----
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
        print_scenario(f"B={B_val} (K={cfg.n_links})", scenario)

        results_B['B'].append(B_val)
        results_B['K'].append(cfg.n_links)
        for method in ['Naive', 'WMMSE', 'GNN']:
            results_B[f'{method}_sr'].append(scenario[method]['green_sr'])
            results_B[f'{method}_viol'].append(scenario[method]['viol_rate'])
            results_B[f'{method}_time'].append(scenario[method]['time'])

    # ---- Axis 2: Vary M (links per Blue car) ----
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
        print_scenario(f"M_y={my_val}, M_g={mg_val} (K={cfg.n_links})", scenario)

        results_M['M_y'].append(my_val)
        results_M['M_g'].append(mg_val)
        results_M['M'].append(my_val + mg_val)
        results_M['K'].append(cfg.n_links)
        for method in ['Naive', 'WMMSE', 'GNN']:
            results_M[f'{method}_sr'].append(scenario[method]['green_sr'])
            results_M[f'{method}_viol'].append(scenario[method]['viol_rate'])
            results_M[f'{method}_time'].append(scenario[method]['time'])

    # Save results
    if ENABLE_SAVING:
        with open(os.path.join(SAVE_DIR, 'results_sweep_B.pkl'), 'wb') as f:
            pickle.dump(results_B, f)
        with open(os.path.join(SAVE_DIR, 'results_sweep_M.pkl'), 'wb') as f:
            pickle.dump(results_M, f)

    # =========================================================================
    #  Phase 4: Plotting
    # =========================================================================
    print("\n=== Phase 4: Plotting ===")
    plot_sweep_results(results_B, results_M, save_dir=SAVE_DIR if ENABLE_SAVING else '.')
    print("Done.")


# =============================================================================
#  Shared Helpers (importable by resume scripts)
# =============================================================================

def _print_scenario(label, scenario):
    """Pretty-print one scenario's results."""
    print(f"\n  {label}")
    print(f"  {'Method':<12} {'Green SR':>10} {'Viol %':>8} {'Time (s)':>10}")
    print(f"  {'-'*42}")
    for name in ['Naive', 'WMMSE', 'GNN']:
        s = scenario[name]
        print(f"  {name:<12} {s['green_sr']:>10.2f} {s['viol_rate']:>7.1f}% "
              f"{s['time']:>10.3f}")
    ratio = scenario['GNN']['green_sr'] / (scenario['WMMSE']['green_sr'] + 1e-12)
    print(f"  GNN/WMMSE ratio: {ratio*100:.1f}%")


def print_sweep_tables(results_B, results_M):
    """Reprint all sweep results from saved dicts."""
    methods = ['Naive', 'WMMSE', 'GNN']

    print("\n--- Sweep: Vary B (Blue cars) ---")
    for i, B_val in enumerate(results_B['B']):
        K = results_B['K'][i]
        scenario = {}
        for m in methods:
            scenario[m] = {
                'green_sr': results_B[f'{m}_sr'][i],
                'viol_rate': results_B[f'{m}_viol'][i],
                'time': results_B[f'{m}_time'][i],
            }
        _print_scenario(f"B={B_val} (K={K})", scenario)

    print(f"\n--- Sweep: Vary (M_y, M_g) per Blue car ---")
    for i in range(len(results_M['M'])):
        my, mg = results_M['M_y'][i], results_M['M_g'][i]
        K = results_M['K'][i]
        scenario = {}
        for m in methods:
            scenario[m] = {
                'green_sr': results_M[f'{m}_sr'][i],
                'viol_rate': results_M[f'{m}_viol'][i],
                'time': results_M[f'{m}_time'][i],
            }
        _print_scenario(f"M_y={my}, M_g={mg} (K={K})", scenario)


# =============================================================================
#  Plotting
# =============================================================================

COLORS = {'Naive': '#888888', 'WMMSE': '#2196F3', 'GNN': '#FF5722'}
MARKERS = {'Naive': 's', 'WMMSE': '^', 'GNN': 'o'}

def plot_sweep_results(results_B, results_M, save_dir='.'):
    """
    Generate comparison plots for both sweep axes.
    Row 1: Vary B — Green sum-rate and Yellow SINR violation
    Row 2: Vary M — Green sum-rate and Yellow SINR violation
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # ---- Row 1: Sweep B ----
    x_B = results_B['B']
    _plot_metric(axes[0, 0], x_B, results_B, '_sr',
                 xlabel='Number of Blue Cars (B)',
                 ylabel='Green Sum-Rate (bits/s/Hz)',
                 title='Green Sum-Rate vs. B')
    _plot_metric(axes[0, 1], x_B, results_B, '_viol',
                 xlabel='Number of Blue Cars (B)',
                 ylabel='Yellow SINR Violation (%)',
                 title='Yellow Violation Rate vs. B')

    # ---- Row 2: Sweep M ----
    x_M = results_M['M']
    x_labels_M = [f"({my},{mg})" for my, mg in zip(results_M['M_y'], results_M['M_g'])]
    _plot_metric(axes[1, 0], x_M, results_M, '_sr',
                 xlabel='Links per Blue Car (M)',
                 ylabel='Green Sum-Rate (bits/s/Hz)',
                 title='Green Sum-Rate vs. M',
                 x_labels=x_labels_M)
    _plot_metric(axes[1, 1], x_M, results_M, '_viol',
                 xlabel='Links per Blue Car (M)',
                 ylabel='Yellow SINR Violation (%)',
                 title='Yellow Violation Rate vs. M',
                 x_labels=x_labels_M)

    fig.suptitle('JSAC Power Control: Multi-Scenario Comparison', fontsize=14, y=1.01)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'jsac_results.png'), dpi=130, bbox_inches='tight')
    plt.show()


def _plot_metric(ax, x_vals, results, suffix, xlabel, ylabel, title, x_labels=None):
    """Helper: plot one metric (sum-rate or violation) for all methods."""
    for method in ['Naive', 'WMMSE', 'GNN']:
        ax.plot(x_vals, results[f'{method}{suffix}'],
                marker=MARKERS[method], color=COLORS[method],
                label=method, linewidth=2, markersize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if x_labels is not None:
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_labels)


# =============================================================================
#  Entry Point
# =============================================================================

if __name__ == "__main__":
    run_main()
