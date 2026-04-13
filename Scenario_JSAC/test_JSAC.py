# ==============================================================================
# Filename: test_JSAC.py
# Description: Single-scenario deep evaluation for JSAC wireless power control.
#   Trains a GNN on one fixed JSAC configuration and compares against
#   Naive Equal-Power and WMMSE_JSAC baselines with detailed metrics and
#   rich visualizations.
#
# Metrics reported:
#   - Green sum-rate (communication objective)
#   - Yellow SINR violation rate (sensing constraint)
#   - Per-group power utilization
#   - Jain's fairness index (over Green links)
#   - Rate distribution percentiles
#
# Plots:
#   Figure 1 — Dashboard (1×3):
#       Green Rate CDF, Yellow SINR CDF, Radar chart
#   Figure 2 — Deep Dive (2×3):
#       Per-link power bars (×3 methods), Per-group budget violin,
#       Yellow/Green power share, SINR–Rate scatter
#   Figure 3 — Layout Snapshots (n_sample×3):
#       Geographic layout per (sample, method) — Rx brightness ∝ allocated power
# ==============================================================================

import os
import pickle

import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from torch_geometric.loader import DataLoader

import config_system as CS
import utils_objective as UO
import baselines as BL
import model_gnn as MG
import main_supporter as Sapo


# =============================================================================
# Main Test  (hyperparameters live here — tune below)
# =============================================================================

def run_test():
    # --- Save Feature Toggle ---
    ENABLE_SAVING = True
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(SCRIPT_DIR, 'save_test')
    if ENABLE_SAVING:
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"Saving enabled → {SAVE_DIR}")

    # --- Configuration ---
    N_BLUE          = 10
    N_YELLOW        = 2
    N_GREEN         = 3
    FIELD_LENGTH    = 225

    TRAIN_LAYOUTS   = 5000
    TEST_LAYOUTS    = 1000

    GNN_EPOCHS      = 20
    GNN_LR          = 2e-3
    GNN_BATCH_SIZE  = 64

    SINR_MIN_DB     = 0.0
    SINR_MIN        = 10 ** (SINR_MIN_DB / 10)   # 1.0
    PENALTY_WEIGHT  = 2.0
    WMMSE_ALPHA_YELLOW = 0.1

    NOISE_DB = -169 # mili-dB noise power density

    # --- Plot controls ---
    SHOW_PLOTS          = False   # True = display, False = save only
    N_SAMPLE_LAYOUTS    = 2      # snapshots: how many random layouts to draw
    SNAPSHOT_NODE_SIZE  = 35    # snapshots: base marker size for cars

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")

    para = CS.init_parameters_JSAC(
        n_blue=N_BLUE,
        n_yellow_per_blue=N_YELLOW,
        n_green_per_blue=N_GREEN,
        field_length=FIELD_LENGTH,
        noise_density_milli_decibel= NOISE_DB,
    )
    K = para.n_links
    M = para.n_links_per_blue
    var = para.output_noise_power / para.tx_power

    print(f"\nTopology: B={N_BLUE}, M_y={N_YELLOW}, M_g={N_GREEN} → K={K}, M={M}")
    print(f"SINR_min = {SINR_MIN_DB} dB ({SINR_MIN:.2f} linear), "
          f"penalty = {PENALTY_WEIGHT}")

    # ==================================================================
    # Phase 1: Data Generation & Scalers
    # ==================================================================
    print("\n--- Phase 1: Data Generation ---")
    train_data, dist_scaler, loss_scaler, meta = \
        Sapo.generate_training_dataset(para, TRAIN_LAYOUTS)

    green_mask  = meta['green_mask']
    yellow_mask = meta['yellow_mask']
    group_ids   = meta['group_ids']
    interf_mask = meta['interf_mask']

    train_loader = DataLoader(train_data, batch_size=GNN_BATCH_SIZE, shuffle=True)

    # Test data
    ch_test, test_graph, dLoss_test, cLoss_test, layouts_test, _ = \
        Sapo.generate_evaluation_dataset(
            para, TEST_LAYOUTS, dist_scaler, loss_scaler,
            interf_mask, group_ids, green_mask, yellow_mask,
        )
    test_loader = DataLoader(test_graph, batch_size=TEST_LAYOUTS, shuffle=False)

    if ENABLE_SAVING:
        with open(os.path.join(SAVE_DIR, 'scalers.pkl'), 'wb') as f:
            pickle.dump({'dist': dist_scaler, 'loss': loss_scaler}, f)
        with open(os.path.join(SAVE_DIR, 'train_meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)
        hyperparams = {
            'N_BLUE': N_BLUE, 'N_YELLOW': N_YELLOW, 'N_GREEN': N_GREEN,
            'FIELD_LENGTH': FIELD_LENGTH, 'TEST_LAYOUTS': TEST_LAYOUTS,
            'SINR_MIN_DB': SINR_MIN_DB, 'SINR_MIN': SINR_MIN,
            'PENALTY_WEIGHT': PENALTY_WEIGHT, 'WMMSE_ALPHA_YELLOW': WMMSE_ALPHA_YELLOW,
        }
        with open(os.path.join(SAVE_DIR, 'hyperparams.pkl'), 'wb') as f:
            pickle.dump(hyperparams, f)

    # ==================================================================
    # Phase 2: Train GNN
    # ==================================================================
    print("\n--- Phase 2: Train GNN ---")
    model = MG.IGCNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_losses = []
    for epoch in range(1, GNN_EPOCHS + 1):
        loss = MG.train_epoch(
            model, train_loader, optimizer, DEVICE,
            K, var, N_BLUE, SINR_MIN, PENALTY_WEIGHT,
        )
        scheduler.step()
        train_losses.append(loss)
        if epoch % 5 == 0 or epoch == GNN_EPOCHS:
            _, green_sr, viol = MG.eval_epoch(
                model, test_loader, DEVICE, K, para,
                dLoss_test, cLoss_test,
                green_mask, yellow_mask, N_BLUE, SINR_MIN, PENALTY_WEIGHT,
            )
            print(f"  Epoch {epoch:3d} | loss {loss:.4f} | "
                  f"GreenSR {green_sr:.2f} | Viol {viol:.1f}%")

    if ENABLE_SAVING:
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'gnn_model.pth'))

    # ==================================================================
    # Phase 3: Evaluation
    # ==================================================================
    print("\n--- Phase 3: Evaluation ---")
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
        TEST_LAYOUTS, alpha, H_test, 1.0, var,
        group_ids, SINR_MIN, yellow_mask,
    )
    t_wm = time.time() - t0
    allocs_dict['WMMSE'] = allocs_wm

    # 3. GNN
    t0 = time.time()
    model.eval()
    gnn_powers = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            raw_p = out[:, 3]
            p = MG.apply_group_softmax(raw_p, batch.group_ids, batch.batch, N_BLUE)
            gnn_powers.append(p.reshape(-1, K).cpu().numpy())
    allocs_gnn = np.concatenate(gnn_powers, axis=0)
    t_gnn = time.time() - t0
    allocs_dict['GNN'] = allocs_gnn

    # Compute all metrics
    all_metrics = {}
    timings = {'Equal Power': t_eq, 'WMMSE': t_wm, 'GNN': t_gnn}
    for name, allocs in allocs_dict.items():
        all_metrics[name] = calculate_metrics_jsac(
            para, allocs, dLoss_test, cLoss_test,
            green_mask, yellow_mask, group_ids, SINR_MIN,
        )
        all_metrics[name]['time'] = timings[name]

    # ==================================================================
    # Phase 4: Build plot-ready summary & save
    # ==================================================================
    # Pick random sample layouts for the snapshot plot
    sample_idx = np.random.choice(TEST_LAYOUTS, N_SAMPLE_LAYOUTS, replace=False)
    sample_layouts = layouts_test[sample_idx]

    summary = build_results_summary(
        all_metrics, allocs_dict, sample_idx, sample_layouts,
        group_ids, yellow_mask, green_mask, N_BLUE,
    )

    if ENABLE_SAVING:
        with open(os.path.join(SAVE_DIR, 'test_jsac_results.pkl'), 'wb') as f:
            pickle.dump(summary, f)

    # Free the bulky in-memory arrays — plots consume the summary from here on
    del all_metrics, allocs_dict

    # ==================================================================
    # Phase 5: Reporting & Plots
    # ==================================================================
    print("\n--- Phase 5: Results ---")
    print_metrics_summary(summary, SINR_MIN_DB)
    print(f"  Inference time:  EqPow {t_eq:.3f}s  |  WMMSE {t_wm:.3f}s  |  GNN {t_gnn:.3f}s")

    print("\n--- Phase 6: Plotting ---")
    plot_dashboard(summary, SINR_MIN_DB, para,
                   save_dir=SAVE_DIR, show=SHOW_PLOTS)
    plot_deep_dive(summary, para, group_ids, yellow_mask, green_mask,
                   SINR_MIN, save_dir=SAVE_DIR, show=SHOW_PLOTS)
    plot_layout_snapshots(summary, para, group_ids, yellow_mask, green_mask,
                          SINR_MIN, save_dir=SAVE_DIR, show=SHOW_PLOTS,
                          node_size=SNAPSHOT_NODE_SIZE)

    print("Done, plots saved to {}.".format(SAVE_DIR))


# =============================================================================
# Metrics
# =============================================================================

def calculate_metrics_jsac(general_para, allocs, dLoss, cLoss,
                           green_mask, yellow_mask, group_ids, sinr_min):
    """
    Computes JSAC-specific metrics for a batch of power allocations.

    Args:
        general_para: Config object.
        allocs:       [Batch, K] power allocations.
        dLoss:        [Batch, K] direct-link channel losses.
        cLoss:        [Batch, K, K] cross-link channel losses.
        green_mask:   (K,) bool — True for Green links.
        yellow_mask:  (K,) bool — True for Yellow links.
        group_ids:    (K,) int — Blue car index per link.
        sinr_min:     float — minimum SINR threshold (linear).

    Returns:
        dict with:
            green_sumrate:    scalar — average Green sum-rate across batch
            yellow_viol_rate: scalar — percentage of Yellow links below sinr_min
            green_rates:      [Batch, n_green] — per-Green-link rates
            yellow_sinrs:     [Batch, n_yellow] — per-Yellow-link SINRs
            all_rates:        [Batch, K] — all link rates
            all_sinrs:        [Batch, K] — all link SINRs
            group_power_util: [Batch, B] — fraction of Pmax used per group
            jains_green:      scalar — Jain's fairness over Green links
            percentiles:      dict of Green rate percentiles (5, 10, 25, 50)
    """
    rates = UO.compute_rates(general_para, allocs, dLoss, cLoss)
    sinrs = UO.compute_SINRs(general_para, allocs, dLoss, cLoss)

    green_rates = rates[:, green_mask]
    yellow_sinrs = sinrs[:, yellow_mask]

    # Green sum-rate
    green_sumrate = np.mean(np.sum(green_rates, axis=1))

    # Yellow SINR violation
    yellow_viol_rate = np.mean(yellow_sinrs < sinr_min) * 100

    # Per-group power utilization (fraction of budget used)
    B = general_para.n_blue
    Batch = allocs.shape[0]
    group_power_util = np.zeros((Batch, B))
    for b in range(B):
        mask_b = (group_ids == b)
        group_power_util[:, b] = allocs[:, mask_b].sum(axis=1)
    # allocs are normalized (group sums to 1.0 = Pmax), so utilization = group sum

    # Jain's fairness index over Green links (per layout, then averaged)
    n_green = green_rates.shape[1]
    numerator = np.square(np.sum(green_rates, axis=1))
    denominator = n_green * np.sum(np.square(green_rates), axis=1) + 1e-12
    jains_green = np.mean(numerator / denominator)

    # Green rate percentiles
    green_flat = green_rates.flatten()
    percentiles = {
        'p05': np.percentile(green_flat, 5),
        'p10': np.percentile(green_flat, 10),
        'p25': np.percentile(green_flat, 25),
        'p50': np.percentile(green_flat, 50),
    }

    return {
        'green_sumrate': green_sumrate,
        'yellow_viol_rate': yellow_viol_rate,
        'green_rates': green_rates,
        'yellow_sinrs': yellow_sinrs,
        'all_rates': rates,
        'all_sinrs': sinrs,
        'group_power_util': group_power_util,
        'jains_green': jains_green,
        'percentiles': percentiles,
    }


# =============================================================================
# Results Summary (plot-ready, compact pickle)
# =============================================================================

def build_results_summary(all_metrics, allocs_dict, sample_idx, sample_layouts,
                          group_ids, yellow_mask, green_mask, n_blue,
                          n_cdf=500, n_scatter=500, n_violin=500, seed=None):
    """
    Compress per-layout metrics + full allocations into a compact plot-ready
    dict. Replaces the old 4.8 MB `{all_metrics, allocs_dict, ...}` dump with
    a ~40 KB summary that preserves plot fidelity.

    Downsampling strategy:
      - CDFs: `n_cdf` quantile samples of the flattened per-link arrays.
      - Scatter (per-layout means): random subsample of `n_scatter` layouts.
      - Violin / share samples: random subsample of `n_violin` flattened values.
      - Snapshot layouts: per-sample slices preserved verbatim.

    All arrays stored as float32. Pass `seed` for reproducible subsampling.
    """
    rng = np.random.default_rng(seed)
    methods = list(all_metrics.keys())
    B = n_blue
    q_cdf = np.linspace(0.0, 1.0, n_cdf)

    summary = {
        'methods': {},
        'sample_layouts': np.asarray(sample_layouts, dtype=np.float32),
        'sample_idx': np.asarray(sample_idx, dtype=np.int64),
        'cdf_quantiles': q_cdf.astype(np.float32),
    }

    for name in methods:
        m = all_metrics[name]
        allocs = allocs_dict[name]            # (N, K)
        N = allocs.shape[0]

        green_rates  = m['green_rates']       # (N, n_g)
        yellow_sinrs = m['yellow_sinrs']      # (N, n_y)
        group_util   = m['group_power_util']  # (N, B)

        # --- CDFs: quantile samples of the flattened arrays ---
        green_rate_cdf  = np.quantile(green_rates.ravel(),  q_cdf)
        yellow_sinr_cdf = np.quantile(yellow_sinrs.ravel(), q_cdf)

        # --- Scatter: per-layout means, random subsample ---
        sc_size = min(n_scatter, N)
        sc_idx = rng.choice(N, sc_size, replace=False)
        scatter_green_rate  = green_rates[sc_idx].mean(axis=1)
        scatter_yellow_sinr = yellow_sinrs[sc_idx].mean(axis=1)

        # --- Violin: flattened group-utilization, random subsample ---
        gu_flat = group_util.ravel()
        vio_size = min(n_violin, gu_flat.size)
        group_util_sample = rng.choice(gu_flat, vio_size, replace=False)

        # --- Yellow / Green power shares per (layout, Blue car) ---
        yellow_shares, green_shares = [], []
        for b in range(B):
            mask_b = (group_ids == b)
            group_total = allocs[:, mask_b].sum(axis=1) + 1e-12
            y_in = yellow_mask[mask_b]
            g_in = green_mask[mask_b]
            yellow_shares.append(allocs[:, mask_b][:, y_in].sum(axis=1) / group_total)
            green_shares.append(allocs[:, mask_b][:, g_in].sum(axis=1) / group_total)
        yellow_shares = np.concatenate(yellow_shares)
        green_shares  = np.concatenate(green_shares)
        sh_size = min(n_violin, yellow_shares.size)
        sh_idx = rng.choice(yellow_shares.size, sh_size, replace=False)

        # --- Per-link average allocation (for power bars) ---
        mean_power_per_link = allocs.mean(axis=0)

        # --- Snapshot slices (verbatim) ---
        sample_allocs = allocs[sample_idx]
        sample_green_sumrate = green_rates[sample_idx].sum(axis=1)
        sample_yellow_sinrs  = yellow_sinrs[sample_idx]

        summary['methods'][name] = {
            # scalars
            'green_sumrate':         float(m['green_sumrate']),
            'yellow_viol_rate':      float(m['yellow_viol_rate']),
            'jains_green':           float(m['jains_green']),
            'percentiles':           {k: float(v) for k, v in m['percentiles'].items()},
            'group_power_util_mean': float(group_util.mean()),
            'time':                  float(m.get('time', 0.0)),
            # plot-ready arrays (float32)
            'mean_power_per_link':   mean_power_per_link.astype(np.float32),
            'green_rate_cdf':        green_rate_cdf.astype(np.float32),
            'yellow_sinr_cdf':       yellow_sinr_cdf.astype(np.float32),
            'scatter_green_rate':    scatter_green_rate.astype(np.float32),
            'scatter_yellow_sinr':   scatter_yellow_sinr.astype(np.float32),
            'group_util_sample':     group_util_sample.astype(np.float32),
            'yellow_share_sample':   yellow_shares[sh_idx].astype(np.float32),
            'green_share_sample':    green_shares[sh_idx].astype(np.float32),
            # snapshot-indexed
            'sample_allocs':         sample_allocs.astype(np.float32),
            'sample_green_sumrate':  sample_green_sumrate.astype(np.float32),
            'sample_yellow_sinrs':   sample_yellow_sinrs.astype(np.float32),
        }

    return summary


# =============================================================================
# Plotting
# =============================================================================

# Shared style
COLORS = {'Equal Power': '#888888', 'WMMSE': '#2196F3', 'GNN': '#FF5722'}
LINESTYLES = {'Equal Power': '--', 'WMMSE': '-', 'GNN': '-'}
YELLOW_COLOR = '#FFD700'
GREEN_COLOR  = '#2ECC71'


def plot_dashboard(summary, sinr_min_db, general_para, save_dir='.', show=True):
    """
    Figure 1: 1×3 dashboard. Consumes the compact `summary` dict produced by
    `build_results_summary()`.
      (0) CDF of Green link rates     — plotted from `green_rate_cdf`
      (1) CDF of Yellow link SINRs    — plotted from `yellow_sinr_cdf`
      (2) Radar chart — scalar metrics
    """
    fig = plt.figure(figsize=(17, 5))
    ax_cdf_green  = fig.add_subplot(1, 3, 1)
    ax_cdf_yellow = fig.add_subplot(1, 3, 2)
    ax_radar      = fig.add_subplot(1, 3, 3, polar=True)

    methods = list(summary['methods'].keys())
    K = general_para.n_links
    B = general_para.n_blue
    cdf_y = summary['cdf_quantiles']   # shared y-axis for inverse-CDF plots

    # ---- (0) Green Rate CDF ----
    ax = ax_cdf_green
    for name in methods:
        m = summary['methods'][name]
        ax.plot(m['green_rate_cdf'], cdf_y, color=COLORS[name],
                linestyle=LINESTYLES[name], linewidth=2, label=name)
    ax.set_xlabel('Green Link Rate (bits/s/Hz)')
    ax.set_ylabel('CDF')
    ax.set_title('CDF of Green (Communication) Link Rates')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # ---- (1) Yellow SINR CDF (in dB) ----
    ax = ax_cdf_yellow
    for name in methods:
        m = summary['methods'][name]
        sinrs_db = 10 * np.log10(m['yellow_sinr_cdf'].astype(np.float64) + 1e-12)
        ax.plot(sinrs_db, cdf_y, color=COLORS[name], linestyle=LINESTYLES[name],
                linewidth=2, label=name)
    ax.axvline(x=sinr_min_db, color='red', linestyle=':', linewidth=1.5,
               label=f'SINR_min = {sinr_min_db} dB')
    ax.set_xlabel('Yellow Link SINR (dB)')
    ax.set_ylabel('CDF')
    ax.set_title('CDF of Yellow (Sensing) Link SINRs')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # ---- (2) Radar Chart — Multi-Metric Comparison ----
    radar_labels = ['Green SR\n(norm)', 'Jain\'s\nFairness', '5th %ile\nRate (norm)',
                    'SINR Satisfaction\n(%)', 'Power\nEfficiency']

    sr_vals = {n: summary['methods'][n]['green_sumrate'] for n in methods}
    sr_max = max(sr_vals.values())

    p5_vals = {n: summary['methods'][n]['percentiles']['p05'] for n in methods}
    p5_max = max(p5_vals.values()) if max(p5_vals.values()) > 0 else 1.0

    radar_data = {}
    for name in methods:
        m = summary['methods'][name]
        radar_data[name] = [
            sr_vals[name] / sr_max,
            m['jains_green'],
            p5_vals[name] / p5_max,
            (100 - m['yellow_viol_rate']) / 100,
            m['group_power_util_mean'],
        ]

    n_metrics = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    for name in methods:
        values = radar_data[name] + radar_data[name][:1]
        ax_radar.plot(angles, values, color=COLORS[name], linewidth=2, label=name)
        ax_radar.fill(angles, values, color=COLORS[name], alpha=0.1)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_labels, fontsize=8)
    ax_radar.set_ylim(0, 1.1)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=7, color='grey')
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax_radar.set_title('Multi-Metric Comparison', pad=20, fontsize=11)

    fig.suptitle(f'JSAC Deep Evaluation — B={B}, M_y={general_para.n_yellow_per_blue}, '
                 f'M_g={general_para.n_green_per_blue}, K={K}',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'jsac_dashboard.png'), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_deep_dive(summary, general_para, group_ids, yellow_mask, green_mask,
                   sinr_min, save_dir='.', show=True):
    """
    Figure 2: 2×3 deep dive. Consumes the compact `summary` dict.
      Top row:    Per-link avg power (from `mean_power_per_link`)
      Bottom row: Per-group budget violin (`group_util_sample`),
                  Yellow/Green share boxes (`*_share_sample`),
                  Trade-off scatter (`scatter_*` subsampled per-layout means).
    """
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    methods = list(summary['methods'].keys())
    B = general_para.n_blue
    K = general_para.n_links
    M = general_para.n_links_per_blue

    # ---- Top row: Per-link power allocation (one subplot per method) ----
    for col, name in enumerate(methods):
        ax = axes[0, col]
        avg_power = summary['methods'][name]['mean_power_per_link']
        x = np.arange(K)
        colors = [YELLOW_COLOR if yellow_mask[i] else GREEN_COLOR for i in range(K)]
        ax.bar(x, avg_power, color=colors, width=0.8, edgecolor='none')
        ax.set_title(name)
        ax.set_xlabel('Link index')
        if col == 0:
            ax.set_ylabel('Avg power allocation')
        ax.set_xlim(-0.5, K - 0.5)
        # Group separators
        for b in range(1, B):
            ax.axvline(x=b * M - 0.5, color='gray', linestyle=':', alpha=0.5)

    # Shared y-limit across top row
    y_max = max(axes[0, c].get_ylim()[1] for c in range(3))
    for c in range(3):
        axes[0, c].set_ylim(0, y_max)

    # Legend on last subplot
    axes[0, 2].legend(handles=[
        Patch(facecolor=YELLOW_COLOR, label='Yellow (sensing)'),
        Patch(facecolor=GREEN_COLOR, label='Green (comm)')
    ], loc='upper right', fontsize=8)

    # ---- (1,0) Per-group power utilization violin ----
    ax = axes[1, 0]
    positions, violin_data, tick_positions, tick_labels = [], [], [], []
    for i, name in enumerate(methods):
        pos = i * 1.5
        positions.append(pos)
        violin_data.append(summary['methods'][name]['group_util_sample'])
        tick_positions.append(pos)
        tick_labels.append(name)

    vp = ax.violinplot(violin_data, positions=positions, showmeans=True,
                       showmedians=True, widths=0.9)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(COLORS[methods[i]])
        body.set_alpha(0.6)
    vp['cmeans'].set_color('black')
    vp['cmedians'].set_color('darkred')

    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Pmax budget')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('Group Power Sum (normalized)')
    ax.set_title('Per-Group Power Budget Utilization')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # ---- (1,1) Yellow vs Green power share box plot ----
    ax = axes[1, 1]
    box_positions, box_data, box_colors, tick_pos, tick_lab = [], [], [], [], []
    for i, name in enumerate(methods):
        m = summary['methods'][name]
        base = i * 3
        box_positions.extend([base + 0.5, base + 1.5])
        box_data.extend([m['yellow_share_sample'], m['green_share_sample']])
        box_colors.extend([YELLOW_COLOR, GREEN_COLOR])
        tick_pos.append(base + 1.0)
        tick_lab.append(name)

    bp = ax.boxplot(box_data, positions=box_positions, patch_artist=True,
                    widths=0.6, showfliers=False)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    n_y = general_para.n_yellow_per_blue
    n_g = general_para.n_green_per_blue
    ax.axhline(y=n_y / M, color=YELLOW_COLOR, linestyle='--', alpha=0.4,
               label=f'Equal Yellow ({n_y}/{M})')
    ax.axhline(y=n_g / M, color=GREEN_COLOR, linestyle='--', alpha=0.4,
               label=f'Equal Green ({n_g}/{M})')

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_ylabel('Share of Group Power')
    ax.set_title('Yellow vs Green Power Share')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')

    # ---- (1,2) Green Rate vs Yellow SINR scatter (per-layout) ----
    ax = axes[1, 2]
    for name in methods:
        m = summary['methods'][name]
        sinr_db = 10 * np.log10(m['scatter_yellow_sinr'].astype(np.float64) + 1e-12)
        ax.scatter(sinr_db, m['scatter_green_rate'],
                   color=COLORS[name], alpha=0.3, s=15, label=name)

    ax.axvline(x=10 * np.log10(sinr_min), color='red', linestyle=':',
               linewidth=1.5, label=f'SINR_min')
    ax.set_xlabel('Avg Yellow SINR per Layout (dB)')
    ax.set_ylabel('Avg Green Rate per Layout (bits/s/Hz)')
    ax.set_title('Communication-Sensing Trade-off')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    fig.suptitle('JSAC Deep Dive — Power & Trade-off Analysis', fontsize=13, y=1.01)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'jsac_deep_dive.png'), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_layout_snapshots(summary, general_para, group_ids, yellow_mask, green_mask,
                          sinr_min, save_dir='.', show=True, node_size=100):
    """
    Figure 3: n_sample × 3 grid of geographic layout snapshots. Consumes the
    compact `summary` dict — each method only needs `sample_allocs`,
    `sample_green_sumrate`, `sample_yellow_sinrs` (all pre-sliced to the chosen
    sample layouts).

    Each column shows one method (Equal Power / WMMSE / GNN); each row shows
    one randomly chosen evaluation layout. Blue cars (Tx) are blue squares.
    Yellow/Green Rx markers use brightness (alpha) proportional to allocated
    power within that layout. Per-layout Green sum-rate and Yellow violation
    count are annotated in each subplot title.
    """
    methods = list(summary['methods'].keys())
    sample_layouts = summary['sample_layouts']
    sample_idx = summary['sample_idx']
    n_sample = sample_layouts.shape[0]
    n_methods = len(methods)
    K = general_para.n_links
    B = general_para.n_blue
    field = general_para.field_length

    fig, axes = plt.subplots(n_sample, n_methods,
                             figsize=(5 * n_methods, 4.8 * n_sample),
                             squeeze=False)

    # Marker sizes derived from node_size base (Tx larger, Rx smaller)
    blue_size = node_size * 1.1
    rx_size   = node_size * 0.9
    legend_ms = max(8, np.sqrt(node_size))   # legend marker scales mildly

    # Pre-compute first-link index per Blue car for Tx positions
    first_in_group = [np.where(group_ids == b)[0][0] for b in range(B)]

    for row in range(n_sample):
        layout = sample_layouts[row]   # (K, 4)
        idx = int(sample_idx[row])
        blue_xs = layout[first_in_group, 0]
        blue_ys = layout[first_in_group, 1]
        rx_xs   = layout[:, 2]
        rx_ys   = layout[:, 3]

        for col, name in enumerate(methods):
            ax = axes[row, col]
            m = summary['methods'][name]
            powers = m['sample_allocs'][row]   # (K,)
            p_max = float(powers.max()) + 1e-12

            # Thin Tx→Rx association lines
            for k in range(K):
                ax.plot([layout[k, 0], layout[k, 2]],
                        [layout[k, 1], layout[k, 3]],
                        color='gray', alpha=0.18, linewidth=0.6, zorder=1)

            # Blue cars (Tx) — fixed appearance
            ax.scatter(blue_xs, blue_ys, s=blue_size, c='#1565C0', marker='s',
                       edgecolors='black', linewidths=0.8, zorder=4)

            # Yellow Rx — brightness ∝ power
            yel_idx = np.where(yellow_mask)[0]
            yel_alpha = np.clip(powers[yel_idx] / p_max, 0.12, 1.0)
            ax.scatter(rx_xs[yel_idx], rx_ys[yel_idx],
                       s=rx_size, c=YELLOW_COLOR, alpha=yel_alpha,
                       edgecolors='black', linewidths=0.5, zorder=3)

            # Green Rx — brightness ∝ power
            grn_idx = np.where(green_mask)[0]
            grn_alpha = np.clip(powers[grn_idx] / p_max, 0.12, 1.0)
            ax.scatter(rx_xs[grn_idx], rx_ys[grn_idx],
                       s=rx_size, c=GREEN_COLOR, alpha=grn_alpha,
                       edgecolors='black', linewidths=0.5, zorder=3)

            # Per-layout metrics (already pre-sliced for this sample)
            green_sr = float(m['sample_green_sumrate'][row])
            yellow_sinrs = m['sample_yellow_sinrs'][row]
            n_y = yellow_sinrs.shape[0]
            n_viol = int((yellow_sinrs < sinr_min).sum())
            viol_pct = 100.0 * n_viol / n_y

            ax.set_title(
                f"{name}  —  Layout #{idx}\n"
                f"Green SR = {green_sr:.2f} bits/s/Hz  |  "
                f"Yellow Viol = {n_viol}/{n_y} ({viol_pct:.0f}%)",
                fontsize=10)
            ax.set_xlim(-5, field + 5)
            ax.set_ylim(-5, field + 5)
            ax.set_aspect('equal')
            ax.set_xlabel('x (m)')
            if col == 0:
                ax.set_ylabel('y (m)')
            ax.grid(True, alpha=0.2)

    # Single shared legend at the bottom (opaque proxies, since alpha varies)
    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#1565C0',
                   markeredgecolor='black', markersize=legend_ms, label='Blue (Tx)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=YELLOW_COLOR,
                   markeredgecolor='black', markersize=legend_ms, label='Yellow Rx (sense)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GREEN_COLOR,
                   markeredgecolor='black', markersize=legend_ms, label='Green Rx (comm)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('JSAC Layout Snapshots — Rx marker brightness ∝ allocated power',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'jsac_layout_snapshots.png'),
                dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_metrics_summary(summary, sinr_min_db):
    """Pretty-print the metrics table from the compact `summary` dict."""
    methods = list(summary['methods'].keys())

    print(f"\n{'='*80}")
    print(f"  JSAC Single-Scenario Evaluation Results  (SINR_min = {sinr_min_db} dB)")
    print(f"{'='*80}")

    # Header
    header = f"{'Metric':<30}"
    for name in methods:
        header += f" {name:>14}"
    print(header)
    print('-' * 80)

    # Rows — each entry is (label, accessor, fmt).  Accessor is either a str
    # key into methods[name], or a (group, sub) tuple for nested dicts.
    rows = [
        ('Green Sum-Rate (bits/s/Hz)',  'green_sumrate',          '.2f'),
        ('Yellow SINR Violation (%)',   'yellow_viol_rate',       '.1f'),
        ("Jain's Fairness (Green)",     'jains_green',            '.4f'),
        ('Avg Group Power Utilization', 'group_power_util_mean',  '.4f'),
        ('Green Rate 5th %ile',         ('percentiles', 'p05'),   '.3f'),
        ('Green Rate 10th %ile',        ('percentiles', 'p10'),   '.3f'),
        ('Green Rate 25th %ile',        ('percentiles', 'p25'),   '.3f'),
        ('Green Rate Median',           ('percentiles', 'p50'),   '.3f'),
    ]

    for label, key, fmt in rows:
        line = f"  {label:<28}"
        for name in methods:
            m = summary['methods'][name]
            val = m[key[0]][key[1]] if isinstance(key, tuple) else m[key]
            line += f" {val:>14{fmt}}"
        print(line)

    print('-' * 80)

    # GNN/WMMSE ratio
    if 'GNN' in summary['methods'] and 'WMMSE' in summary['methods']:
        ratio = (summary['methods']['GNN']['green_sumrate'] /
                 (summary['methods']['WMMSE']['green_sumrate'] + 1e-12))
        print(f"  GNN / WMMSE Green SR ratio: {ratio*100:.1f}%")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_test()
