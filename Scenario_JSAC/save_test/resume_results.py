# ==============================================================================
# Filename: resume_results.py
# Description: Quick resume — load saved test_JSAC results summary, reprint
#              metrics table and replot dashboard + deep dive + snapshots.
#              No model loading or data generation; just reads pickle files.
#
# Usage:  cd Framework_ver_4.0/save_test && python resume_results.py
# ==============================================================================

import os
import sys
import pickle

# Add parent directory (Framework_ver_4.0) to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

import config_system as CS
from test_JSAC import (print_metrics_summary, plot_dashboard, plot_deep_dive,
                       plot_layout_snapshots)


def main():
    # --- Plot controls ---
    SHOW_PLOTS          = False   # True = display, False = save only
    SNAPSHOT_NODE_SIZE  = 35    # snapshots: base marker size for cars

    # Load saved artifacts
    with open(os.path.join(SCRIPT_DIR, 'test_jsac_results.pkl'), 'rb') as f:
        summary = pickle.load(f)
    with open(os.path.join(SCRIPT_DIR, 'hyperparams.pkl'), 'rb') as f:
        hp = pickle.load(f)
    with open(os.path.join(SCRIPT_DIR, 'train_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)

    # Reconstruct config for plot titles and topology-dependent masks
    para = CS.init_parameters_JSAC(
        n_blue=hp['N_BLUE'],
        n_yellow_per_blue=hp['N_YELLOW'],
        n_green_per_blue=hp['N_GREEN'],
        field_length=hp['FIELD_LENGTH'],
    )
    K = para.n_links

    green_mask  = meta['green_mask']
    yellow_mask = meta['yellow_mask']
    group_ids   = meta['group_ids']

    # Banner
    print(f"\n{'='*60}")
    print(f"  [Resume Results] JSAC Single-Scenario Evaluation")
    print(f"  Topology: B={hp['N_BLUE']}, M_y={hp['N_YELLOW']}, "
          f"M_g={hp['N_GREEN']}  ->  K={K} links")
    print(f"  SINR_min = {hp['SINR_MIN_DB']} dB, penalty = {hp['PENALTY_WEIGHT']}")
    print(f"{'='*60}")

    # Reprint metrics
    print_metrics_summary(summary, hp['SINR_MIN_DB'])

    # Timing (stored per-method in the summary)
    for name, m in summary['methods'].items():
        print(f"  {name}: {m['time']:.3f}s", end='  |  ')
    print()

    # Replot
    print("\n=== Plotting ===")
    plot_dashboard(summary, hp['SINR_MIN_DB'], para,
                   save_dir=SCRIPT_DIR, show=SHOW_PLOTS)
    plot_deep_dive(summary, para, group_ids, yellow_mask, green_mask,
                   hp['SINR_MIN'], save_dir=SCRIPT_DIR, show=SHOW_PLOTS)
    plot_layout_snapshots(summary, para, group_ids, yellow_mask, green_mask,
                          hp['SINR_MIN'], save_dir=SCRIPT_DIR,
                          show=SHOW_PLOTS, node_size=SNAPSHOT_NODE_SIZE)
    print("Done, plots saved to {}.".format(SCRIPT_DIR))


if __name__ == "__main__":
    main()
