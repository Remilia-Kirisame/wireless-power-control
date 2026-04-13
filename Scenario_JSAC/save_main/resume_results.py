# ==============================================================================
# Filename: resume_results.py
# Description: Quick resume — load saved sweep results, reprint tables and replot.
#              No model loading or data generation; just reads pickle files.
#
# Usage:  cd Framework_ver_4.0/save_main && python resume_results.py
# ==============================================================================

import os
import sys
import pickle

# Add parent directory (Framework_ver_4.0) to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from main import print_sweep_tables, plot_sweep_results


def main():
    # Load saved results
    with open(os.path.join(SCRIPT_DIR, 'results_sweep_B.pkl'), 'rb') as f:
        results_B = pickle.load(f)
    with open(os.path.join(SCRIPT_DIR, 'results_sweep_M.pkl'), 'rb') as f:
        results_M = pickle.load(f)
    with open(os.path.join(SCRIPT_DIR, 'hyperparams.pkl'), 'rb') as f:
        hp = pickle.load(f)

    # Banner
    M_train = hp['TRAIN_N_YELLOW'] + hp['TRAIN_N_GREEN']
    K_train = hp['TRAIN_N_BLUE'] * M_train
    print(f"\n{'='*60}")
    print(f"  [Resume Results] JSAC Wireless Power Control")
    print(f"  Training topology: B={hp['TRAIN_N_BLUE']}, M_y={hp['TRAIN_N_YELLOW']}, "
          f"M_g={hp['TRAIN_N_GREEN']}  ->  K={K_train} links")
    print(f"  SINR_min = {hp['SINR_MIN_DB']} dB, penalty = {hp['PENALTY_WEIGHT']}")
    print(f"{'='*60}")

    # Reprint tables
    print_sweep_tables(results_B, results_M)

    # Replot
    print("\n=== Plotting ===")
    plot_sweep_results(results_B, results_M, save_dir=SCRIPT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
