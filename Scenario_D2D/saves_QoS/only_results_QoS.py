import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import main_supporter as Sapo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# SAVE_DIR = os.path.join(SCRIPT_DIR, 'saves')
SAVE_DIR = SCRIPT_DIR


def load_and_plot_only(k_size=50):
    results_path = os.path.join(SAVE_DIR, f'qos_results_K{k_size}.pkl')

    print(f"Loading results from {results_path}...")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    # --- Reporting ---
    Sapo.print_qos_metrics(results)
    # --- Plotting ---
    Sapo.plot_qos_cdf(results, save_dir=None)
    

if __name__ == "__main__":
    load_and_plot_only(k_size=50)