import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pickle
import main_supporter as Sapo

# 1. Get the absolute path of the directory containing THIS script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. Join it with the target subfolder
# SAVE_DIR = os.path.join(SCRIPT_DIR, 'saves')
SAVE_DIR = SCRIPT_DIR

def run_plot_only():
    print("Loading saved simulation results...")
    with open(os.path.join(SAVE_DIR, 'simulation_results.pkl'), 'rb') as f:
        results = pickle.load(f)
        
    print("Generating plots...")
    # Calls the exact same plotting function from your Phase 4
    Sapo.plot_results(results)

if __name__ == "__main__":
    run_plot_only()