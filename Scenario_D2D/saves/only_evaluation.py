import sys
import os

import time
import pickle
import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config_system as CS
import utils_objective as UO
import baselines as BL
import model_gnn as MG
import model_dnn as MD
import main_supporter as Sapo

# Save directory setup (same as in main.py for consistency)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# SAVE_DIR = os.path.join(SCRIPT_DIR, 'saves')
SAVE_DIR = SCRIPT_DIR

def run_evaluation_only():
    # -------------------------------------------------------------------------
    # 0. Setup Configuration
    # -------------------------------------------------------------------------
    TEST_LAYOUTS = 300 
    COMPARE_K_LIST = [5, 10, 20, 30, 40] 
    FIELD_LENGTH = 1000 
    Field_AREA = FIELD_LENGTH ** 2 
    ADAPTIVE_DENSITY = True 
    TRAIN_K_GNN = 50 
    DENSITY = None if ADAPTIVE_DENSITY else (Field_AREA / TRAIN_K_GNN) 
    
    DNN_CONFIGS = {
        5:  {"shape": [50, 100, 25]}, 
        10: {"shape": [200, 400, 100]}, 
        20: {"shape": [800, 1600, 400]}, 
        30: {"shape": [1800, 3600, 900]}, 
        40: {"shape": [3200, 6400, 1600]} 
    }
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(f"Device: {DEVICE}")

    # -------------------------------------------------------------------------
    # 1. Load Scalers
    # -------------------------------------------------------------------------
    print("Loading saved scalers...")
    with open(os.path.join(SAVE_DIR, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    dist_scaler = scalers['dist']
    loss_scaler = scalers['loss']

    # -------------------------------------------------------------------------
    # 2. Load Models
    # -------------------------------------------------------------------------
    print("Loading trained GNN...")
    gnn_model = MG.IGCNet().to(DEVICE) 
    gnn_model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'gnn_model_K{TRAIN_K_GNN}.pth'), map_location=DEVICE, weights_only=True))
    gnn_model.eval()

    print("Loading specialized DNNs...")
    trained_dnns = {}
    for Kx in COMPARE_K_LIST:
        # Reconstruct DNN architecture: input size is K*K, output is K
        n_input = Kx * Kx 
        n_output = Kx
        dnn_model_K = MD.DNN_T1(n_input, n_output, DNN_CONFIGS[Kx]['shape']).to(DEVICE)
        dnn_model_K.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'dnn_model_K{Kx}.pth'), map_location=DEVICE, weights_only=True))
        dnn_model_K.eval()
        trained_dnns[Kx] = dnn_model_K

    # -------------------------------------------------------------------------
    # 3. Multi-Scenario Evaluation (Phase 3 logic)
    # -------------------------------------------------------------------------
    print("\n=== Phase 3: Multi-Scenario Evaluation ===")
    results = { 'K': [], 'FPLinQ': [], 'WMMSE': [], 'Greedy': [], 'GNN': [], 'DNN': [], 
                'Time_FPL': [], 'Time_WMM': [], 'Time_Grd': [], 'Time_GNN': [], 'Time_DNN': [] }

    for Kx in COMPARE_K_LIST:
        print(f'\n--- Evaluating K = {Kx} ---')
        dnn_model_K = trained_dnns[Kx]
        
        # Generate Test Set
        ch_test, graph_data_K, dLoss_K, cLoss_K, cfg_K = Sapo.generate_evaluation_dataset(
            K_size=Kx, num_layouts=TEST_LAYOUTS, 
            dist_scaler=dist_scaler, loss_scaler=loss_scaler, 
            density=DENSITY
        )
        var_K = cfg_K.output_noise_power / cfg_K.tx_power
        
        # Baselines
        start = time.time()
        Y_fp = BL.FP_optimize(cfg_K, ch_test, np.ones([TEST_LAYOUTS, Kx]))
        results['Time_FPL'].append(time.time() - start)
        
        start = time.time()
        Pini = np.random.rand(TEST_LAYOUTS, Kx, 1)
        Y_wm = BL.batch_WMMSE2(Pini, np.ones([TEST_LAYOUTS, Kx]), np.sqrt(ch_test), 1, var_K)
        results['Time_WMM'].append(time.time() - start)
        
        start = time.time()
        Y_gr = BL.simple_greedy(ch_test, np.ones([TEST_LAYOUTS, Kx]), Y_fp)
        results['Time_Grd'].append(time.time() - start)
        
        # GNN
        loader_K = DataLoader(graph_data_K, batch_size=TEST_LAYOUTS, shuffle=False)
        start = time.time()
        with torch.no_grad():
            for batch in loader_K:
                batch = batch.to(DEVICE)
                out = gnn_model(batch)
            Y_gnn = out[:, 2].reshape(-1, Kx).cpu().numpy()
        results['Time_GNN'].append(time.time() - start)
        
        # DNN
        X_test_norm = loss_scaler.transform(np.sqrt(ch_test))
        X_test_dnn = X_test_norm.reshape(ch_test.shape[0], -1)
        start = time.time()
        Y_dnn = MD.predict_dnn_torch(dnn_model_K, X_test_dnn, DEVICE)
        results['Time_DNN'].append(time.time() - start)
        
        # Compute Rates
        def get_sr(Y_alloc): 
            return np.mean(np.sum(UO.compute_rates(cfg_K, Y_alloc, dLoss_K, cLoss_K), axis=1))

        results['K'].append(Kx)
        results['FPLinQ'].append(get_sr(Y_fp))
        results['WMMSE'].append(get_sr(Y_wm))
        results['Greedy'].append(get_sr(Y_gr))
        results['GNN'].append(get_sr(Y_gnn))
        results['DNN'].append(get_sr(Y_dnn))
        
        print(f"Results K={Kx} | GNN: {results['GNN'][-1]:.2f} | DNN: {results['DNN'][-1]:.2f} | WMMSE: {results['WMMSE'][-1]:.2f}")

    print("Generating plots...")
    # Calls the exact same plotting function from your Phase 4
    Sapo.plot_results(results)


if __name__ == "__main__":
    run_evaluation_only()