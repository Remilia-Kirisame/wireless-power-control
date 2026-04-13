# Version 2.1 - Save and Load features added for GNN and DNN models, as well as scalers and final results.
# Version 2.2 - Added WMMSE/GNN with QoS constraints as an additional baseline in test_QoS.py.
# Version 2.3 - Layout Generation rewrite. (Sequential Rejection Sampling, config_system) (For better QoS violation)

# Note: (Ver 2.2, 2.3) For QoS modifications, check test_QoS.py. main.py is without QoS.

# ==============================================================================
# Filename: main.py
# Description: Main execution file.
#              Imports helpers from main_supporter (as Sapo).
# ==============================================================================

import os
import pickle

import numpy as np
import torch
import time
from torch_geometric.loader import DataLoader

# Import Core Modules
import config_system as CS
import utils_objective as UO
import baselines as BL
import model_gnn as MG
import model_dnn as MD

# Import Supporter
import main_supporter as Sapo

def run_main():
    # -------------------------------------------------------------------------
    # Phase 0: Configuration Settings
    # -------------------------------------------------------------------------
    # --- Save Feature Toggle ---
    ENABLE_SAVING = True  # Set to False to disable saving models and results

    if ENABLE_SAVING:
        # Save Directory for Models & Results:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        SAVE_DIR = os.path.join(SCRIPT_DIR, 'saves')
        os.makedirs(SAVE_DIR, exist_ok=True) # Creates the folder if it doesn't exist
        # exist_ok: A boolean value, default is False. If set to True, a FileExistsError will not be raised if the target directory already exists. The function will silently succeed.
        print(f"Save feature ENABLED. Files will be saved to: {SAVE_DIR}")
    else:
        print("Save feature DISABLED.")

    # --------------------------------------------------------------------------

    # Training Settings
    TRAIN_K_GNN = 50
    TRAIN_LAYOUTS = 1000
    TEST_LAYOUTS = 300
    COMPARE_K_LIST = [5, 10, 20, 30, 40] 

    # Field & Density Settings
    FIELD_LENGTH = 1000  # in meters
    Field_AREA = FIELD_LENGTH ** 2
    ADAPTIVE_DENSITY = True  # Whether to adapt density per K 
        # If True, density changes with K, i.e. field length doesn't change
        # If False, field length changes with K to keep density constant
        # We pass a None value of density to `generate_evaluation_dataset` and `train_specialized_dnn`, if we want to keep the field length constant.
    DENSITY = None if ADAPTIVE_DENSITY else (Field_AREA / TRAIN_K_GNN)

    # GNN Hyperparameters
    lr_GNN = 3e-3
    epochs_GNN = 20

    # DNN Hyperparameters per K
    DNN_CONFIGS = {
        5:  {"shape": [50, 100, 25],   "batch_size": 30, "epochs": 20},
        10: {"shape": [200, 400, 100],   "batch_size": 30, "epochs": 20},
        20: {"shape": [800, 1600, 400],  "batch_size": 30, "epochs": 20},
        30: {"shape": [1800, 3600, 900], "batch_size": 60, "epochs": 30},
        40: {"shape": [3200, 6400, 1600],"batch_size": 60, "epochs": 30},
    }
    lr_DNN = 1e-3 # nested call: train_specialized_dnn->train_dnn_torch/unsup

    # Device Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"<<< Wireless Power Control Simulation >>>")
    print(f"Device: {DEVICE}")

    # -------------------------------------------------------------------------
    # Phase 1: Generating Training Data & Scalers
    # This training data is for GNN training and scaler fitting. For DNNs, we
    # generate data on-the-fly per K using the scalers fitted here.
    # -------------------------------------------------------------------------
    print("\n=== Phase 1: Data Generation ===")
    
    # 1.1 Config & Training Data
    train_config = CS.init_parameters(n_links=TRAIN_K_GNN, field_length=FIELD_LENGTH)
    var = train_config.output_noise_power / train_config.tx_power
    
    _, train_dists = CS.generate_layouts(train_config, TRAIN_LAYOUTS)
    train_path = CS.compute_path_losses(train_config, train_dists)
    train_ch_shadow = CS.add_shadowing(train_path)
    train_ch_final = CS.add_fast_fading(train_ch_shadow)
    train_losses_proc = CS.proc_train_losses(train_path, train_ch_final)
    
    # 1.2 Fit Scalers using Sapo
    print("Fitting Scalers on Training Data...")
    dist_scaler = Sapo.WirelessScaler()
    loss_scaler = Sapo.WirelessScaler()
    
    dist_scaler.fit(1.0 / train_dists)
    loss_scaler.fit(np.sqrt(train_ch_final))
    
    # Transform Training Data
    norm_tr_dists = dist_scaler.transform(1.0 / train_dists)
    norm_tr_losses = loss_scaler.transform(np.sqrt(train_ch_final))

    # Save the fitted scalers
    if ENABLE_SAVING:
        with open(os.path.join(SAVE_DIR, 'scalers.pkl'), 'wb') as f:
            pickle.dump({'dist': dist_scaler, 'loss': loss_scaler}, f)
    
    # 1.3 Build Training Graph Dataset for GNN
    train_data_list = MG.proc_data(train_losses_proc, train_dists, norm_tr_dists, norm_tr_losses, TRAIN_K_GNN)
    train_loader = DataLoader(train_data_list, batch_size=64, shuffle=True)
    
    # 1.4 Generate Validation Set (In-Training Test) using Sapo for GNN
    print("Generating Validation Set...")
    _, val_data_list, val_dLoss, val_cLoss, val_cfg = Sapo.generate_evaluation_dataset(
        K_size=TRAIN_K_GNN, num_layouts=TEST_LAYOUTS, 
        dist_scaler=dist_scaler, loss_scaler=loss_scaler, # Pass trained scalers
        density=DENSITY
    )
    test_loader = DataLoader(val_data_list, batch_size=TEST_LAYOUTS, shuffle=False)

    # -------------------------------------------------------------------------
    # Phase 2: Training GNN and DNN Model
    # Note: GNN is trained once, while DNNs are specialized per K.
    # -------------------------------------------------------------------------
    print("\n=== Phase 2: Model Training ===")
    
    # --- 2A. Train GNN ---
    print(f">>> Training General GNN (K={TRAIN_K_GNN})...")
    gnn_model = MG.IGCNet().to(DEVICE)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr_GNN)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    
    best_sr = -1
    for epoch in range(1, epochs_GNN + 1):
        loss_tr = MG.train_epoch(gnn_model, train_loader, optimizer, DEVICE, TRAIN_K_GNN, var)
        if epoch % 4 == 0:
            loss_val, sr = MG.eval_epoch(
                gnn_model, test_loader, DEVICE, TRAIN_K_GNN, val_cfg, val_dLoss, val_cLoss
            )
            print(f'Epoch {epoch:03d} | TrainLoss {loss_tr:.4f} | ValLoss {loss_val:.4f} | SumRate {sr:.4f}')
            if sr > best_sr: best_sr = sr
        scheduler.step()

    # Save the trained GNN weights
    if ENABLE_SAVING:
        torch.save(gnn_model.state_dict(),  os.path.join(SAVE_DIR, f'gnn_model_K{TRAIN_K_GNN}.pth'))

    # --- 2B. Train Specialized DNNs (Pre-training) ---
    print("\n>>> Training Specialized DNNs...")
    trained_dnns = {} # Registry to store trained models
    
    for Kx in COMPARE_K_LIST:
        # Train and store in the dictionary
        model_k = Sapo.train_specialized_dnn(
            K=Kx, train_layouts=TRAIN_LAYOUTS, loss_scaler=loss_scaler,
            dnn_config=DNN_CONFIGS[Kx], device=DEVICE, density=DENSITY,
            lr=lr_DNN
        )
        trained_dnns[Kx] = model_k

        # Save each specialized DNN model
        if ENABLE_SAVING:
            torch.save(model_k.state_dict(), os.path.join(SAVE_DIR, f'dnn_model_K{Kx}.pth'))

    # -------------------------------------------------------------------------
    # Phase 3: Final Comparison Loop
    # -------------------------------------------------------------------------
    print("\n=== Phase 3: Multi-Scenario Evaluation ===")
    results = { 'K': [], 'FPLinQ': [], 'WMMSE': [], 'Greedy': [], 'GNN': [], 'DNN': [], 
                'Time_FPL': [], 'Time_WMM': [], 'Time_Grd': [], 'Time_GNN': [], 'Time_DNN': [] }

    for Kx in COMPARE_K_LIST:
        print(f'\n--- Evaluating K = {Kx} ---')
        
        # A. Retrieve Pre-Trained Models
        dnn_model_K = trained_dnns[Kx] # Fetch from registry
        
        # B. Generate FINAL Comparison Test Set using Sapo
        ch_test, graph_data_K, dLoss_K, cLoss_K, cfg_K = Sapo.generate_evaluation_dataset(
            K_size=Kx, num_layouts=TEST_LAYOUTS, 
            dist_scaler=dist_scaler, loss_scaler=loss_scaler, 
            density=DENSITY
        )
        var_K = cfg_K.output_noise_power / cfg_K.tx_power
        
        # C. Evaluate Baselines
        # 1. FPLinQ
        start = time.time()
        Y_fp = BL.FP_optimize(cfg_K, ch_test, np.ones([TEST_LAYOUTS, Kx]))
        results['Time_FPL'].append(time.time() - start)
        
        # 2. WMMSE
        start = time.time()
        Pini = np.random.rand(TEST_LAYOUTS, Kx, 1)
        Y_wm = BL.batch_WMMSE2(Pini, np.ones([TEST_LAYOUTS, Kx]), np.sqrt(ch_test), 1, var_K)
        results['Time_WMM'].append(time.time() - start)
        
        # 3. Greedy
        start = time.time()
        Y_gr = BL.simple_greedy(ch_test, np.ones([TEST_LAYOUTS, Kx]), Y_fp)
        results['Time_Grd'].append(time.time() - start)
        
        # D. Evaluate Models
        # 4. GNN (Ours)
        loader_K = DataLoader(graph_data_K, batch_size=TEST_LAYOUTS, shuffle=False)
        start = time.time()
        gnn_model.eval()
        with torch.no_grad():
            for batch in loader_K:
                batch = batch.to(DEVICE)
                out = gnn_model(batch)
            Y_gnn = out[:, 2].reshape(-1, Kx).cpu().numpy()
        results['Time_GNN'].append(time.time() - start)
        
        # 5. DNN (Specialized, supervised by WMMSE)
        # X_test_dnn = 10 * np.log10(ch_test + 1e-12).reshape(ch_test.shape[0], -1)
        X_test_norm = loss_scaler.transform(np.sqrt(ch_test))
        X_test_dnn = X_test_norm.reshape(ch_test.shape[0], -1)
        start = time.time()
        Y_dnn = MD.predict_dnn_torch(dnn_model_K, X_test_dnn, DEVICE)
        results['Time_DNN'].append(time.time() - start)
        
        # E. Compute Rates & Store
        def get_sr(Y_alloc): 
            # Sum Rate
            return np.mean(np.sum(UO.compute_rates(cfg_K, Y_alloc, dLoss_K, cLoss_K), axis=1))
            # # Average Sum Rate
            # return np.mean(UO.compute_rates(cfg_K, Y_alloc, dLoss_K, cLoss_K))
        # axis=1 → row-wise (collapse columns)
        # Weights not applied here, default to all-equal.

        results['K'].append(Kx)
        results['FPLinQ'].append(get_sr(Y_fp))
        results['WMMSE'].append(get_sr(Y_wm))
        results['Greedy'].append(get_sr(Y_gr))
        results['GNN'].append(get_sr(Y_gnn))
        results['DNN'].append(get_sr(Y_dnn))
        
        print(f"Results K={Kx} | GNN: {results['GNN'][-1]:.2f} | DNN: {results['DNN'][-1]:.2f} | WMMSE: {results['WMMSE'][-1]:.2f}")

    # Save the final results dictionary
    if ENABLE_SAVING:
        print("\nSaving simulation results...")
        with open(os.path.join(SAVE_DIR, 'simulation_results.pkl'), 'wb') as f:
            pickle.dump(results, f)

    # -------------------------------------------------------------------------
    # Phase 4: Plotting
    # -------------------------------------------------------------------------
    Sapo.plot_results(results)

if __name__ == "__main__":
    run_main()