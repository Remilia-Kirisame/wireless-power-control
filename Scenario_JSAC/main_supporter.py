# ==============================================================================
# Filename: main_supporter.py
# Description: Helper functions and classes for main execution.
#              Includes data generation, normalization, and plotting.
# ==============================================================================

import numpy as np
import torch
import matplotlib.pyplot as plt

# Import project modules
import config_system as CS
import utils_objective as UO
import baselines as BL
import model_gnn as MG

# =============================================================================
# 1. Helper Class: Wireless Scaler
# =============================================================================
class WirelessScaler:
    """
    3-category Z-score normalization for JSAC channel/distance matrices.

    Categories:
        1. Direct-sensing   (diagonal entries where link is Yellow)
        2. Direct-communication (diagonal entries where link is Green)
        3. Interference     (off-diagonal entries where interf_mask == 1)

    All other entries (intra-Blue, different-channel) are structural zeros
    and stay zero after normalization.
    """
    def __init__(self):
        # Category 1: Direct-sensing (Yellow diagonal)
        self.sense_mean = None
        self.sense_var  = None
        # Category 2: Direct-communication (Green diagonal)
        self.comm_mean = None
        self.comm_var  = None
        # Category 3: Interference (off-diagonal, interf_mask == 1)
        self.interf_mean = None
        self.interf_var  = None

    def fit(self, X, interf_mask, yellow_mask, green_mask):
        """
        Compute per-category statistics from training data.

        Args:
            X:           (L, K, K) — data matrix (e.g. 1/dists or sqrt(channel_losses))
            interf_mask: (K, K) binary — 1 where two links interfere
            yellow_mask: (K,) bool — True for Yellow (sensing) links
            green_mask:  (K,) bool — True for Green (communication) links
        """
        L, K, _ = X.shape

        # Build per-category masks [K, K]
        diag = np.eye(K, dtype=bool)
        sense_mask = diag & yellow_mask[:, None]   # diagonal rows where link is Yellow
        comm_mask  = diag & green_mask[:, None]     # diagonal rows where link is Green
        interf = interf_mask.astype(bool)           # off-diagonal interference entries

        n_sense = np.count_nonzero(sense_mask)   # num Yellow links
        n_comm  = np.count_nonzero(comm_mask)     # num Green links
        n_interf = np.count_nonzero(interf)       # num interference pairs

        # Category 1: Direct-sensing
        vals = X[:, sense_mask]                     # (L, n_sense)
        self.sense_mean = np.mean(vals)
        self.sense_var  = np.std(vals)

        # Category 2: Direct-communication
        vals = X[:, comm_mask]                      # (L, n_comm)
        self.comm_mean = np.mean(vals)
        self.comm_var  = np.std(vals)

        # Category 3: Interference
        vals = X[:, interf]                         # (L, n_interf)
        self.interf_mean = np.mean(vals)
        self.interf_var  = np.std(vals)

    def transform(self, X, interf_mask, yellow_mask, green_mask):
        """
        Normalize X using stored statistics. Zero entries stay zero.

        Args: same shapes as fit().
        Returns: (L, K, K) normalized matrix.
        """
        if self.sense_mean is None:
            raise ValueError("Scaler not fitted! Call fit() with training data first.")

        L, K, _ = X.shape
        result = np.zeros_like(X)

        diag = np.eye(K, dtype=bool)
        sense_mask = diag & yellow_mask[:, None]
        comm_mask  = diag & green_mask[:, None]
        interf = interf_mask.astype(bool)

        # Normalize each category independently
        result[:, sense_mask] = (X[:, sense_mask] - self.sense_mean) / (self.sense_var + 1e-12)
        result[:, comm_mask]  = (X[:, comm_mask]  - self.comm_mean)  / (self.comm_var  + 1e-12)
        result[:, interf]     = (X[:, interf]     - self.interf_mean) / (self.interf_var + 1e-12)

        return result


# =============================================================================
# 2. Helper Function: Training Data Generator
# =============================================================================
def generate_training_dataset(general_para, num_layouts, shuffle_channels=False):
    """
    Generate training data AND fit scalers on it.

    Pipeline: JSAC layouts → channel losses → fit scalers → normalize → build GNN graphs.

    Args:
        general_para:    init_parameters_JSAC instance
        num_layouts:     number of training layouts to generate
        shuffle_channels: randomize channel assignment per Blue car

    Returns:
        train_data_list: list of PyG Data objects (ready for DataLoader)
        dist_scaler:     fitted WirelessScaler for distances
        loss_scaler:     fitted WirelessScaler for channel losses
        metadata: dict with keys:
            'channel_losses', 'dists', 'group_ids', 'link_types',
            'channel_ids', 'interf_mask', 'green_mask', 'yellow_mask'
    """
    K = general_para.n_links
    M = general_para.n_links_per_blue

    # 1. Generate layouts + channels
    layouts, dists, group_ids, link_types = CS.generate_layouts_jsac(general_para, num_layouts)
    channel_ids, interf_mask, green_mask, yellow_mask = CS.jsac_metadata(
        group_ids, link_types, M, shuffle_channels=shuffle_channels)
    _, channel_losses = CS.compute_channel_losses_jsac(general_para, dists, interf_mask)

    # 2. Fit scalers on training data (3-category)
    dist_scaler = WirelessScaler()
    loss_scaler = WirelessScaler()
    dist_scaler.fit(1.0 / dists, interf_mask, yellow_mask, green_mask)
    loss_scaler.fit(np.sqrt(channel_losses), interf_mask, yellow_mask, green_mask)

    # 3. Normalize + build graph dataset
    norm_dists  = dist_scaler.transform(1.0 / dists, interf_mask, yellow_mask, green_mask)
    norm_losses = loss_scaler.transform(np.sqrt(channel_losses), interf_mask, yellow_mask, green_mask)
    train_data_list = MG.proc_data(
        channel_losses, dists, norm_dists, norm_losses, K,
        interf_mask, group_ids, green_mask, yellow_mask
    )

    metadata = {
        'group_ids': group_ids,
        # 'link_types': link_types,
        # 'channel_ids': channel_ids,
        'interf_mask': interf_mask,
        'green_mask': green_mask,
        'yellow_mask': yellow_mask,
    }
    return train_data_list, dist_scaler, loss_scaler, metadata


# =============================================================================
# 3. Helper Function: Evaluation Data Generator
# =============================================================================
def generate_evaluation_dataset(general_para, num_layouts, dist_scaler, loss_scaler,
                                interf_mask, group_ids, green_mask, yellow_mask,
                                shuffle_channels=False):
    """
    Generate evaluation data using pre-fitted scalers from training.

    Args:
        general_para:    init_parameters_JSAC instance
        num_layouts:     number of evaluation layouts
        dist_scaler:     fitted WirelessScaler (distances)
        loss_scaler:     fitted WirelessScaler (channel losses)
        interf_mask:     (K, K) binary mask from training topology
        group_ids:       (K,) int array
        green_mask:      (K,) bool array
        yellow_mask:     (K,) bool array
        shuffle_channels: randomize channel assignment per Blue car

    Returns:
        channel_losses:  (N, K, K)
        graph_data_list: list of PyG Data objects
        d_losses:        (N, K) direct-link channel losses
        c_losses:        (N, K, K) cross-link channel losses
        layouts:         (N, K, 4) — [tx_x, tx_y, rx_x, rx_y] per link
        conf:            the general_para used
    """
    K = general_para.n_links
    M = general_para.n_links_per_blue

    # A. Generate new layouts + channels
    layouts, dists, eval_group_ids, eval_link_types = CS.generate_layouts_jsac(general_para, num_layouts)

    # Use fresh metadata if shuffle_channels is enabled (each dataset gets its own
    # random channel permutation); otherwise reuse training metadata for consistency.
    if shuffle_channels:
        _, interf_mask, green_mask, yellow_mask = CS.jsac_metadata(
            eval_group_ids, eval_link_types, M, shuffle_channels=True)
        group_ids = eval_group_ids

    _, channel_losses = CS.compute_channel_losses_jsac(general_para, dists, interf_mask)

    # B. Normalize using pre-fitted scalers
    norm_dists  = dist_scaler.transform(1.0 / dists, interf_mask, yellow_mask, green_mask)
    norm_losses = loss_scaler.transform(np.sqrt(channel_losses), interf_mask, yellow_mask, green_mask)

    # C. Build GNN graphs
    graph_data_list = MG.proc_data(
        channel_losses, dists, norm_dists, norm_losses, K,
        interf_mask, group_ids, green_mask, yellow_mask
    )

    # D. Extract evaluation helpers
    d_losses = UO.get_directLink_channel_losses(channel_losses)
    c_losses = UO.get_crossLink_channel_losses(channel_losses)

    return channel_losses, graph_data_list, d_losses, c_losses, layouts, general_para



# =============================================================================
# 4. Helper Function: Plotting (to be rewritten in Step 6/7)
# =============================================================================
# D2D plot_results, print_qos_metrics, plot_qos_cdf removed.
# JSAC-specific plotting will be added in Step 6 (main.py) / Step 7 (test_JSAC.py).