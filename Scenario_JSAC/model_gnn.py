# =============================================================================
# Filename: model_gnn.py
# Description: Graph Neural Network (GNN) for JSAC Wireless Power Control
#
# GNN consists of two main parts:
# 1. Graph Construction & Data Processing
# 2. Neural Network Layers (GNN Architecture)
#
# JSAC changes from D2D (v2.3):
#   - Edge construction uses interf_mask (same channel, different Blue car)
#     instead of distance-only thresholding
#   - Per-group softmax replaces Sigmoid for per-Blue-car power budget
#   - Loss: Green-only sum-rate + Yellow SINR penalty
#   - Metadata (group_ids, green/yellow masks) stored in PyG Data objects
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import softmax as pyg_softmax
import utils_objective as UO

# =============================================================================
# Graph Construction & Data Processing
# =============================================================================

def build_graph(loss, dist, norm_dist, norm_loss, K, interf_mask, group_ids, green_mask, yellow_mask):
    """
    Converts a single layout's channel data into a PyTorch Geometric Data object.

    Nodes: Transceiver pairs (Links).
    Edges: Two types —
        (a) Interference edges (same channel, different Blue car) from interf_mask
        (b) Intra-group edges (same Blue car, different links) for within-group coordination
    Node Features (x): [Direct Distance, Direct Loss, Link Type (1=Green), Power Placeholder].
    Edge Features (edge_attr): [Norm Distance, Edge Type (0=interf, 1=intra), Norm Channel Gain].
    Target (y): Full channel matrix (magnitude) for loss calculation.
    Metadata: group_ids, green_mask, yellow_mask stored for loss computation.

    Args:
        loss:        Raw Channel Losses (including Shadowing/Fading) [K, K].
        dist:        Raw Distance Matrix [K, K].
        norm_dist:   Normalized Distance Matrix [K, K].
        norm_loss:   Normalized Channel Loss Matrix [K, K].
        K:           Number of links.
        interf_mask: (K, K) binary — 1 where two links interfere.
        group_ids:   (K,) int — Blue car index for each link.
        green_mask:  (K,) bool — True for Green links.
        yellow_mask: (K,) bool — True for Yellow links.

    Returns:
        data: torch_geometric.data.Data object.
    """
    # 1. Node Features (x): 4 features
    x1 = np.expand_dims(np.diag(norm_dist), axis=1)  # (K,1) direct distance
    x2 = np.expand_dims(np.diag(norm_loss), axis=1)  # (K,1) direct loss
    x3 = np.expand_dims(np.asarray(green_mask).astype(float), axis=1)  # (K,1) link type
    x4 = np.zeros((K, 1))  # power placeholder
    x_feat = np.concatenate((x1, x2, x3, x4), axis=1)
    x_tensor = torch.tensor(x_feat, dtype=torch.float)

    # 2. Edge Construction: interference + intra-group
    # Interference: same channel, different Blue car (from interf_mask)
    # Intra-group: same Blue car, different link (enables within-group coordination)
    group_ids_arr = np.asarray(group_ids)
    intra_group = (group_ids_arr[:, None] == group_ids_arr[None, :])
    np.fill_diagonal(intra_group, False)
    combined_mask = interf_mask.astype(bool) | intra_group
    edge_ind = np.nonzero(combined_mask)

    # 3. Edge Features: [normalized distance, edge_type, normalized channel gain]
    edge_dist = norm_dist[edge_ind]
    edge_type = intra_group[edge_ind].astype(float)  # 0=interference, 1=intra-group
    edge_gain = norm_loss[edge_ind]  # normalized channel amplitude (actual interference strength)
    edge_attr_val = np.stack([edge_dist, edge_type, edge_gain], axis=-1)
    edge_attr_tensor = torch.tensor(edge_attr_val, dtype=torch.float)

    # 4. Edge Index [2, Num_Edges]
    edge_arr = np.array(edge_ind)
    adj = np.zeros(edge_arr.shape)
    adj[0, :] = edge_arr[1, :]  # Source (Tx_j)
    adj[1, :] = edge_arr[0, :]  # Target (Rx_i)
    edge_index = torch.tensor(adj, dtype=torch.long)

    # 5. Target: full channel matrix [1, K, K]
    y_tensor = torch.tensor(np.expand_dims(loss, axis=0), dtype=torch.float)

    # 6. JSAC metadata (stored as node-level attributes for proper PyG batching)
    group_ids_tensor = torch.tensor(np.asarray(group_ids), dtype=torch.long)
    green_mask_tensor = torch.tensor(np.asarray(green_mask).copy(), dtype=torch.bool)
    yellow_mask_tensor = torch.tensor(np.asarray(yellow_mask).copy(), dtype=torch.bool)

    data = Data(x=x_tensor, edge_index=edge_index.contiguous(),
                edge_attr=edge_attr_tensor, y=y_tensor,
                group_ids=group_ids_tensor,
                green_mask=green_mask_tensor,
                yellow_mask=yellow_mask_tensor)

    return data


def proc_data(HH, dists, norm_dists, norm_HH, K, interf_mask, group_ids, green_mask, yellow_mask):
    """
    Batch processes layouts into a list of Graph Data objects.
    """
    n_samples = HH.shape[0]
    data_list = []
    for i in range(n_samples):
        data = build_graph(HH[i], dists[i], norm_dists[i], norm_HH[i], K,
                           interf_mask, group_ids, green_mask, yellow_mask)
        data_list.append(data)
    return data_list


# =============================================================================
# Neural Network Layers
# =============================================================================

try:
    from torch_geometric.nn.inits import reset
except ImportError:
    def reset(module: nn.Module):
        """Recursively call reset_parameters() if present (PyG compat)."""
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
        for child in module.children():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()

def MLP(channels, batch_norm=True):
    """Helper to create a Multi-Layer Perceptron (Linear -> ReLU)."""
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])


class IGConv(MessagePassing):
    """Interference Graph Convolution Layer."""
    def __init__(self, mlp1, mlp2, **kwargs):
        super(IGConv, self).__init__(aggr='max', **kwargs)
        self.mlp1 = mlp1
        self.mlp2 = mlp2

    def reset_parameters(self):
        reset(self.mlp1)
        reset(self.mlp2)

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        comb = self.mlp2(tmp)
        return torch.cat([x[:, :3], comb], dim=1)  # preserve [dist, loss, link_type]

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        return self.mlp1(tmp)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1, self.mlp2)


class IGCNet(torch.nn.Module):
    """
    IGCNet for JSAC: outputs raw logits (no Sigmoid).
    Per-group softmax is applied in the loss function for differentiable
    power allocation that respects the per-Blue-car power budget.

    Node features: [direct_dist, direct_loss, link_type, power_logit] (4 dims)
    Edge features: [norm_distance, edge_type, norm_channel_gain] (3 dims)
    """
    def __init__(self):
        super(IGCNet, self).__init__()

        # MLP1: Message Generation (input: 4 node feat + 3 edge feat = 7)
        self.mlp1 = MLP([7, 32, 32])

        # MLP2: Node Update (input: 4 self feat + 32 aggregated = 36)
        # Output: raw logit — per-group softmax applied externally
        self.mlp2 = MLP([36, 32])
        self.mlp2 = Seq(*[self.mlp2, Lin(32, 1)])

        self.conv = IGConv(self.mlp1, self.mlp2)

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index

        x1 = self.conv(x=x0, edge_index=edge_index, edge_attr=edge_attr)
        x2 = self.conv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        x3 = self.conv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
        out = self.conv(x=x3, edge_index=edge_index, edge_attr=edge_attr)

        return out  # [total_nodes, 4], col 3 = raw logit


# =============================================================================
# Per-Group Softmax (Option B from WORKPLAN)
# =============================================================================

def apply_group_softmax(raw_logits, group_ids, batch, n_blue):
    """
    Apply softmax within each (graph, Blue car) group.
    Guarantees per-group power sums to 1.0 (= normalized Pmax).

    Args:
        raw_logits: [total_nodes] raw output from GNN (col 2)
        group_ids:  [total_nodes] Blue car index per node
        batch:      [total_nodes] graph index per node (from PyG batching)
        n_blue:     int — number of Blue cars

    Returns:
        powers: [total_nodes] — sums to 1.0 within each (graph, Blue car) group
    """
    group_key = batch * n_blue + group_ids
    return pyg_softmax(raw_logits, group_key)


# =============================================================================
# JSAC Loss Function
# =============================================================================

def constrained_sumrate_loss_jsac(data, out, K, var_noise, n_blue, sinr_min, penalty_weight):
    """
    JSAC training loss: maximize Green sum-rate, penalize Yellow SINR violations.
    Per-group power constraint is enforced via softmax (sum=1 per group).

    Args:
        data:            PyG batch (y, group_ids, green_mask, yellow_mask, batch)
        out:             Model output [total_nodes, 4]; col 3 = raw logit
        K:               Number of links per layout
        var_noise:       Normalized noise (output_noise_power / tx_power)
        n_blue:          Number of Blue cars
        sinr_min:        Minimum SINR for Yellow links (linear scale)
        penalty_weight:  Weight for Yellow SINR violation penalty
                         (set to 0 for unconstrained Green-only optimization)

    Returns:
        loss: scalar tensor
    """
    # 1. Per-group softmax → power allocation
    raw_p = out[:, 3]
    p = apply_group_softmax(raw_p, data.group_ids, data.batch, n_blue)
    p = p.reshape(-1, K, 1)

    # 2. Channel matrix H² [batch, K, K]
    absH2 = data.y.permute(0, 2, 1)

    # 3. Received power [batch, K, K]
    rx_power = absH2 * p

    # 4. Signal (diagonal) and interference+noise (off-diagonal)
    eye = torch.eye(K, device=out.device)
    signal = torch.sum(rx_power * eye, dim=1)               # [batch, K]
    interference = torch.sum(rx_power * (1 - eye), dim=1) + var_noise  # [batch, K]

    # 5. SINR [batch, K]
    sinr = signal / (interference + 1e-12)

    # 6. Green-only sum-rate (objective)
    green_mask = data.green_mask.reshape(-1, K).float()
    green_rate = torch.log2(1 + sinr) * green_mask
    green_sumrate = torch.mean(torch.sum(green_rate, dim=1))

    # 7. Yellow SINR penalty (constraint)
    yellow_mask = data.yellow_mask.reshape(-1, K).float()
    yellow_violations = torch.clamp(sinr_min - sinr, min=0.0) * yellow_mask
    penalty = torch.mean(torch.sum(torch.square(yellow_violations), dim=1))

    return -green_sumrate + penalty_weight * penalty


# =============================================================================
# Training & Evaluation Loops
# =============================================================================

def train_epoch(model, loader, optimizer, device, K, var_noise, n_blue,
                sinr_min=1.0, penalty_weight=10.0):
    """Performs one epoch of JSAC training."""
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        loss = constrained_sumrate_loss_jsac(
            data, out, K, var_noise, n_blue, sinr_min, penalty_weight)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device, K, test_config, directLoss, crossLoss,
               green_mask_np, yellow_mask_np, n_blue,
               sinr_min=1.0, penalty_weight=10.0):
    """
    Evaluation: computes loss, Green sum-rate, and Yellow SINR violation rate.

    Args:
        model, loader, device, K: standard
        test_config:     Config object (for noise params)
        directLoss:      [N_test, K] direct channel gains (numpy)
        crossLoss:       [N_test, K, K] cross channel gains (numpy)
        green_mask_np:   (K,) bool numpy array
        yellow_mask_np:  (K,) bool numpy array
        n_blue:          int
        sinr_min:        float (linear scale)
        penalty_weight:  float

    Returns:
        avg_loss:       float — training-style loss for monitoring
        green_sumrate:  float — average Green sum-rate
        violation_rate: float — percentage of Yellow links below sinr_min
    """
    model.eval()
    total_loss = 0.0
    all_powers = []

    var_noise = test_config.output_noise_power / test_config.tx_power

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)

            # Loss
            loss = constrained_sumrate_loss_jsac(
                data, out, K, var_noise, n_blue, sinr_min, penalty_weight)
            total_loss += loss.item() * data.num_graphs

            # Extract powers (after group softmax)
            raw_p = out[:, 3]
            p = apply_group_softmax(raw_p, data.group_ids, data.batch, n_blue)
            p_matrix = p.reshape(data.num_graphs, K).cpu().numpy()
            all_powers.append(p_matrix)

    all_powers = np.concatenate(all_powers, axis=0)

    # Numpy evaluation metrics
    green_sr = UO.compute_green_sumrate(
        test_config, all_powers, directLoss, crossLoss, green_mask_np)
    viol_rate, _ = UO.compute_yellow_sinr_violation(
        test_config, all_powers, directLoss, crossLoss, yellow_mask_np, sinr_min)

    return total_loss / len(loader.dataset), green_sr, viol_rate
