# =============================================================================
# Filename: model_gnn.py
# Description: Graph Neural Network (GNN) for Wireless Power Control
#
# GNN consists of two main parts:
# 1. Graph Construction & Data Processing
# 2. Neural Network Layers (GNN Architecture)
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data
import utils_objective as UO  # Required for loss and rate calculation

# =============================================================================
# Graph Construction & Data Processing
# =============================================================================

# Build a single graph based on channel data (system setup)
def build_graph(loss, dist, norm_dist, norm_loss, K, threshold=300):
    """
    Converts a single layout's channel data into a PyTorch Geometric Data object.
    
    Nodes: Transceiver pairs (Links).
    Edges: Interference links between pairs (if distance < threshold).
    Node Features (x): [Direct Link Distance, Direct Link Loss, Power Placeholder].
    Edge Features (edge_attr): [Interference Link Distance].
    Target (y): Full channel matrix (magnitude) for loss calculation.
    
    Args:
        loss: Raw Channel Losses (including Shadowing/Fading) [K, K].
        dist: Raw Distance Matrix [K, K].
        norm_dist: Normalized Distance Matrix [K, K].
        norm_loss: Normalized Channel Loss Matrix [K, K].
        K: Number of links.
        threshold: Distance threshold to prune weak interference edges.
        
    Returns:
        data: torch_geometric.data.Data object.
    """
    # 1. Node Features (x)
    # x1: Normalized distance of direct link (diagonal)
    x1 = np.expand_dims(np.diag(norm_dist), axis=1)  # (K,1)
    # x2: Normalized channel loss of direct link (diagonal)
    x2 = np.expand_dims(np.diag(norm_loss), axis=1)  # (K,1)
    # x3: Placeholder for power/probability (initialized to 0)
    x3 = np.zeros((K, 1))  
    
    # Concatenate to shape (K, 3)
    x_feat = np.concatenate((x1, x2, x3), axis=1)
    x_tensor = torch.tensor(x_feat, dtype=torch.float)

    # 2. Edge Construction (Adjacency)
    # Mask diagonals (self-loops) to process interference only
    dist_masked = np.copy(dist)
    mask = np.eye(K)
    diag_dist = np.multiply(mask, dist_masked)
    
    # Set diagonals to a large value to avoid self-loops in edge list
    # (Self-loops are effectively handled by node features x1/x2)
    dist_masked = dist_masked + 1000 * diag_dist 
    
    # Pruning: Remove edges where distance > threshold (too far to interfere)
    dist_masked[dist_masked > threshold] = 0

    # Extract indices of remaining edges
    attr_ind = np.nonzero(dist_masked)
    
    # 3. Edge Features (edge_attr)
    # Use normalized distances for the edges found above
    edge_attr_val = norm_dist[attr_ind]
    edge_attr_val = np.expand_dims(edge_attr_val, axis=-1)
    edge_attr_tensor = torch.tensor(edge_attr_val, dtype=torch.float)

    # 4. Edge Index (Connectivity)
    # Convert tuple of arrays to tensor [2, Num_Edges]
    attr_ind = np.array(attr_ind)
    adj = np.zeros(attr_ind.shape)
    adj[0, :] = attr_ind[1, :] # Source
    adj[1, :] = attr_ind[0, :] # Target (Interference flows Tx j -> Rx i)
    edge_index = torch.tensor(adj, dtype=torch.long)

    # 5. Target / Label (y)
    # Store the full channel matrix (loss) for calculating Sum Rate in the loss function
    # Shape: (1, K, K) to fit batch dimension later
    y_tensor = torch.tensor(np.expand_dims(loss, axis=0), dtype=torch.float)

    data = Data(x=x_tensor, edge_index=edge_index.contiguous(), 
                edge_attr=edge_attr_tensor, y=y_tensor)
    
    return data


def proc_data(HH, dists, norm_dists, norm_HH, K):
    """
    Batch processes layouts into a list of Graph Data objects.
    """
    n_samples = HH.shape[0]
    data_list = []
    for i in range(n_samples):
        # HH[i] is the channel matrix for layout i
        data = build_graph(HH[i, :, :], dists[i, :, :], 
                           norm_dists[i, :, :], norm_HH[i, :, :], K, threshold=300)
        data_list.append(data)
    return data_list


# =============================================================================
# Neural Network Layers
# =============================================================================

# Compatibility for 'reset_parameters' in different PyG versions
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
    """
    Helper to create a Multi-Layer Perceptron (Linear -> ReLU).
    """
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])


class IGConv(MessagePassing):
    """
    Interference Graph Convolution Layer.
    """
    def __init__(self, mlp1, mlp2, **kwargs):
        # Aggregation method: 'max' (robust to number of interferers)
        super(IGConv, self).__init__(aggr='max', **kwargs)
        self.mlp1 = mlp1 # Processes messages (neighbor + edge)
        self.mlp2 = mlp2 # Processes update (self + aggregated messages)

    def reset_parameters(self):
        reset(self.mlp1)
        reset(self.mlp2)

    def update(self, aggr_out, x):
        """
        Node Update Step: Combine node's own features with aggregated interference.
        """
        # Concatenate self features (x) and aggregated interference (aggr_out)
        tmp = torch.cat([x, aggr_out], dim=1)
        
        # Apply Update MLP
        comb = self.mlp2(tmp)
        
        # Output: Keep original static features (x[:, :2]) 
        # and replace the dynamic power/state feature (3rd dim) with new output
        return torch.cat([x[:, :2], comb], dim=1)

    def forward(self, x, edge_index, edge_attr):
        """
        Propagate messages along edges.
        """
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        Message Generation Step: Create interference message from neighbor j to node i.
        """
        # Concatenate neighbor node features (x_j) and edge features (edge_attr)
        tmp = torch.cat([x_j, edge_attr], dim=1)
        
        # Apply Message MLP
        agg = self.mlp1(tmp)
        return agg

    def __repr__(self):
        """For printing layer info."""
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1, self.mlp2)


class IGCNet(torch.nn.Module):
    """
    Main GNN Architecture: Interference Graph Convolution Network.
    """
    def __init__(self):
        super(IGCNet, self).__init__()
        
        # MLP1: Message Generation
        # Input: 3 (Node Feat) + 1 (Edge Feat) = 4
        # Output: 32 (Message embedding size)
        self.mlp1 = MLP([4, 32, 32])
        
        # MLP2: Node Update
        # Input: 3 (Self Feat) + 32 (Aggregated Message) = 35
        # Output: 16 (Hidden State) -> 1 (Power/Prob)
        # Note: The final layer uses Sigmoid to output probability/power in [0, 1]
        self.mlp2 = MLP([35, 16])
        self.mlp2 = Seq(*[self.mlp2, Seq(Lin(16, 1), Sigmoid())])

        # Convolution Layer (Shared Weights across iterations/layers)
        self.conv = IGConv(self.mlp1, self.mlp2)

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        
        # Unroll the GNN for 3 iterations (Layers)
        # The output of one layer is fed as input to the next (Recurrent/Unrolled)
        x1 = self.conv(x=x0, edge_index=edge_index, edge_attr=edge_attr)
        x2 = self.conv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        out = self.conv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
        
        return out  # Shape: [Total_Nodes(K), 3], where col 2 (third dimension) is the power p
    

# =============================================================================
# Training & Evaluation Loops
# =============================================================================

def train_epoch(model, loader, optimizer, device, K, var_noise):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        loss = UO.sumrate_loss(data, out, K, var_noise)
        
        loss.backward()
        optimizer.step()
        
        # Accumulate loss (weighted by batch size for accuracy)
        total_loss += loss.item() * data.num_graphs
        
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device, K, test_config, directLoss, crossLoss):
    """
    Performs evaluation on the test set.
    Returns: Average Loss (Validation) and Average Sum Rate (Testing Metric).
    """
    model.eval()
    total_loss = 0.0
    last_out = None
    
    # 1. Compute Validation Loss
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            
            # Use test configuration noise for loss calculation
            noise_val = test_config.output_noise_power / test_config.tx_power
            loss = UO.sumrate_loss(data, out, K, noise_val)
            
            total_loss += loss.item() * data.num_graphs
            last_out = out # Store last batch for quick sanity check if needed
            
    # 2. Compute Actual Sum Rate (using numpy utility)
    # We need to extract the 'p' (power) predictions from the model for the whole dataset
    # Note: For strict accuracy on large datasets, we should accumulate 'p' inside the loop.
    # However, following the original code structure, we iterate again or assume loader structure.
    # **Correction/Improvement**: The original code only computed rate on 'last_out' or had a logic gap.
    # Here we will iterate properly to compute the full test set rate.
    
    all_rates = []
    
    with torch.no_grad():
        batch_idx = 0
        for data in loader:
            data = data.to(device)
            out = model(data)
            
            # Extract p: [Batch_Size * K]
            p_tensor = out[:, 2] 
            # Reshape to [Batch_Size, K]
            # Note: data.num_graphs is the actual batch size of this iteration
            p_matrix = p_tensor.reshape(data.num_graphs, K).cpu().numpy()
            
            # Compute rates for this batch
            # We need to slice the pre-computed losses (directLoss/crossLoss) for this batch
            start = batch_idx * loader.batch_size
            end = start + data.num_graphs
            
            # Slice the global loss matrices to match the current batch
            # Ensure directLoss/crossLoss passed in are large enough
            batch_dLoss = directLoss[start:end]
            batch_cLoss = crossLoss[start:end]
            
            rates = UO.compute_rates(test_config, p_matrix, batch_dLoss, batch_cLoss)
            all_rates.append(np.sum(rates, axis=1)) # Sum over users (axis 1)
            
            batch_idx += 1

    # Concatenate all batch results
    all_rates_flat = np.concatenate(all_rates)
    avg_sum_rate = np.mean(all_rates_flat)
    
    return total_loss / len(loader.dataset), avg_sum_rate

# =============================================================================
# QoS constrained loss and training
# =============================================================================
def train_epoch_QoS(model, loader, optimizer, device, K, var_noise, r_min=0.5, penalty_weight=10.0):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        # loss = UO.sumrate_loss(data, out, K, var_noise)
        # Replace UO.sumrate_loss with the new constrained version
        loss = constrained_sumrate_loss(data, out, K, var_noise, r_min, penalty_weight)
        
        loss.backward()
        optimizer.step()
        
        # Accumulate loss (weighted by batch size for accuracy)
        total_loss += loss.item() * data.num_graphs
        
    return total_loss / len(loader.dataset)

# New loss function that incorporates QoS constraint (r_min) with a penalty for violations.
def constrained_sumrate_loss(data, out, K, var_noise, r_min, penalty_weight=10.0):
    """
    Calculates the negative Sum Rate loss PLUS a penalty for violating the minimum rate constraint.
    Implemented directly in model_gnn.py to avoid modifying utils_objective.py.
    """
    # 1. Extract and reshape power predictions
    p = out[:, 2]
    p = torch.reshape(p, (-1, K, 1))
    
    # 2. Prepare Channel Matrix H
    absH2 = data.y
    absH2 = absH2.permute(0, 2, 1)

    # 3. Calculate Received Power
    rx_power = torch.mul(absH2, p)
    
    # 4. Separate Signal and Interference
    mask = torch.eye(K, device=out.device)
    valid_rx_power = torch.sum(rx_power * mask, dim=1)
    interference = torch.sum(rx_power * (1 - mask), dim=1) + var_noise 
    
    # 5. Calculate Individual Rates
    rate = torch.log2(1 + valid_rx_power / (interference + 1e-12))
    
    # --- NEW CONSTRAINT LOGIC ---
    # 6. Calculate Sum Rate (Original Objective) 
    sum_rate = torch.mean(torch.sum(rate, dim=1))
    
    # 7. Calculate Penalty for constraint violations
    # torch.clamp isolates rates BELOW r_min, ignoring rates ABOVE r_min
    violations = torch.clamp(r_min - rate, min=0.0)
    
    # Use squared penalty for smoother gradients during backpropagation
    penalty = torch.mean(torch.sum(torch.square(violations), dim=1))
    
    # 8. Total Loss
    # We want to minimize the negative sum rate AND minimize the penalty
    total_loss = -sum_rate + (penalty_weight * penalty)
    
    return total_loss