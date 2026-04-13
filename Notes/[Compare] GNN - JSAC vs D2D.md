# GNN — JSAC vs D2D: What Changes and What Doesn't

This note analyzes how the D2D-to-JSAC transition affects the GNN (IGCNet) implementation. Written before coding Step 4 of the WORKPLAN, following the same format as [WMMSE - JSAC vs D2D.md](WMMSE%20-%20JSAC%20vs%20D2D.md).

---

## 0. Is It Really "Just the Loss Function"?

**Short answer: mostly yes, but not entirely.**

The core insight is that GNN is trained end-to-end — the network learns whatever behavior the loss function rewards. Unlike WMMSE, where every constraint must be manually coded into the iterative update equations, a GNN can absorb new objectives and constraints through the training signal alone. This is why GNN adaptation is structurally simpler than WMMSE adaptation.

However, there are changes beyond the loss function:

| Component | Change needed? | Why |
|---|---|---|
| **Loss function** | Yes (main change) | Green-only objective, Yellow SINR penalty |
| **Power output layer** | Yes | Per-group softmax replaces per-link sigmoid |
| **Graph construction** | Yes | Two edge types: interference (interf_mask) + intra-group (same Blue car); 3D edge features |
| **Node features** | Yes | Added link_type (Green/Yellow) as 4th feature — GNN needs this to differentiate link roles |
| **IGConv message passing** | No | General enough — works on any graph topology |
| **MLP dimensions** | Yes | 4 node feat + 3 edge feat → MLP1 input=7; MLP2 input=36, hidden=32; 4 iterations |
| **Training loop** | Minor | Pass extra metadata (masks, group_ids, n_blue) |
| **Evaluation** | Yes | Green sum-rate + Yellow violation rate |

The IGConv message-passing mechanism is reused as-is. But the **inputs, outputs, and dimensions** all changed — more than initially expected. See Sections 2, 5, and 6 below for the full reasoning and the troubleshooting that led to these changes.

---

## 1. Problem Recap: What the GNN Must Learn

### D2D
```
Maximize:   sum over ALL k of rate_k
Subject to: 0 <= p_k <= Pmax  for each link k
            (QoS variant: rate_k >= r_min for all k)

GNN output:  p_k = sigmoid(.) * Pmax  →  each link independently in [0, Pmax]
```

### JSAC
```
Maximize:   sum over GREEN k of rate_k
Subject to: sum_{k in Blue_b} p_k <= Pmax    for each Blue car b   (shared budget)
            SINR_k >= gamma_min               for Yellow k only     (sensing QoS)

GNN output:  p_k needs to satisfy per-GROUP budget, not per-link
```

The GNN must learn three new behaviors:
1. **Prioritize Green links** in its power allocation
2. **Protect Yellow links** from falling below SINR threshold
3. **Coordinate within each group** to share the power budget

---

## 2. Graph Construction — Structural Change

### D2D `build_graph`
```python
# Edges: any pair within 300m distance
dist_masked[dist_masked > threshold] = 0
edge_index = nonzero(dist_masked)
```

Every link potentially interferes with every other link. The 300m threshold is a computational optimization (pruning weak interference), not a structural constraint.

### JSAC `build_graph`
```python
# Two types of edges:
# 1. Interference: same channel, different Blue car (from interf_mask)
# 2. Intra-group: same Blue car, different link (for within-group coordination)
intra_group = (group_ids[:, None] == group_ids[None, :]) & ~eye(K)
combined_mask = interf_mask | intra_group
edge_index = nonzero(combined_mask)
```

In JSAC, the orthogonal channel structure means most link pairs do **not** interfere. A link on channel 0 in Blue car A only interferes with links on channel 0 in Blue cars B, C, D, etc. This is not an approximation — it's the physics.

**Key difference:** In D2D, edges encode "might interfere" (distance heuristic). In JSAC, edges encode two distinct relationships: "does interfere" (interference edges) and "shares a power budget" (intra-group edges).

**Why intra-group edges are essential:** Without channel shuffling, links within the same Blue car are on different orthogonal channels and share zero interference edges. The 5 links per group exist in 5 completely disconnected subgraphs. Without intra-group edges, the GNN cannot coordinate power allocation within a group — the per-group softmax just normalizes uncoordinated logits to near-uniform output (≈ equal power). This was the primary reason the initial GNN implementation barely beat the equal-power baseline.

### Impact on graph density

| | D2D (K=50) | JSAC (B=10, M=5, K=50) |
|---|---|---|
| Max possible edges | K(K-1) = 2450 | K(K-1) = 2450 |
| Interference edges | ~1000-2000 (distance pruning) | 450 (9 interferers per link × 50 / 2 directions) |
| Intra-group edges | N/A | 200 (10 groups × 5×4 directed pairs) |
| **Total edges** | ~1000-2000 | **650** |
| Edge density | ~40-80% | ~27% |

The JSAC graph is sparser overall but richer in structure (two edge types with different semantics).

### Pseudocode: `build_graph_jsac`

```
Input:
    loss:        (K, K) channel magnitudes (already masked by interf_mask + diagonal)
    dist:        (K, K) raw distances
    norm_dist:   (K, K) normalized distances
    norm_loss:   (K, K) normalized channel losses
    interf_mask: (K, K) binary — 1 where two links interfere
    group_ids:   (K,) int — Blue car index
    green_mask:  (K,) bool
    yellow_mask: (K,) bool
    K:           int

Output:
    PyG Data object

Algorithm:
    # --- Node features (4 dims — expanded from D2D's 3) ---
    x1 = diag(norm_dist)                       # (K,1) direct-link distance
    x2 = diag(norm_loss)                       # (K,1) direct-link channel loss
    x3 = green_mask.astype(float)              # (K,1) link type: 1=Green, 0=Yellow
    x4 = zeros(K, 1)                           # (K,1) power placeholder
    x = concat(x1, x2, x3, x4)                # (K, 4)

    # --- Edges: interference + intra-group ---
    intra_group = (group_ids[:, None] == group_ids[None, :]) & ~eye(K)
    combined_mask = interf_mask | intra_group
    edge_index = nonzero(combined_mask)

    # --- Edge features (3 dims — expanded from D2D's 1) ---
    edge_dist  = norm_dist[edge_index]         # normalized distance
    edge_type  = intra_group[edge_index]       # 0=interference, 1=intra-group
    edge_gain  = norm_loss[edge_index]         # normalized channel amplitude
    edge_attr  = stack(edge_dist, edge_type, edge_gain)  # (num_edges, 3)

    # --- Target ---
    y = loss.unsqueeze(0)                      # (1, K, K) for loss computation

    # --- Metadata ---
    data.green_mask = green_mask               # (K,) bool
    data.yellow_mask = yellow_mask             # (K,) bool
    data.group_ids = group_ids                 # (K,) int

    return Data(x, edge_index, edge_attr, y, ...)
```

**Changes from D2D:**

1. **Edge construction** uses `interf_mask | intra_group` instead of distance thresholding — two edge types with distinct semantics
2. **Node features** expanded from 3 to 4 dims (added `link_type`)
3. **Edge features** expanded from 1 to 3 dims (added `edge_type` and `norm_channel_gain`)
4. Data object carries extra metadata (masks, group_ids) for the loss function

---

## 3. Power Output Layer — The Per-Group Constraint

This is the most conceptually important change after the loss function.

### D2D: Independent Sigmoid

```python
# In IGCNet.__init__:
self.mlp2 = Seq(MLP([35, 16]), Lin(16, 1), Sigmoid())

# Effect: each node outputs p_k in [0, 1], scaled by Pmax later
# Constraint: 0 <= p_k <= Pmax  ← automatically satisfied
```

Each link's power is independent. The sigmoid naturally enforces the per-link box constraint.

### JSAC: Per-Group Softmax (WORKPLAN Option B)

The per-group constraint `sum_{k in group_b} p_k <= Pmax` cannot be enforced by independent sigmoids. We need a **post-processing step** that coordinates within each group.

**Option A — Post-hoc projection (WORKPLAN Option A):**
```python
# After GNN forward pass (sigmoid outputs):
for each group b:
    raw = sigmoid_output[group_b]       # [M] values in (0,1)
    if sum(raw) > 1:
        raw = raw / sum(raw)            # scale down to sum = 1
    p[group_b] = raw * Pmax             # sum <= Pmax
```
Allows allocating less than Pmax, but the `if` branch creates a non-smooth gradient landscape. Also risks the GNN getting stuck at low power early in training (same issue WMMSE had — the projection only scales *down*, never *up*).

**Option B — Softmax per group (WORKPLAN Option B, our choice):**
```python
# Replace the final Sigmoid with per-group Softmax:
for each group b:
    logits = raw_output[group_b]         # [M] raw logits (before any activation)
    p[group_b] = softmax(logits) * Pmax  # sums to exactly Pmax per group
```
Forces each group to use the full budget. This is actually desirable:
1. **Avoids the low-power trap** — same reasoning as WMMSE initialization at equal power. The softmax ensures the GNN always works with the full power budget, and learns to *distribute* it well rather than deciding *how much* to use.
2. **Fully differentiable** — softmax gradients are well-behaved and standard in PyTorch.
3. **Hard constraint satisfaction** — per-group sum is exactly Pmax by construction. No penalty term needed for the power constraint.

The tradeoff (can't allocate less than Pmax) is minor: in interference channels, using full power is almost always near-optimal because the objective is sum-rate, not energy efficiency.

**This is our choice, consistent with WORKPLAN Section 4C.**

### Pseudocode: Per-Group Softmax

```python
def softmax_power_per_group(logits, group_ids, n_blue):
    """
    logits: (Batch, K) — raw GNN output (before activation)
    Returns: (Batch, K) — power fractions that sum to 1.0 per group
                          (multiply by Pmax for absolute power)
    """
    p_out = torch.zeros_like(logits)
    for b in range(n_blue):
        mask = (group_ids == b)
        p_out[:, mask] = torch.softmax(logits[:, mask], dim=1)
    return p_out
```

**Architecture change:** The final layer of `mlp2` drops the `Sigmoid()` — we output raw logits and apply `softmax_power_per_group` in the forward pass or training loop. This means the `update()` method in IGConv appends the raw logit (1 dim) instead of a sigmoid output.

---

## 4. Loss Function — The Main Adaptation

### D2D Loss: `sumrate_loss`

```python
rate = log2(1 + signal / interference)          # (Batch, K)
sum_rate = mean(sum(rate, dim=1))                # scalar: average over batch
loss = -sum_rate                                 # minimize negative = maximize
```

### D2D QoS Loss: `constrained_sumrate_loss`

```python
rate = log2(1 + signal / interference)           # (Batch, K)
sum_rate = mean(sum(rate, dim=1))                 # over ALL K links
violations = clamp(r_min - rate, min=0)           # over ALL K links
penalty = mean(sum(violations^2, dim=1))
loss = -sum_rate + penalty_weight * penalty
```

### JSAC Loss: `jsac_sumrate_loss` (new)

```python
rate = log2(1 + signal / interference)            # (Batch, K)

# 1. Objective: Green links only
green_rate = rate[:, green_mask]                   # (Batch, K_green)
green_sumrate = mean(sum(green_rate, dim=1))

# 2. Constraint: Yellow SINR only
sinr = signal / interference                       # (Batch, K)
yellow_sinr = sinr[:, yellow_mask]                  # (Batch, K_yellow)
sinr_violations = clamp(sinr_min - yellow_sinr, min=0)
sinr_penalty = mean(sum(sinr_violations^2, dim=1))

# 3. Total loss
loss = -green_sumrate + penalty_weight * sinr_penalty
```

### Side-by-side comparison

| Aspect | D2D `constrained_sumrate_loss` | JSAC `jsac_sumrate_loss` |
|---|---|---|
| **Objective links** | ALL K links | GREEN links only |
| **Constraint type** | Rate >= r_min | SINR >= gamma_min |
| **Constrained links** | ALL K links | YELLOW links only |
| **Power constraint** | Implicit (sigmoid → [0, Pmax]) | Per-group normalization (Section 3) |
| **SINR computation** | Same formula | Same formula |
| **Penalty mechanism** | Same (squared violation) | Same (squared violation) |

The internal SINR/rate computation is identical — `signal / interference` uses the same matrix operations. The masks simply **select which links contribute to which term**.



---

## 5. Node Features — Expanded from 3 to 4 Dimensions

### D2D node features: 3 dimensions

```
x = [direct_distance, direct_loss, power_placeholder]
     x[:, 0]          x[:, 1]       x[:, 2]
```

All links are homogeneous — there's no need to distinguish link types.

### JSAC node features: 4 dimensions

```
x = [direct_distance, direct_loss, link_type, power_placeholder]
     x[:, 0]          x[:, 1]       x[:, 2]    x[:, 3]
```

**Changed from the original plan.** The initial assumption was that gradient signal alone would teach the GNN to differentiate Yellow vs Green links. This turned out to be wrong — see the troubleshooting note below.

> **Troubleshooting note — why link_type is necessary:**
>
> The initial implementation kept 3 node features (same as D2D), relying on the loss function's masks to drive differentiation through gradients. Result: the GNN barely beat equal power (~120 vs ~119 Green SR, while WMMSE achieved ~146).
>
> The problem: during backprop, the gradient for a Yellow link's power comes from the penalty term, while the gradient for a Green link's power comes from the rate term. But in the forward pass, the GNN treated them identically — it only saw distance and channel quality. Since Yellow and Green links have similar distance/channel characteristics (both 2–20m from the same Blue car), the GNN could not distinguish them and produced near-equal power allocations for both types.
>
> Adding `link_type` (1=Green, 0=Yellow) as a node feature lets the GNN learn type-specific power allocation strategies in its forward pass, analogous to how WMMSE uses the `alpha` weight vector (1 for Green, small for Yellow). This single change was one of three fixes that brought GNN performance from ~82% to ~95% of WMMSE.

The original reasons for omitting link_type were:

1. ~~"The loss function handles Yellow/Green distinction"~~ — True for gradient computation, but insufficient for the forward pass. The GNN needs to produce different outputs for different link types, which requires type information as input.

2. ~~"The graph structure encodes group membership"~~ — Partially true (edge topology reflects channel structure), but group membership alone doesn't tell the GNN which links are Yellow vs Green within a group.

3. ~~"Adding categorical features is awkward"~~ — In practice, a binary 0/1 feature works perfectly well. The MLP can easily learn to threshold on it.

**Impact on architecture dimensions:** mlp1 input = 4 node + 3 edge = 7 (was 5 in D2D). mlp2 input = 4 self + 32 aggregated = 36 (was 35). Power output is `out[:, 3]` (was `out[:, 2]`).

---

## 6. IGCNet Architecture — Message Passing Reused, Dimensions Changed

The message passing *mechanism* is entirely reusable:

```python
class IGConv(MessagePassing):
    # message():   x_j (neighbor features) + edge_attr → mlp1 → message
    # aggregate(): max over incoming messages
    # update():    [x_i, aggregated_msg] → mlp2 → new node state
```

**Why message passing still works for JSAC:**

1. **Message passing is topology-agnostic.** The IGConv layer doesn't care whether the graph is dense (D2D) or sparse (JSAC). It processes whatever edges exist, including the two edge types (interference + intra-group).

2. **Max aggregation still makes sense.** With two edge types, the MLP can learn to map them into different regions of the 32-dim message space. Max aggregation then selects the most informative signal from each type.

3. **Shared weights across iterations** — still valid. The interference-management problem has the same iterative structure in JSAC.

**What changed in the architecture:**

| Dimension | D2D | JSAC | Reason |
|---|---|---|---|
| Node features | 3 | **4** | Added link_type |
| Edge features | 1 | **3** | Added edge_type + channel_gain |
| MLP1 input | 4 (3+1) | **7** (4+3) | More features |
| MLP2 input | 35 (3+32) | **36** (4+32) | More node features |
| MLP2 hidden | 16 | **32** | More capacity for the harder problem |
| Iterations | 3 | **4** | JSAC problem is more complex; intra-group coordination benefits from an extra hop |
| Static features preserved | x[:, :2] | x[:, :3] | dist, loss, link_type all static |
| Logit column | out[:, 2] | out[:, 3] | Shifted by the added link_type feature |

---

## 7. Training & Evaluation Loop Changes

### Training: `train_epoch`

```python
def train_epoch(model, loader, optimizer, device, K, var_noise, n_blue,
                sinr_min=1.0, penalty_weight=10.0):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # Per-group softmax is applied INSIDE the loss function:
        #   raw_p = out[:, 3]  →  apply_group_softmax(raw_p, data.group_ids, data.batch, n_blue)
        loss = constrained_sumrate_loss_jsac(data, out, K, var_noise, n_blue, sinr_min, penalty_weight)
        loss.backward()
        optimizer.step()
```

**Differences from D2D `train_epoch`:**
- Extra arguments: `n_blue`, `sinr_min`, `penalty_weight`
- `group_ids`, `green_mask`, `yellow_mask` are stored in the PyG Data object (not passed separately)
- Per-group softmax inside loss (replaces independent sigmoid)
- Logit extracted from `out[:, 3]` (was `out[:, 2]`)

### Evaluation: `eval_epoch`

Returns three values: `avg_loss, green_sumrate, violation_rate`. Extracts powers via group softmax, then uses numpy evaluation helpers:

```python
# Extract GNN powers after group softmax
raw_p = out[:, 3]
p = apply_group_softmax(raw_p, data.group_ids, data.batch, n_blue)
all_powers = p.reshape(num_graphs, K).cpu().numpy()

# Numpy evaluation
green_sr = UO.compute_green_sumrate(config, all_powers, directLoss, crossLoss, green_mask)
viol, _  = UO.compute_yellow_sinr_violation(config, all_powers, directLoss, crossLoss, yellow_mask, sinr_min)
```

---

## 8. Full Implementation Map

### 8.1 Files modified

| File | Changes |
|---|---|
| `model_gnn.py` | `build_graph` (interf + intra-group edges, 4D nodes, 3D edges), `proc_data` (new args), `IGCNet` (drop Sigmoid, new dims, 4 iters), `apply_group_softmax`, `constrained_sumrate_loss_jsac`, `train_epoch`, `eval_epoch` |
| `utils_objective.py` | Removed old `sumrate_loss()` (PyTorch); kept numpy eval functions; added `compute_green_sumrate`, `compute_yellow_sinr_violation` |
| `config_system.py` | None — layout generation already done in Step 1 |

### 8.2 Functions: reuse vs new vs changed

| Function | Status | Notes |
|---|---|---|
| `IGConv` | **Reuse** (update preserves 3 static features instead of 2) | Message passing mechanism unchanged |
| `IGCNet` | **Changed** | Drop Sigmoid, MLP1 [7,32,32], MLP2 [36,32,1], 4 iterations |
| `build_graph` | **Rewritten for JSAC** | Two edge types, 4D node feat (added link_type), 3D edge feat (dist, type, channel gain) |
| `proc_data` | **Updated** | Passes interf_mask, group_ids, green_mask, yellow_mask |
| `apply_group_softmax` | **New** | Per-group softmax via `pyg_softmax` with composite group key |
| `constrained_sumrate_loss_jsac` | **New** | Green objective + Yellow penalty (replaces both D2D loss functions) |
| `train_epoch` | **Updated** | Accepts n_blue, sinr_min, penalty_weight; calls JSAC loss |
| `eval_epoch` | **Updated** | Returns green_sumrate + violation_rate via numpy eval |
| `sumrate_loss` (D2D) | **Removed** | No longer needed |

### 8.3 Pseudocode: `jsac_sumrate_loss` (full)

```python
def jsac_sumrate_loss(data, p_frac, K, var_noise, green_mask, yellow_mask,
                      sinr_min, penalty_weight):
    """
    JSAC training loss: maximize Green sum-rate, penalize Yellow SINR violations.

    Args:
        data:           PyG batch (data.y = channel matrix)
        p_frac:         (Batch, K) power fractions from softmax (per-group sum = 1)
        K:              total links
        var_noise:      normalized noise
        green_mask:     (K,) bool tensor
        yellow_mask:    (K,) bool tensor
        sinr_min:       minimum SINR for Yellow (linear scale)
        penalty_weight: weight for violation penalty

    Returns:
        loss: scalar tensor
    """
    p = p_frac.unsqueeze(-1)          # (Batch, K, 1)
    H2 = data.y.permute(0, 2, 1)     # (Batch, K, K)
    rx_power = H2 * p                 # (Batch, K, K)

    eye = torch.eye(K, device=p.device)
    signal    = (rx_power * eye).sum(dim=2)          # (Batch, K)
    interf    = (rx_power * (1 - eye)).sum(dim=2)    # (Batch, K)
    sinr      = signal / (interf + var_noise + 1e-12)

    # --- Green objective: maximize sum-rate ---
    green_rate = torch.log2(1 + sinr[:, green_mask])
    green_sumrate = torch.mean(torch.sum(green_rate, dim=1))

    # --- Yellow constraint: penalize SINR violations ---
    yellow_sinr = sinr[:, yellow_mask]
    violations = torch.clamp(sinr_min - yellow_sinr, min=0.0)
    penalty = torch.mean(torch.sum(violations ** 2, dim=1))

    return -green_sumrate + penalty_weight * penalty
```

---

