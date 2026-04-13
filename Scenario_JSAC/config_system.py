# ==============================================================================
# Filename: config_system.py
# Description: Defines the communication system model, parameters, and 
#              data generation/processing routines.
# Includes:
#   - System parameter initialization
#   - Layout and path loss generation
#   - Channel fading and shadowing
#   - Data normalization
# ==============================================================================

import numpy as np

# =============================================================================
# System Parameters and Initialization
# =============================================================================

class init_parameters_JSAC():
    """
    Configuration container for the JSAC Wireless Network System.
    Blue cars (Tx) each serve Yellow (sensing) and Green (communication) receivers.
    """
    def __init__(self, n_blue=10, n_yellow_per_blue=2, n_green_per_blue=3, field_length=225, noise_density_milli_decibel=-169):
        # JSAC topology
        self.n_blue = n_blue
        self.n_yellow_per_blue = n_yellow_per_blue
        self.n_green_per_blue = n_green_per_blue
        self.n_links_per_blue = n_yellow_per_blue + n_green_per_blue  # links per Blue car
        self.n_links = n_blue * self.n_links_per_blue  # total number of links
        self.field_length = field_length

        # Blue car placement
        self.min_blue_dist = 50  # minimum separation between Blue cars (m)

        # Rx placement around their Blue car
        self.rx_min_radius = 2   # min distance Blue-to-its-Rx (m)
        self.rx_max_radius = 20  # max distance Blue-to-its-Rx (m)

        # Minimum separation between Rx within the same Blue car's cluster (m)
        self.min_rx_separation = 2

        # Physics / Frequency settings
        self.bandwidth = 5e6
        self.carrier_f = 2.4e9
        # self.carrier_f = 1.2e9
        self.tx_height = 1.5
        self.rx_height = 1.5
        self.antenna_gain_decibel = 2.5

        # Power settings (may need to get larger)
        self.tx_power_milli_decibel = 40
        self.tx_power = np.power(10, (self.tx_power_milli_decibel-30)/10)

        # Noise settings
        self.noise_density_milli_decibel = noise_density_milli_decibel
        self.input_noise_power = np.power(10, ((self.noise_density_milli_decibel-30)/10)) * self.bandwidth
        self.output_noise_power = self.input_noise_power

        # SNR settings
        self.SNR_gap_dB = 6
        self.SNR_gap = np.power(10, self.SNR_gap_dB/10)

        # String identifier for saving/loading
        self.setting_str = "JSAC_B{}_My{}_Mg{}_{}X{}".format(
            self.n_blue, self.n_yellow_per_blue, self.n_green_per_blue,
            self.field_length, self.field_length
        )

# =============================================================================
# Layout Generation
# =============================================================================

# JSAC hierarchical layout generation
def layout_generate_jsac(general_para):
    """
    Generate a single JSAC layout with hierarchical placement.
    1. Place B Blue cars (Tx) with >= min_blue_dist separation.
    2. For each Blue car, place M_y Yellow + M_g Green Rx within [rx_min, rx_max] radius,
       enforcing min_rx_separation between Rx in the same cluster.
    3. Build layout [K, 4] and distance matrix [K, K].

    Returns:
        layout:     (K, 4) — [tx_x, tx_y, rx_x, rx_y] per link
        distances:  (K, K) — dist[i,j] = distance from Tx_j to Rx_i
        group_ids:  (K,) int — Blue car index for each link
        link_types: (K,) int — 0 = Yellow, 1 = Green
    """
    B = general_para.n_blue
    M_y = general_para.n_yellow_per_blue
    M_g = general_para.n_green_per_blue
    M = M_y + M_g  # links per Blue car
    K = general_para.n_links
    field = general_para.field_length
    min_blue = general_para.min_blue_dist
    r_min = general_para.rx_min_radius
    r_max = general_para.rx_max_radius
    min_sep = general_para.min_rx_separation

    # --- Step 1: Place B Blue cars with >= min_blue_dist separation ---
    blue_xs, blue_ys = np.zeros(B), np.zeros(B)
    for b in range(B):
        attempts = 0
        while True:
            cx = np.random.uniform(0, field)
            cy = np.random.uniform(0, field)
            # Check separation against already-placed Blue cars
            if b > 0:
                dists_to_placed = np.hypot(blue_xs[:b] - cx, blue_ys[:b] - cy)
                if np.min(dists_to_placed) < min_blue:
                    attempts += 1
                    if attempts > 10000:
                        return layout_generate_jsac(general_para)  # restart
                    continue
            blue_xs[b], blue_ys[b] = cx, cy
            break

    # --- Step 2: Place Rx around each Blue car ---
    tx_xs, tx_ys = np.zeros(K), np.zeros(K)
    rx_xs, rx_ys = np.zeros(K), np.zeros(K)
    group_ids = np.zeros(K, dtype=int)
    link_types = np.zeros(K, dtype=int)

    for b in range(B):
        bx, by = blue_xs[b], blue_ys[b]
        base = b * M  # starting index for this Blue car's links
        cluster_rx_xs, cluster_rx_ys = [], []

        for m in range(M):
            idx = base + m
            tx_xs[idx], tx_ys[idx] = bx, by
            group_ids[idx] = b
            link_types[idx] = 0 if m < M_y else 1  # first M_y are Yellow, rest Green

            attempts = 0
            while True:
                dist = np.random.uniform(r_min, r_max)
                angle = np.random.uniform(0, 2 * np.pi)
                rx_x = bx + dist * np.cos(angle)
                rx_y = by + dist * np.sin(angle)

                # Check field boundaries
                if not (0 <= rx_x <= field and 0 <= rx_y <= field):
                    attempts += 1
                    if attempts > 10000:
                        return layout_generate_jsac(general_para)
                    continue

                # Check min separation against already-placed Rx in this cluster
                if len(cluster_rx_xs) > 0:
                    dists_to_placed = np.hypot(
                        np.array(cluster_rx_xs) - rx_x,
                        np.array(cluster_rx_ys) - rx_y
                    )
                    if np.min(dists_to_placed) < min_sep:
                        attempts += 1
                        if attempts > 10000:
                            return layout_generate_jsac(general_para)
                        continue

                rx_xs[idx], rx_ys[idx] = rx_x, rx_y
                cluster_rx_xs.append(rx_x)
                cluster_rx_ys.append(rx_y)
                break

    # --- Step 3: Build layout and distance matrix ---
    layout = np.column_stack((tx_xs, tx_ys, rx_xs, rx_ys))  # (K, 4)

    # distances[i, j] = distance from Tx_j to Rx_i
    tx_coords = layout[:, :2]  # (K, 2)
    rx_coords = layout[:, 2:]  # (K, 2)
    dx = rx_coords[:, 0:1] - tx_coords[:, 0:1].T  # (K, K)
    dy = rx_coords[:, 1:2] - tx_coords[:, 1:2].T  # (K, K)
    distances = np.sqrt(dx**2 + dy**2)

    return layout, distances, group_ids, link_types


def generate_layouts_jsac(general_para, number_of_layouts):
    """
    Generate multiple JSAC layouts.
    Returns:
        layouts:    (N_layouts, K, 4)
        dists:      (N_layouts, K, K)
        group_ids:  (K,) int — same for all layouts (topology is fixed)
        link_types: (K,) int — same for all layouts
    """
    K = general_para.n_links
    print("<<<<<<<<<<<<<{} layouts: {}>>>>>>>>>>>>".format(
        number_of_layouts, general_para.setting_str))

    layouts, dists_list = [], []
    group_ids, link_types = None, None

    for _ in range(number_of_layouts):
        layout, dist, gids, ltypes = layout_generate_jsac(general_para)
        layouts.append(layout)
        dists_list.append(dist)
        if group_ids is None:
            group_ids, link_types = gids, ltypes

    layouts = np.array(layouts)
    dists = np.array(dists_list)

    assert layouts.shape == (number_of_layouts, K, 4)
    assert dists.shape == (number_of_layouts, K, K)

    return layouts, dists, group_ids, link_types


# =============================================================================
# Channel: Path Loss, Fading, Shadowing
# =============================================================================

# compute path loss components of channel path_losses
# should be used with multiple layouts:
#        distances shape: number of layouts X N X N
def compute_path_losses(general_para, distances):
    """
    Computes deterministic path loss based on physics parameters.
    """
    N = np.shape(distances)[-1]
    assert N==general_para.n_links

    h1 = general_para.tx_height
    h2 = general_para.rx_height
    signal_lambda = 2.998e8 / general_para.carrier_f
    antenna_gain_decibel = general_para.antenna_gain_decibel

    # Compute breakpoint distance
    # compute relevant quantity
    Rbp = 4 * h1 * h2 / signal_lambda
    Lbp = abs(20 * np.log10(np.power(signal_lambda, 2) / (8 * np.pi * h1 * h2)))

    # Compute path loss
    # compute coefficient matrix for each Tx/Rx pair
    sum_term = 20 * np.log10(distances / Rbp)
    Tx_over_Rx = Lbp + 6 + sum_term + ((distances > Rbp).astype(int)) * sum_term  # adjust for longer path loss

    # Add antenna gain only for direct links (direct channel, diagonal elements) if needed
    # but original code adds it to diagonal via np.eye(N)
    pathlosses = -Tx_over_Rx + np.eye(N) * antenna_gain_decibel
    pathlosses = np.power(10, (pathlosses / 10))  # convert from decibel to absolute

    return pathlosses

# Add in shadowing into channel losses
def add_shadowing(channel_losses):
    """Adds Log-normal shadowing."""
    shadow_coefficients = np.random.normal(loc=0, scale=8, size=np.shape(channel_losses))
    channel_losses = channel_losses * np.power(10.0, shadow_coefficients / 10)
    return channel_losses

# Add in fast fading into channel losses
def add_fast_fading(channel_losses):
    """Adds Rayleigh fast fading."""
    fastfadings = (np.power(np.random.normal(loc=0, scale=1, size=np.shape(channel_losses)), 2) +
                   np.power(np.random.normal(loc=0, scale=1, size=np.shape(channel_losses)), 2)) / 2
    channel_losses = channel_losses * fastfadings
    return channel_losses

# =============================================================================
# Combined Channel Pipeline (JSAC)
# =============================================================================

def compute_channel_losses_jsac(general_para, distances, interf_mask):
    """
    Full channel pipeline: path loss → shadowing → fading → keep only interfering entries.

    Each Blue car uses M orthogonal channels. Only links sharing the same channel
    (in different Blue cars) interfere. The interf_mask encodes this.

    Args:
        general_para: init_parameters_JSAC instance
        distances:    (N_layouts, K, K) distance matrices
        interf_mask:  (K, K) binary — 1 where two links interfere (same channel, different group)

    Returns:
        path_losses:    (N_layouts, K, K) deterministic path loss only
        channel_losses: (N_layouts, K, K) full channel; only diagonal + interfering entries kept
    """
    K = general_para.n_links

    # 1. Deterministic path loss
    path_losses = compute_path_losses(general_para, distances)

    # 2. Add shadowing + fast fading
    channel_losses = add_shadowing(np.copy(path_losses))
    channel_losses = add_fast_fading(channel_losses)

    # 3. Keep only diagonal (direct links) and interfering off-diagonal entries
    keep_mask = np.eye(K) + interf_mask  # 1 on diagonal + interfering pairs
    channel_losses = channel_losses * keep_mask

    return path_losses, channel_losses


# =============================================================================
# Process Training Losses
# =============================================================================

# Legacy?
def proc_train_losses(train_path_losses, train_channel_losses):
    """
    Combines path loss (for interference/cross links) and fading (for direct links)
    if specific training logic requires it. 
    Note: Based on original code logic where diagonal uses fading but off-diagonal uses path loss.
    """
    # Combine path losses and channel losses
    L, K, _ = train_path_losses.shape
    mask = np.eye(K)

    # Off-diagonal: Use path losses only (cross-links)
    diag_path = np.multiply(mask, train_path_losses)
    off_diag_path = train_path_losses - diag_path   # off-diagonal elements only

    # Diagonal: Use channel losses (direct-links, Path Loss + Fading + Shadowing)
    diag_channel = np.multiply(mask, train_channel_losses)
    
    # Combine
    train_losses = diag_channel + off_diag_path
    
    return train_losses


# =============================================================================
# JSAC Metadata Helpers
# =============================================================================

def jsac_metadata(group_ids, link_types, M, shuffle_channels=False):
    """
    Build masks from JSAC layout metadata.

    Each Blue car has M orthogonal channels, one per Rx slot. Links in different
    Blue cars interfere only if they share the same channel (same intra-group slot).

    When shuffle_channels=True, each Blue car gets a random permutation of
    channel indices. This mixes Yellow and Green links onto the same channels
    across different Blue cars, creating Yellow-Green cross-interference that
    gives WMMSE a meaningful optimization lever. Without shuffling, Yellow and
    Green occupy disjoint channel sets and never interfere with each other,
    making the problem nearly trivial (equal power is near-optimal).

    Args:
        group_ids:  (K,) int array — which Blue car each link belongs to (0..B-1)
        link_types: (K,) int array — 0 = Yellow (sensing), 1 = Green (communication)
        M:          int — number of links per Blue car (M_y + M_g), = number of channels
        shuffle_channels: bool — if True, randomize channel assignment per Blue car

    Returns:
        channel_ids: (K,) int — orthogonal channel index for each link (0..M-1)
        interf_mask: (K, K) float — 1 where two links interfere (same channel, different group)
        green_mask:  (K,) bool  — True for Green links
        yellow_mask: (K,) bool  — True for Yellow links
    """
    group_ids = np.asarray(group_ids)
    link_types = np.asarray(link_types)
    K = len(group_ids)
    B = int(np.max(group_ids)) + 1

    # Channel assignment
    channel_ids = np.arange(K) % M

    if shuffle_channels:
        # Each Blue car gets a random permutation of channel indices.
        # This ensures Yellow/Green from different Blue cars can share a channel.
        for b in range(B):
            links = np.where(group_ids == b)[0]
            channel_ids[links] = np.random.permutation(M)

    # Two links interfere iff same channel AND different Blue car
    same_channel = (channel_ids[:, None] == channel_ids[None, :])
    diff_group = (group_ids[:, None] != group_ids[None, :])
    interf_mask = (same_channel & diff_group).astype(np.float64)

    green_mask = (link_types == 1)
    yellow_mask = (link_types == 0)

    return channel_ids, interf_mask, green_mask, yellow_mask