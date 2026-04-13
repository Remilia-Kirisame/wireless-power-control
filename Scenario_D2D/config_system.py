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

# Initialization of system parameters
class init_parameters():
    """
    Configuration container for the Wireless Network System.
    Includes physical layer parameters and simulation settings.
    """
    def __init__(self, n_links=50, field_length=1000):
        # wireless network settings
        self.n_links = n_links
        self.field_length = field_length

        # Link length settings
        self.shortest_directLink_length = 1
        self.longest_directLink_length = 20
        self.shortest_crossLink_length = 30

        # Physics / Frequency settings
        self.bandwidth = 5e6
        self.carrier_f = 2.4e9
        self.tx_height = 1.5
        self.rx_height = 1.5
        self.antenna_gain_decibel = 2.5

        # Power settings
        self.tx_power_milli_decibel = 40
        self.tx_power = np.power(10, (self.tx_power_milli_decibel-30)/10)
        
        # Noise settings
        self.noise_density_milli_decibel = -169
        self.input_noise_power = np.power(10, ((self.noise_density_milli_decibel-30)/10)) * self.bandwidth
        self.output_noise_power = self.input_noise_power
        
        # SNR settings
        self.SNR_gap_dB = 6
        self.SNR_gap = np.power(10, self.SNR_gap_dB/10)

        # String identifier for saving/loading
        self.setting_str = "{}_links_{}X{}_{}_{}_length".format(
            self.n_links, self.field_length, self.field_length, self.shortest_directLink_length, self.longest_directLink_length
        )

        # 2D occupancy grid setting
        self.cell_length = 5
        self.n_grids = np.round(self.field_length/self.cell_length).astype(int)

# =============================================================================
# Layout and Path Loss Generation
# =============================================================================

# To generate multiple layouts at once
def generate_layouts(general_para, number_of_layouts):
    """
    Generates N layouts based on the parameters.
    Args:
        general_para: Configuration object with system parameters. (see init_parameters)
        number_of_layouts: Number of layouts to generate.
    Returns:
        layouts: (number_of_layouts, n_links, 4) -> [tx_x, tx_y, rx_x, rx_y]
        dists:   (number_of_layouts, n_links, n_links) -> Distance matrix
    """
    N = general_para.n_links
    print("<<<<<<<<<<<<<{} layouts: {}>>>>>>>>>>>>".format(
        number_of_layouts, general_para.setting_str))
    
    layouts = []
    dists = []

    for i in range(number_of_layouts):
        layout, dist = layout_generate_sequential(general_para)
        layouts.append(layout)
        dists.append(dist)

    layouts = np.array(layouts)
    dists = np.array(dists)

    assert np.shape(layouts)==(number_of_layouts, N, 4)
    assert np.shape(dists)==(number_of_layouts, N, N)

    return layouts, dists

"""
For better QoS constraint handling.
    To decrease the minimum data rate violation, we can ensure that no cross-link is too close (i.e., below shortest_crossLink_length) during layout generation.
    (Global Rejction Sampling method takes much longer time to generate valid layouts.)
"""
# Generate layout one at a time, with sequential checking of validity to ensure no cross-links are too close.
# This is a **Sequential Rejection Sampling** method.
def layout_generate_sequential(general_para):
    N = general_para.n_links
    field_length = general_para.field_length
    short_direct = general_para.shortest_directLink_length
    long_direct = general_para.longest_directLink_length
    short_cross = general_para.shortest_crossLink_length

    tx_xs, tx_ys = np.zeros(N), np.zeros(N)
    rx_xs, rx_ys = np.zeros(N), np.zeros(N)

    for i in range(N):
        valid_pair = False
        attempts = 0
        
        while not valid_pair:
            # 1. Propose a new Tx
            cand_tx_x = np.random.uniform(low=0, high=field_length)
            cand_tx_y = np.random.uniform(low=0, high=field_length)
            
            # 2. Propose a corresponding Rx
            u = np.random.uniform(0, 1) # Generate a random fraction between 0 and 1
            skewed_fraction = u ** 2 # Square the fraction to skew it towards 0.
            pair_dist = short_direct + (long_direct - short_direct) * skewed_fraction
            # pair_dist = np.random.uniform(low=short_direct, high=long_direct)
            pair_angle = np.random.uniform(low=0, high=np.pi * 2)
            cand_rx_x = cand_tx_x + pair_dist * np.cos(pair_angle)
            cand_rx_y = cand_tx_y + pair_dist * np.sin(pair_angle)

            # Check field boundaries
            if not (0 <= cand_rx_x <= field_length and 0 <= cand_rx_y <= field_length):
                continue

            # 3. Check cross-link distances ONLY against already placed pairs (0 to i-1)
            conflict = False
            for j in range(i):
                # Check Tx_i interfering with Rx_j
                dist_tx_i_rx_j = np.hypot(cand_tx_x - rx_xs[j], cand_tx_y - rx_ys[j])
                # Check Tx_j interfering with Rx_i
                dist_tx_j_rx_i = np.hypot(tx_xs[j] - cand_rx_x, tx_ys[j] - cand_rx_y)
                
                if dist_tx_i_rx_j < short_cross or dist_tx_j_rx_i < short_cross:
                    conflict = True
                    break # Break early, try a new candidate for pair i

            if not conflict:
                tx_xs[i], tx_ys[i] = cand_tx_x, cand_tx_y
                rx_xs[i], rx_ys[i] = cand_rx_x, cand_rx_y
                valid_pair = True
            
            attempts += 1
            if attempts > 10000:
                # Failsafe: if the field is physically too packed, restart the whole layout
                print("Too many attempts for pair {}, restarting layout generation...".format(i))
                return layout_generate_sequential(general_para) 

    # 4. Format output to match original code structure
    layout = np.column_stack((tx_xs, tx_ys, rx_xs, rx_ys))
    distances = np.zeros([N, N])
    
    for rx_index in range(N):
        for tx_index in range(N):
            tx_coor = layout[tx_index][0:2]
            rx_coor = layout[rx_index][2:4]
            distances[rx_index][tx_index] = np.linalg.norm(tx_coor - rx_coor)

    return layout, distances

# Generate layout one at a time
# This is a **Global Rejection Sampling** method.
#     - We generate the entire layout (all Tx/Rx pairs) and check validity (shortest_crossLink_length) at the end.
#     - If invalid, we reject the entire layout and generate a new one.
def layout_generate(general_para):
    """
    Internal helper to generate a single layout.
    """
    N = general_para.n_links

    # 1. Generate transmitters
    # first, generate transmitters' coordinates
    tx_xs = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
    tx_ys = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])

    while(True): # loop until a valid layout generated
        # generate rx one by one rather than N together to ensure checking validity one by one
        rx_xs = []
        rx_ys = []
        # 2. Generate receivers
        for i in range(N):
            got_valid_rx = False
            while(not got_valid_rx):
                pair_dist = np.random.uniform(low=general_para.shortest_directLink_length, 
                                              high=general_para.longest_directLink_length)
                pair_angles = np.random.uniform(low=0, high=np.pi*2)
                rx_x = tx_xs[i] + pair_dist * np.cos(pair_angles)
                rx_y = tx_ys[i] + pair_dist * np.sin(pair_angles)

                if(0<=rx_x<=general_para.field_length and 0<=rx_y<=general_para.field_length):
                    got_valid_rx = True
            rx_xs.append(rx_x)
            rx_ys.append(rx_y)

        # For now, assuming equal weights and equal power, so not generating them
        layout = np.concatenate((tx_xs, tx_ys, rx_xs, rx_ys), axis=1)
        distances = np.zeros([N, N])

        # 3. Compute distances (Hij is from j-th transmitter to i-th receiver)
        # compute distance between every possible Tx/Rx pair
        for rx_index in range(N):
            for tx_index in range(N):
                tx_coor = layout[tx_index][0:2]
                rx_coor = layout[rx_index][2:4]
                # according to paper notation convention, Hij is from jth transmitter to ith receiver
                distances[rx_index][tx_index] = np.linalg.norm(tx_coor - rx_coor)

        # 4. Check validity: whether any cross-link is too close
        # Check whether a tx-rx link (potentially cross-link) is too close
        if(np.min(distances) > general_para.shortest_crossLink_length):
            break

    return layout, distances

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
# Process Training Losses
# =============================================================================

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
# Data Normalization
# =============================================================================

# Legacy, now replaced by WirelessScaler class in main_supporter.py
def normalize_data(train_data, test_data):
    """
    Normalizes inputs (distances or channel gains) for Neural Network stability.
    """
    # ---- Normalize Train Data ----
    Ltr, Ktr, _ = train_data.shape
    mask_tr = np.eye(Ktr)

    train_copy = np.copy(train_data)

    # Normalize Diagonal (Direct Links)
    diag_H = np.multiply(mask_tr, train_copy)
    diag_mean = np.sum(diag_H) / (Ltr * Ktr)    # mean
    diag_var  = np.sqrt(np.sum(np.square(diag_H)) / (Ltr * Ktr))    # std (standard deviation)
    tmp_diag  = (diag_H - diag_mean) / (diag_var + 1e-12)   # normalize

    # Normalize Off-Diagonal (Interence Links)
    off_diag   = train_copy - diag_H
    off_mean   = np.sum(off_diag) / (Ltr * Ktr * (Ktr - 1))
    off_var    = np.sqrt(np.sum(np.square(off_diag)) / (Ltr * Ktr * (Ktr - 1)))
    tmp_off    = (off_diag - off_mean) / (off_var + 1e-12)  # non-diagonal mean and std
    tmp_off_dg = tmp_off - np.multiply(tmp_off, mask_tr)

    norm_train = np.multiply(tmp_diag, mask_tr) + tmp_off_dg    # diagonal from normalized direct links tmp_diag;
                    # off-diagonal from normalized interference tmp_off_dg
    # ---- Normalize Test Data (using Train stats) ----
    Lte, Kte, _ = test_data.shape
    mask_te = np.eye(Kte)
    test_copy = np.copy(test_data)

    # Apply Train mean/var to Test Diagonal
    diag_H = np.multiply(mask_te, test_copy)
    tmp_diag = (diag_H - diag_mean) / (diag_var + 1e-12)

    # Apply Train mean/var to Test Off-Diagonal
    off_diag   = test_copy - diag_H
    tmp_off    = (off_diag - off_mean) / (off_var + 1e-12)
    tmp_off_dg = tmp_off - np.multiply(tmp_off, mask_te)

    norm_test  = np.multiply(tmp_diag, mask_te) + tmp_off_dg    # same as before, test data processing

    return norm_train, norm_test