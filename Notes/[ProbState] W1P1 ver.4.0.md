

[TOC]



# Problem Statement and Connection to Codes

---

[TOC]

修订版：

The project W1P1, titled "Learning to optimize: Training deep neural networks for wireless resource management", proposes using a Deep Neural Network (DNN) approach (specifically a Multi-Layer Perceptron or MLP) to approximate the output of the classical **Weighted Minimum Mean Square Error (WMMSE)** algorithm. The primary motivation is to drastically reduce the online computational time required for power allocation in interference-limited wireless networks, compared to the iterative WMMSE algorithm.

## Specific Problem Statement of W1P1

The problem is structured around the conventional non-convex optimization challenge inherent in wireless resource allocation and the subsequent supervised machine learning task designed to solve it efficiently.

> 综述：模拟阶段有两个channel model
>
> **Model 1: Gaussian IC** - Interference Channel
> 	Each channel coefficient is generated according to a standard normal distribution, i.e., Rayleigh fading distribution with zero mean and unit vari-
> ance.
> 	$K\in\{10, 20, 30\}$
>
> **Model 2: Practical IMAC (interfering multiple-access channel)** - multi-cell interfering
>
> <img src="IMAC.png" style="zoom:30%;" />
>
> - $N$ cells and $K$ users.
> - Distance between centers of adjacent cells is set to be $200$ meters.
> - In each cell, one BS (base station) is placed at the center of the cell.
> - Users are randomly and uniformly distributed.
>   - Assume: same number of users in each cell.
> - The channel between each user and each BS is randomly generated according to a Rayleigh fading distribution with zero mean and variance $(200/d)^3L$, where
>   - $d$ denotes the distance between the BS and user, and the
>   - quantity $L$ represents the shadow fading following a log-normal distribution with zero mean and variance $64$.
> - $(N, K) \in \{(3,16), (3,24), (3,60), (7,28)\}$
> - **Uplink Signal**: consider signals transmitted from multiple users/mobile stations to one base station.
> - **同频组网 (Universal Frequency Reuse / Frequency Reuse Factor = 1)**: All cells and users works on the same frequency 
>
> - **Radiation Pattern**：Omnidirectional Broadcast
> - Cell & User Generation: Per-Cell Generation. (Determine the centre of N BSs of each cell, then randomly independently distribute users, same #users for each cell).

### I. Communication System Model & Parameters

1.1 **Network Architecture:** IC and IMAC

1.2 **Antenna Config** The network consists of $K$ independent transceiver pairs, each equipped with a **single antenna**.

1.3 **Channel Model:** The wireless channel state is defined by the complex channel coefficients $h_{kj}$ (direct link $k \to k$, interference link $j \to k$). 

- Gaussian IC: **Rayleigh fading** (standard normal distribution with zero mean and unit variance).
- IMAC: incorporates Rayleigh fading, path loss (distance-dependent, $d^3$), and shadow fading (log-normal distribution with variance 64)

1.4 **Interference:** Multiuser interference is **treated as additive noise (TIN)** at the receiver.

*   **Noise Power ($\sigma^2$):** The additive noise variance $\sigma_k^2$ is assumed to be Gaussian. In the primary code implementation (`wf.generate_Gaussian`), the noise variance (`var_noise`) is universally set to **1**.

### II. Optimization Objective and Constraints

2.1 **Objective Function (Primal Problem):** To find the optimal transmit power vector $\mathbf{p} = [p_1, \dots, p_K]^T$ that maximizes the **Weighted Sum-Rate (WSR)**. 
$$
\max_{\mathbf{p}}  \sum_{k=1}^{K} \alpha_k \log_2 \left( 1 + \frac{|h_{kk}|^2 p_k}{\sum_{j \neq k} |h_{kj}|^2 p_j + \sigma_k^2} \right)
$$

> **Average Weighted Sum-Rate (WSR)**. 
> $$
> \max_{\mathbf{p}} \frac{1}{K}  \sum_{k=1}^{K} \alpha_k \log_2 \left( 1 + \frac{|h_{kk}|^2 p_k}{\sum_{j \neq k} |h_{kj}|^2 p_j + \sigma_k^2} \right)
> $$
>
> > SR vs WSR vs ASR/AWSR

> This problem is non-convex and is known to be **NP-hard**.

2.2 **Power Constraint:** The transmission power $p_k$ for each user $k$ must be non-negative and bounded by a maximum capacity $P_{max}$.
$$
0 \le p_k \le P_{max}, \quad \forall k \in \{1, 2, \dots, K\}
$$
2.3 **Maximum Power Value:** In the simulation setup (Model 1, Gaussian IC), the maximum power (`Pmax`) is set to **1**.

> TODO:
>
> 2.4 Rate Cpmstraint:
>
> Each user has a minimum rate:
> $$
> r_k \geq r_{\min}
> $$



### III. Opt Baseline (Ground Truth)  (For Supervised DNN)

3.1 **Algorithm:** 
	The iterative **WMMSE algorithm** is executed to generate the "near-optimal" (a local minimiser) power vectors $\mathbf{p}_{\text{WMMSE}}$ that serve as the target output for the DNN.

3.2 **Function:**
	WMMSE converts the non-convex WSR maximization problem into an equivalent weighted Mean-Squared Error (MSE) minimization problem, which can be solved iteratively to find a **high-quality local optimum**

3.3 **Computational Bottleneck:** 
	WMMSE is iterative and computationally expensive, posing challenges for real-time operation in fast-changing wireless channels

> 注：有监督学习，训练集每次都要跑WMMSE作为“label”。后面比较运行速度时，不需要计算训练时间，只看验证集跑的时间。

### IV. Machine Learning Task (Approximation)

4.1 **Approach:** A **Supervised Learning** paradigm ("Learning to optimize") is used, treating the WMMSE output as the ground truth (label).

4.2 **Model Architecture:** A standard, fully connected **DNN**, often referred to as a Multi-Layer Perceptron (MLP), is employed, typically consisting of three hidden layers (e.g., 200, 80, 80 neurons).

4.3 **Input Data:** The input to the DNN is the vector containing the flattened magnitudes of all channel coefficients $\{|h_{kj}|\}$ (size $K^2 \times 1$).

4.4 **Output Data:** The output is the predicted power allocation vector $\hat{\mathbf{p}}$ (size $K \times 1$), approximated to minimize the difference from $\mathbf{p}_{\text{WMMSE}}$.

4.5 **Learning Objective (Loss Function):** The DNN is trained to minimize the **Mean Square Error (MSE)** between the predicted power vector ($\hat{\mathbf{p}}$) and the WMMSE-derived target power vector ($\mathbf{p}_{\text{WMMSE}}$).
$$
\min_{\Theta} \mathbb{E}_\bold{h} \left[\norm{\text{DNN}(\bold{h},\Theta) - \mathbf{p}_{\text{WMMSE}}(\bold{h})}^2\right]
$$
4.6 **Performance Goal:** Achieve a sum-rate performance comparable to WMMSE (near-optimal) while achieving orders of magnitude **speedup** in computational time during inference/testing.

***



[TOC]

## QoS constraint

> *# Version 2.1 - Save and Load features added for GNN and DNN models, as well as scalers and final results.*
>
> *# Version 2.2 - Added WMMSE/GNN with QoS constraints as an additional baseline in test_QoS.py.*
>
> *# Version 2.3 - Layout Generation rewrite. (Sequential Rejection Sampling, config_system) (For better QoS violation)*

We now applied a minimum rate constraint for each user (Transceiver pair) to secure the service. Tested the wmmse, gnn methods in test_QoS.py



## JSAC (Joint Sensing And Communication)

Ref: paper W4P2

### D2D vs JSAC Comparison Scenario-Level Differences

| Dimension                  | D2D (Framework_ver_2.3)                                      | JSAC (Framework_ver_4.0)                                     |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Network topology**       | K independent Tx-Rx pairs, all distinct                      | B Blue cars (Tx), each serving M_y Yellow + M_g Green Rx; K = B×(M_y+M_g) |
| **Tx co-location**         | All Tx at unique positions                                   | M_y+M_g links share the same physical Tx (same Blue car)     |
| **Rx types**               | One type (all communication)                                 | Two types: Yellow (sensing target), Green (communication target) |
| **Interference structure** | All K links mutually interfere (K² channel matrix, all non-zero) | M orthogonal channels per Blue car; only links in **different** groups on the **same channel** interfere → sparse channel matrix |
| **Objective**              | Max sum-rate over **all K links**                            | Max sum-rate over **Green links only**                       |
| **Constraint**             | Min rate per link (QoS, v2.3)                                | Min SINR per **Yellow** (sensing) link                       |
| **Layout structure**       | Uniform random scatter across field                          | Hierarchical: Blue cars spread (≥50m), Rx clustered around their Blue car (2–20m, ≥2m apart within cluster) |
| **Tx-Rx distance**         | ~50–250m (D2D range)                                         | ~2–20m (short-range vehicular)                               |
| **Power variable**         | One power value per link, 0 ≤ p_k ≤ Pmax                     | Per-link power under shared budget per Blue car: $\sum_{k \in \text{Blue}_b} p_k ≤ P_{\max}$ |
| **Path loss model**        | Single ITU-R model for all links                             | Currently single ITU-R for all links (MVP). Triple model (sensing d⁴+RCS, direct d², cross-Blue d² with NLOS shadowing) scheduled in FUTURE_WORKPLAN §1 |
| **Scaler categories**      | 2 (diagonal / off-diagonal)                                  | 3 (direct-sensing / direct-communication / interference)     |
| **GNN graph edges**        | Distance-pruned (300m threshold)                             | Interference edges (`interf_mask`) + intra-group edges (for within-Blue coordination) |
| **GNN output head**        | Sigmoid per-link                                             | Per-group Softmax (enforces ∑ p = Pmax per Blue car)         |
| **Training loss**          | `sumrate_loss` / `constrained_sumrate_loss` (rate-based QoS) | `constrained_sumrate_loss_jsac` (Green rate + Yellow SINR hinge) |
