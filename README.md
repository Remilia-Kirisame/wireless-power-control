# Wireless Power Control for Interference Channels



## Project Overview

Capstone project: ML-based wireless power allocation for interference channels. The core objective is to approximate the iterative WMMSE algorithm with neural networks (specifically Graph Neural Networks and Deep Neural Networks) to achieve near-optimal sum-rate performance at a fraction of the computational time during inference.

The work proceeds in two scenarios that together form a single narrative:

- **Scenario_D2D — Foundation.** A Device-to-Device / interference-channel setup where we built and compared **both a DNN (MLP) and a GNN (IGCNet)** against the iterative WMMSE baseline. A central finding: **the DNN's approximation quality degrades as the problem size K grows**, while the GNN's stays close to WMMSE — because the GNN's weight-sharing and node-count-agnostic message passing scale naturally with K, whereas the MLP's fixed input/output dimensionality does not. This is what motivated the move to GNN-only for the next scenario. A later extension adds a minimum-rate QoS constraint per user (`test_QoS.py`).
- **Scenario_JSAC — Application.** The GNN approach is applied to a richer **Joint Sensing and Communication** setting over vehicular links: Blue-car transmitters serve Yellow (sensing) and Green (communication) receivers on orthogonal channels inside each cluster, with same-channel interference across clusters. The problem adds a **hard per-Blue-car power budget** (enforced by construction via per-group softmax) and a **soft Yellow-SINR constraint** (squared-hinge penalty). The GNN is trained unsupervised against the sum-rate loss and compared to WMMSE and a naive equal-power baseline.

Both scenarios are first-class deliverables of the capstone. D2D produced the scaling result that justifies the method; JSAC demonstrates that the same architecture handles a constrained, structured problem.

> [!note]
>
> For the static capstone showcase website and local viewing instructions, see [README_WEB.md](README_WEB.md).

---



## Problem Statement

The project tackles the non-convex, NP-hard optimization challenge inherent in wireless resource allocation across two distinct scenarios:

### 1. Scenario_D2D: Device-to-Device Communications
- **Network Scenario:** A network consisting of $K$ independent transceiver pairs (users), each equipped with a single antenna. All $K$ links mutually interfere with each other (Gaussian Interference Channel / Interfering Multiple-Access Channel).
- **Objective:** Maximize the Weighted Sum-Rate (WSR) or Average Sum-Rate across all $K$ links.
- **Constraints:** A non-negative maximum transmit power constraint ($P_{max}$) for each link. Later versions introduced a Minimum Rate Quality of Service (QoS) constraint to secure service for each user.
- **Approach:** Both Deep Neural Networks (DNN/MLP) and Graph Neural Networks (GNN) were trained in a supervised or unsupervised manner to approximate the power allocation output of the iterative WMMSE algorithm.

### 2. Scenario_JSAC: Joint Sensing And Communication
- **Network Scenario:** 
  - **Blue cars (Service Provider Vehicle / Tx):** Base transmitters that serve multiple receivers. 
  - **Yellow cars (Sensing Target Vehicle / Rx):** Receivers that require a minimum SINR for the sensing service.
  - **Green cars (Communication Target Vehicle / Rx):** Receivers targeted for data rate maximization.

  Each Blue car operates as a cluster head, serving multiple Yellow and Green cars using orthogonal channels internally. Links from *different* Blue cars utilizing the *same* orthogonal channel interfere.
- **Objective:** Maximize the sum-rate of all communication links (Blue $\to$ Green).
- **Constraints:** 
  1. Maintain a strict minimum SINR requirement for the sensing service links (Blue $\to$ Yellow).
  2. The total transmit power allocated across all links originating from a single Blue car cannot exceed its maximum capacity ($P_{max}$).
- **Approach:** An unsupervised **Graph Neural Network (IGCNet)** trained to output a per-group softmax. This enforces the Blue car power budget natively. The custom loss function balances Green sum-rate maximization with a squared-hinge penalty for Yellow SINR violations.

---



## Modules Overview

The codebase is structured similarly across both scenarios. Key modules include:

- **`config_system.py`**: Manages the physical layer setup, system constants, and layout generation. 
  - *D2D*: Generates a uniform random scatter of transceiver pairs.
  - *JSAC*: Handles hierarchical sampling (Blue cars spread out, Yellow/Green cars clustered around them). 
  - Both compute path loss, shadowing, and Rayleigh fading to construct the channel matrices.
- **`model_gnn.py`**: Contains the IGCNet architecture.
  - *D2D*: Uses distance-pruned edges and a Sigmoid output head per-link.
  - *JSAC*: Utilizes two distinct edge types (interference edges and intra-group edges) and a per-group Softmax output to coordinate power within a group budget. Includes the constrained sum-rate loss function.
- **`model_dnn.py` (*Scenario_D2D only*)**: A standard Multi-Layer Perceptron (MLP) trained via supervised learning to mimic the WMMSE output.
- **`baselines.py`**: Implements the algorithmic baselines for comparison, including iterative `WMMSE` solvers.
  - *JSAC*: Adds a Lagrange multiplier for the SINR constraint and a naive equal-power baseline.
- **`utils_objective.py`**: Provides NumPy-based utility functions for computing communication rates, SINRs, Green sum-rate (JSAC), and QoS/SINR violations.
- **`main_supporter.py`**: Handles training and evaluation dataset generation. Contains the `WirelessScaler` to normalize channel data to ensure stable neural network training.
- **`main.py`**: The primary execution pipeline. Generates datasets, trains the models, evaluates baselines across parameter sweeps, and visualizes results.
- **`test_JSAC.py` / `test_QoS.py`**: Conducts a deep-dive evaluation on a single scenario layout, generating comprehensive plots (CDFs, radar charts, power distribution shares, layout snapshots, etc.).

---



## Requirements

The codebase was developed with Python `3.11.14`. The following core packages are required:

```txt
torch==2.8.0
torch-geometric==2.6.1 
numpy==1.26.4 
matplotlib==3.8.4
```

You can install these dependencies via pip:

```bash
pip install torch==2.8.0 torch-geometric==2.6.1 numpy==1.26.4 matplotlib==3.8.4
```

*(Note: If you intend to run the older, archived scripts in `Scenario_D2D/`, you may additionally need `scipy==1.11.4` and `tensorflow==2.20.0`)*.

---



## How to Run

It is recommended to use a virtual environment. If using the provided `.venv_wpc`, activate it first:
```bash
source .venv_wpc/bin/activate
```

All commands should be run from the root directory of the workspace. The modules use relative imports so executing them as modules or targeting the scripts directly is supported.

**Full training and evaluation pipeline (multi-scenario sweep):**
```bash
python Scenario_JSAC/main.py
```

**Single-scenario deep evaluation (rich plots and detailed metrics):**
```bash
python Scenario_JSAC/test_JSAC.py
```

Artifacts, trained models (`*.pth`), and results (`*.pkl`, `*.png`) are automatically saved into `save_main/` and `save_test/` (or `saves/`, `saves_QoS/` for D2D).

---



## Reference:

- H. Sun, X. Chen, Q. Shi, M. Hong, X. Fu and N. D. Sidiropoulos, "Learning to optimize: Training deep neural networks for wireless resource management," *2017 IEEE 18th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC)*, Sapporo, Japan, 2017, pp. 1-6, doi: 10.1109/SPAWC.2017.8227766. 【W1P1】
- Y. Shen, Y. Shi, J. Zhang and K. B. Letaief, "Graph Neural Networks for Scalable Radio Resource Management: Architecture Design and Theoretical Analysis," in *IEEE Journal on Selected Areas in Communications*, vol. 39, no. 1, pp. 101-115, Jan. 2021, doi: 10.1109/JSAC.2020.3036965. 【W2P2】
- X. Li, M. Chen, Y. Liu, Z. Zhang, D. Liu and S. Mao, "Graph Neural Networks for Joint Communication and Sensing Optimization in Vehicular Networks," in *IEEE Journal on Selected Areas in Communications*, vol. 41, no. 12, pp. 3893-3907, Dec. 2023, doi: 10.1109/JSAC.2023.3322761. 【W4P2】
