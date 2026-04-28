"""Export the JSAC GNN for the website Live Run widget.

The browser widget uses small editable JSAC layouts, but the trained IGCNet is
node-count agnostic. PyG's MessagePassing is not a stable browser export
boundary, so this script wraps the trained model in a dense-mask module:

    messages[target, source] = MLP1([x_source, edge_attr[target, source]])
    aggr[target] = max_source(messages[target, source]) over active edges
    x_next[target] = [x_static, MLP2([x_self, aggr])]

The wrapper exports raw per-link logits. The website applies the per-Blue-car
softmax in JavaScript so every Blue car's normalized power budget sums to 1.

Run from the repository root:

    python web_tools/export_jsac_live_run.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCENARIO_DIR = ROOT / "Scenario_JSAC"
if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

import config_system as CS  # noqa: E402
import model_gnn as MG      # noqa: E402


DEFAULT_CHECKPOINT = SCENARIO_DIR / "save_main" / "gnn_model.pth"
DEFAULT_SCALERS = SCENARIO_DIR / "save_main" / "scalers.pkl"
DEFAULT_HYPERPARAMS = SCENARIO_DIR / "save_main" / "hyperparams.pkl"
DEFAULT_OUT = ROOT / "web" / "assets" / "models"
MAX_K = 50


class DenseJSACIGCNet(nn.Module):
    """Dense-mask export wrapper around the trained JSAC IGCNet weights."""

    def __init__(self, base: MG.IGCNet):
        super().__init__()
        self.mlp1 = base.mlp1
        self.mlp2 = base.mlp2

    def conv(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        # x: [Kmax, 4]
        # edge_attr: [Kmax, Kmax, 3], indexed [target, source]
        # edge_mask: [Kmax, Kmax], 1 where source -> target is active
        kmax = x.shape[0]
        x_src = x.unsqueeze(0).expand(kmax, kmax, 4)
        msg_in = torch.cat([x_src, edge_attr], dim=-1)
        msg = self.mlp1(msg_in.reshape(kmax * kmax, 7)).reshape(kmax, kmax, 32)

        active = edge_mask.unsqueeze(-1) > 0.5
        msg_masked = torch.where(active, msg, torch.full_like(msg, -1.0e9))
        aggr = msg_masked.max(dim=1).values
        incoming = edge_mask.sum(dim=1, keepdim=True)
        aggr = torch.where(incoming > 0.5, aggr, torch.zeros_like(aggr))

        comb = self.mlp2(torch.cat([x, aggr], dim=1))
        return torch.cat([x[:, :3], comb], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        x1 = self.conv(x, edge_attr, edge_mask)
        x2 = self.conv(x1, edge_attr, edge_mask)
        x3 = self.conv(x2, edge_attr, edge_mask)
        out = self.conv(x3, edge_attr, edge_mask)
        return out[:, 3] * node_mask


def scaler_to_dict(scaler) -> dict[str, float]:
    return {
        "sense_mean": float(scaler.sense_mean),
        "sense_std": float(scaler.sense_var),
        "comm_mean": float(scaler.comm_mean),
        "comm_std": float(scaler.comm_var),
        "interf_mean": float(scaler.interf_mean),
        "interf_std": float(scaler.interf_var),
    }


def linear_layers(module: nn.Module) -> list[dict[str, list]]:
    layers = []
    for child in module.modules():
        if isinstance(child, nn.Linear):
            layers.append({
                "weight": child.weight.detach().cpu().numpy().astype(float).tolist(),
                "bias": child.bias.detach().cpu().numpy().astype(float).tolist(),
            })
    return layers


def make_smoke_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Data, int]:
    rng = np.random.default_rng(11)
    k = 9
    x_small = np.zeros((k, 4), dtype=np.float32)
    x_small[:, 0] = rng.normal(size=k)
    x_small[:, 1] = rng.normal(size=k)
    x_small[:, 2] = rng.integers(0, 2, size=k).astype(np.float32)

    edge_attr_dense = np.zeros((MAX_K, MAX_K, 3), dtype=np.float32)
    edge_mask = np.zeros((MAX_K, MAX_K), dtype=np.float32)
    rows, cols, attrs = [], [], []
    for target in range(k):
        for source in range(k):
            if target == source:
                continue
            if rng.random() < 0.42:
                attr = rng.normal(size=3).astype(np.float32)
                edge_mask[target, source] = 1.0
                edge_attr_dense[target, source] = attr
                rows.append(source)
                cols.append(target)
                attrs.append(attr)

    x = np.zeros((MAX_K, 4), dtype=np.float32)
    x[:k] = x_small
    node_mask = np.zeros((MAX_K,), dtype=np.float32)
    node_mask[:k] = 1.0

    data = Data(
        x=torch.tensor(x_small),
        edge_index=torch.tensor([rows, cols], dtype=torch.long),
        edge_attr=torch.tensor(np.asarray(attrs), dtype=torch.float32),
    )
    return (
        torch.tensor(x),
        torch.tensor(edge_attr_dense),
        torch.tensor(edge_mask),
        torch.tensor(node_mask),
        data,
        k,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--scalers", default=str(DEFAULT_SCALERS))
    ap.add_argument("--hyperparams", default=str(DEFAULT_HYPERPARAMS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    scalers_path = Path(args.scalers).resolve()
    hyperparams_path = Path(args.hyperparams).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not scalers_path.exists():
        raise FileNotFoundError(f"Missing scalers: {scalers_path}")
    if not hyperparams_path.exists():
        raise FileNotFoundError(f"Missing hyperparams: {hyperparams_path}")

    base = MG.IGCNet()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    base.load_state_dict(state)
    base.eval()
    wrapper = DenseJSACIGCNet(base).eval()

    x, edge_attr, edge_mask, node_mask, data, k = make_smoke_inputs()
    with torch.no_grad():
        dense_logits = wrapper(x, edge_attr, edge_mask, node_mask)[:k]
        pyg_logits = base(data)[:, 3]
    if not torch.isfinite(dense_logits).all():
        raise RuntimeError("Dense wrapper produced non-finite output during smoke check")
    if not torch.allclose(dense_logits, pyg_logits, atol=1e-5, rtol=1e-5):
        diff = torch.max(torch.abs(dense_logits - pyg_logits)).item()
        raise RuntimeError(f"Dense wrapper does not match PyG model; max diff={diff:.3e}")

    onnx_path = out_dir / "jsac_igcnet_k50.onnx"
    torch.onnx.export(
        wrapper,
        (x, edge_attr, edge_mask, node_mask),
        onnx_path,
        input_names=["x", "edge_attr", "edge_mask", "node_mask"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )

    import onnx
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    with scalers_path.open("rb") as f:
        scalers = pickle.load(f)
    with hyperparams_path.open("rb") as f:
        hp = pickle.load(f)

    train_conf = CS.init_parameters_JSAC(
        n_blue=int(hp["TRAIN_N_BLUE"]),
        n_yellow_per_blue=int(hp["TRAIN_N_YELLOW"]),
        n_green_per_blue=int(hp["TRAIN_N_GREEN"]),
        field_length=float(hp["FIELD_LENGTH"]),
    )

    manifest = {
        "scenario": "jsac",
        "max_k": MAX_K,
        "model": "jsac_igcnet_k50.onnx",
        "weights_fallback": "jsac_igcnet_k50_weights.json",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "inputs": {
            "x": [MAX_K, 4],
            "edge_attr": [MAX_K, MAX_K, 3],
            "edge_mask": [MAX_K, MAX_K],
            "node_mask": [MAX_K],
        },
        "trained_config": {
            "B": int(hp["TRAIN_N_BLUE"]),
            "M_y": int(hp["TRAIN_N_YELLOW"]),
            "M_g": int(hp["TRAIN_N_GREEN"]),
            "K": int(hp["TRAIN_N_BLUE"]) * (int(hp["TRAIN_N_YELLOW"]) + int(hp["TRAIN_N_GREEN"])),
            "shuffle_channels": bool(hp["SHUFFLE_CHANNELS"]),
        },
        "live_defaults": {
            "B": 4,
            "M_y": int(hp["TRAIN_N_YELLOW"]),
            "M_g": int(hp["TRAIN_N_GREEN"]),
            "B_min": 2,
            "B_max": 6,
            "M_y_min": 1,
            "M_y_max": 3,
            "M_g_min": 1,
            "M_g_max": 4,
        },
        "physics": {
            "field_length": float(train_conf.field_length),
            "min_blue_dist": float(train_conf.min_blue_dist),
            "rx_min_radius": float(train_conf.rx_min_radius),
            "rx_max_radius": float(train_conf.rx_max_radius),
            "min_rx_separation": float(train_conf.min_rx_separation),
            "tx_height": float(train_conf.tx_height),
            "rx_height": float(train_conf.rx_height),
            "carrier_f": float(train_conf.carrier_f),
            "antenna_gain_decibel": float(train_conf.antenna_gain_decibel),
            "tx_power": float(train_conf.tx_power),
            "output_noise_power": float(train_conf.output_noise_power),
            "var_noise": float(train_conf.output_noise_power / train_conf.tx_power),
            "pmax": 1.0,
            "sinr_min": float(hp["SINR_MIN"]),
            "sinr_min_db": float(hp["SINR_MIN_DB"]),
            "wmmse_alpha_yellow": float(hp["WMMSE_ALPHA_YELLOW"]),
        },
        "scalers": {
            "dist": scaler_to_dict(scalers["dist"]),
            "loss": scaler_to_dict(scalers["loss"]),
        },
    }
    (out_dir / "jsac_live_manifest.json").write_text(json.dumps(manifest, indent=2))

    fallback = {
        "max_k": MAX_K,
        "mlp1": linear_layers(base.mlp1),
        "mlp2": linear_layers(base.mlp2),
    }
    (out_dir / "jsac_igcnet_k50_weights.json").write_text(json.dumps(fallback))

    print(f"wrote {onnx_path}")
    print(f"wrote {out_dir / 'jsac_live_manifest.json'}")
    print(f"wrote {out_dir / 'jsac_igcnet_k50_weights.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
