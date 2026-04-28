"""Export the D2D GNN for the website Live Run widget.

The browser widget runs layouts up to ``K=20``. PyG's MessagePassing is not a
good browser export boundary, so this script wraps the trained IGCNet weights in
a dense-mask module:

    messages[target, source] = MLP1([x_source, edge_attr[target, source]])
    aggr[target] = max_source(messages[target, source]) over active edges
    x_next[target] = [x_static, MLP2([x_self, aggr])]

That is equivalent to the D2D PyG graph for the small dense/padded layouts the
site needs, and it exports to ONNX Runtime Web cleanly.

Run from the repository root or from ``Scenario_D2D``:

    python Scenario_D2D/export_live_run.py
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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config_system as CS  # noqa: E402
import model_gnn as MG      # noqa: E402


DEFAULT_CHECKPOINT = HERE / "saves" / "gnn_model_K50.pth"
DEFAULT_SCALERS = HERE / "saves" / "scalers.pkl"
DEFAULT_OUT = ROOT / "prototype" / "assets" / "models"
MAX_K = 20


class DenseD2DIGCNet(nn.Module):
    """Dense-mask export wrapper around the trained D2D IGCNet weights."""

    def __init__(self, base: MG.IGCNet):
        super().__init__()
        self.mlp1 = base.mlp1
        self.mlp2 = base.mlp2

    def conv(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        # x: [Kmax, 3]
        # edge_attr: [Kmax, Kmax, 1], indexed [target, source]
        # edge_mask: [Kmax, Kmax], 1 where source -> target is active
        kmax = x.shape[0]
        x_src = x.unsqueeze(0).expand(kmax, kmax, 3)
        msg_in = torch.cat([x_src, edge_attr], dim=-1)
        msg = self.mlp1(msg_in.reshape(kmax * kmax, 4)).reshape(kmax, kmax, 32)

        active = edge_mask.unsqueeze(-1) > 0.5
        msg_masked = torch.where(active, msg, torch.full_like(msg, -1.0e9))
        aggr = msg_masked.max(dim=1).values
        incoming = edge_mask.sum(dim=1, keepdim=True)
        aggr = torch.where(incoming > 0.5, aggr, torch.zeros_like(aggr))

        comb = self.mlp2(torch.cat([x, aggr], dim=1))
        return torch.cat([x[:, :2], comb], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        x1 = self.conv(x, edge_attr, edge_mask)
        x2 = self.conv(x1, edge_attr, edge_mask)
        out = self.conv(x2, edge_attr, edge_mask)
        return out[:, 2] * node_mask


def scaler_to_dict(scaler) -> dict[str, float]:
    return {
        "diag_mean": float(scaler.diag_mean),
        "diag_var": float(scaler.diag_var),
        "off_mean": float(scaler.off_mean),
        "off_var": float(scaler.off_var),
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


def make_smoke_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(7)
    k = 8
    x = np.zeros((MAX_K, 3), dtype=np.float32)
    x[:k, 0] = rng.normal(size=k)
    x[:k, 1] = rng.normal(size=k)

    edge_attr = np.zeros((MAX_K, MAX_K, 1), dtype=np.float32)
    edge_mask = np.zeros((MAX_K, MAX_K), dtype=np.float32)
    for target in range(k):
        for source in range(k):
            if target == source:
                continue
            if rng.random() < 0.65:
                edge_mask[target, source] = 1.0
                edge_attr[target, source, 0] = rng.normal()

    node_mask = np.zeros((MAX_K,), dtype=np.float32)
    node_mask[:k] = 1.0
    return (
        torch.tensor(x),
        torch.tensor(edge_attr),
        torch.tensor(edge_mask),
        torch.tensor(node_mask),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--scalers", default=str(DEFAULT_SCALERS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    scalers_path = Path(args.scalers).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not scalers_path.exists():
        raise FileNotFoundError(f"Missing scalers: {scalers_path}")

    base = MG.IGCNet()
    state = torch.load(checkpoint, map_location="cpu")
    base.load_state_dict(state)
    base.eval()
    wrapper = DenseD2DIGCNet(base).eval()

    x, edge_attr, edge_mask, node_mask = make_smoke_inputs()
    with torch.no_grad():
        smoke = wrapper(x, edge_attr, edge_mask, node_mask)
    if not torch.isfinite(smoke).all():
        raise RuntimeError("Dense wrapper produced non-finite output during smoke check")

    onnx_path = out_dir / "d2d_igcnet_k20.onnx"
    torch.onnx.export(
        wrapper,
        (x, edge_attr, edge_mask, node_mask),
        onnx_path,
        input_names=["x", "edge_attr", "edge_mask", "node_mask"],
        output_names=["powers"],
        opset_version=17,
        do_constant_folding=True,
    )

    import onnx
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    with scalers_path.open("rb") as f:
        scalers = pickle.load(f)

    conf = CS.init_parameters(n_links=MAX_K, field_length=1000)
    manifest = {
        "scenario": "d2d",
        "max_k": MAX_K,
        "model": "d2d_igcnet_k20.onnx",
        "weights_fallback": "d2d_igcnet_k20_weights.json",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "runtime": {
            "vendor": "../vendor/onnxruntime-web/ort.wasm.min.js",
            "wasm_path": "../vendor/onnxruntime-web/",
        },
        "inputs": {
            "x": [MAX_K, 3],
            "edge_attr": [MAX_K, MAX_K, 1],
            "edge_mask": [MAX_K, MAX_K],
            "node_mask": [MAX_K],
        },
        "physics": {
            "field_length": conf.field_length,
            "shortest_direct_link_length": conf.shortest_directLink_length,
            "longest_direct_link_length": conf.longest_directLink_length,
            "shortest_cross_link_length": conf.shortest_crossLink_length,
            "tx_height": conf.tx_height,
            "rx_height": conf.rx_height,
            "carrier_f": conf.carrier_f,
            "antenna_gain_decibel": conf.antenna_gain_decibel,
            "tx_power": float(conf.tx_power),
            "output_noise_power": float(conf.output_noise_power),
            "var_noise": float(conf.output_noise_power / conf.tx_power),
            "threshold": 300,
            "pmax": 1.0,
        },
        "scalers": {
            "dist": scaler_to_dict(scalers["dist"]),
            "loss": scaler_to_dict(scalers["loss"]),
        },
    }
    (out_dir / "d2d_live_manifest.json").write_text(json.dumps(manifest, indent=2))

    fallback = {
        "max_k": MAX_K,
        "mlp1": linear_layers(base.mlp1),
        "mlp2": linear_layers(base.mlp2),
    }
    (out_dir / "d2d_igcnet_k20_weights.json").write_text(json.dumps(fallback))

    print(f"wrote {onnx_path}")
    print(f"wrote {out_dir / 'd2d_live_manifest.json'}")
    print(f"wrote {out_dir / 'd2d_igcnet_k20_weights.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
