"""Export JSAC pickles to JSON for the static showcase site.

Run from this directory:
    python export_for_site.py [--out ../prototype/assets/data]

Reads:
    save_main/results_sweep_B.pkl
    save_main/results_sweep_M.pkl
    save_test/test_jsac_results.pkl
    save_test/hyperparams.pkl

Writes to <out>/:
    sweep_B.json, sweep_M.json, layouts_index.json,
    layouts/jsac_layout_{id}.json (one per exported sample)

No model retraining. No pickle shipping.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path


# --------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = (HERE / ".." / "prototype" / "assets" / "data").resolve()


def _n(v):
    """Cast numpy scalars/arrays into JSON-safe Python values."""
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
    except ImportError:
        pass
    return v


def _round_list(xs, ndigits=4):
    return [round(float(x), ndigits) for x in xs]


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")


def _stub(path: Path, shape: dict, reason: str) -> None:
    """Write a stub JSON so the component can render an empty state."""
    obj = dict(shape)
    obj["_stub"] = True
    obj["_reason"] = reason
    _write_json(path, obj)


# --------------------------------------------------------------------
# exporters
# --------------------------------------------------------------------
def export_sweeps(save_main: Path, out_dir: Path) -> None:
    """sweep_B.json and sweep_M.json in the shape documented in WEB_PROMPT §8.1."""
    methods = ["Naive", "WMMSE", "GNN"]

    # --- Sweep B ---
    b_pkl = save_main / "results_sweep_B.pkl"
    b_out = out_dir / "sweep_B.json"
    if not b_pkl.exists():
        _stub(
            b_out,
            {"sweep": "B", "x_key": "B", "x_label": "Number of Blue cars (B)",
             "methods": methods, "points": []},
            f"missing {b_pkl} — re-run Scenario_JSAC/main.py to regenerate.",
        )
        print(f"  WARNING: {b_pkl} missing; wrote stub.")
    else:
        with b_pkl.open("rb") as f:
            r = pickle.load(f)
        points = []
        for i, x in enumerate(r["B"]):
            point = {"x": int(x), "K": int(r["K"][i]), "metrics": {}}
            for m in methods:
                point["metrics"][m] = {
                    "green_sumrate":         round(float(r[f"{m}_sr"][i]), 3),
                    "yellow_violation_pct":  round(float(r[f"{m}_viol"][i]), 3),
                    "inference_ms":          round(float(r[f"{m}_time"][i]) * 1000.0, 3),
                }
            points.append(point)
        _write_json(b_out, {
            "sweep": "B",
            "x_key": "B",
            "x_label": "Number of Blue cars (B)",
            "methods": methods,
            "points": points,
        })

    # --- Sweep M ---
    m_pkl = save_main / "results_sweep_M.pkl"
    m_out = out_dir / "sweep_M.json"
    if not m_pkl.exists():
        _stub(
            m_out,
            {"sweep": "M", "x_key": "M", "x_label": "Links per Blue car (M)",
             "methods": methods, "points": []},
            f"missing {m_pkl} — re-run Scenario_JSAC/main.py to regenerate.",
        )
        print(f"  WARNING: {m_pkl} missing; wrote stub.")
    else:
        with m_pkl.open("rb") as f:
            r = pickle.load(f)
        points = []
        for i, x in enumerate(r["M"]):
            point = {
                "x": int(x),
                "K": int(r["K"][i]),
                "M_y": int(r["M_y"][i]),
                "M_g": int(r["M_g"][i]),
                "label": f"({int(r['M_y'][i])},{int(r['M_g'][i])})",
                "metrics": {},
            }
            for m in methods:
                point["metrics"][m] = {
                    "green_sumrate":         round(float(r[f"{m}_sr"][i]), 3),
                    "yellow_violation_pct":  round(float(r[f"{m}_viol"][i]), 3),
                    "inference_ms":          round(float(r[f"{m}_time"][i]) * 1000.0, 3),
                }
            points.append(point)
        _write_json(m_out, {
            "sweep": "M",
            "x_key": "M",
            "x_label": "Links per Blue car (M = My+Mg)",
            "methods": methods,
            "points": points,
        })


def export_layouts(save_test: Path, out_dir: Path) -> None:
    """Export 2 representative layouts (the `sample_layouts` from test_JSAC.py).

    Method names in the pickle are ('Equal Power', 'WMMSE', 'GNN'); we normalize
    them to the site's vocabulary ('Naive', 'WMMSE', 'GNN').
    """
    pkl = save_test / "test_jsac_results.pkl"
    hp_pkl = save_test / "hyperparams.pkl"
    layouts_dir = out_dir / "layouts"
    index_path = out_dir / "layouts_index.json"

    if not pkl.exists() or not hp_pkl.exists():
        _stub(
            index_path,
            {"layouts": []},
            f"missing {pkl} or {hp_pkl} — re-run Scenario_JSAC/test_JSAC.py to regenerate.",
        )
        print(f"  WARNING: layout pickles missing; wrote stub index.")
        return

    with pkl.open("rb") as f:
        test = pickle.load(f)
    with hp_pkl.open("rb") as f:
        hp = pickle.load(f)

    B   = int(hp["N_BLUE"])
    My  = int(hp["N_YELLOW"])
    Mg  = int(hp["N_GREEN"])
    M   = My + Mg
    field = float(hp["FIELD_LENGTH"])

    # Map the pickle's method names to the site's palette-keyed names.
    method_map = {"Equal Power": "Naive", "WMMSE": "WMMSE", "GNN": "GNN"}

    sample_layouts = test["sample_layouts"]   # (n_samples, K, 4) → Tx_x, Tx_y, Rx_x, Rx_y
    index_entries = []

    for s in range(sample_layouts.shape[0]):
        K = sample_layouts.shape[1]
        assert K == B * M, f"K mismatch: pickle says {K}, hyperparams say B*M={B*M}"

        # Build blue-car positions from the first link in each group (Tx is shared
        # across all M links in a group).
        blue = []
        rx   = []
        for g in range(B):
            tx_x = float(sample_layouts[s, g * M, 0])
            tx_y = float(sample_layouts[s, g * M, 1])
            blue.append({"id": g, "x": round(tx_x, 2), "y": round(tx_y, 2)})

            for m in range(M):
                idx = g * M + m
                # Convention: first M_y links per blue-car are Yellow, rest are Green.
                # The channel index (orthogonal slot 0..M-1) is m modulo M.
                link_type = "yellow" if m < My else "green"
                rx.append({
                    "id":      int(idx),
                    "blue":    int(g),
                    "channel": int(m),
                    "type":    link_type,
                    "x":       round(float(sample_layouts[s, idx, 2]), 2),
                    "y":       round(float(sample_layouts[s, idx, 3]), 2),
                })

        power = {}
        metrics = {"green_sumrate": {}, "yellow_viol_rate": {}}
        for method_src, method_dst in method_map.items():
            allocs = test["methods"][method_src]["sample_allocs"][s]  # shape (K,)
            power[method_dst]               = _round_list(allocs, 4)
            metrics["green_sumrate"][method_dst]    = round(float(test["methods"][method_src]["sample_green_sumrate"][s]), 3)
            metrics["yellow_viol_rate"][method_dst] = round(float(test["methods"][method_src]["yellow_viol_rate"]), 3)

        layout_id = f"{s+1:02d}"
        layout_obj = {
            "scenario": "jsac",
            "id": layout_id,
            "config": {"B": B, "M_y": My, "M_g": Mg, "K": K, "field": field},
            "blue": blue,
            "rx":   rx,
            "power": power,
            "metrics": metrics,
        }
        _write_json(layouts_dir / f"jsac_layout_{layout_id}.json", layout_obj)
        index_entries.append({
            "id":      layout_id,
            "config":  layout_obj["config"],
            "metrics": metrics,
            "path":    f"layouts/jsac_layout_{layout_id}.json",
        })

    _write_json(index_path, {"layouts": index_entries})


# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"output directory (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {out_dir}")
    save_main = HERE / "save_main"
    save_test = HERE / "save_test"

    print("\n[1/2] Sweep exports...")
    export_sweeps(save_main, out_dir)

    print("\n[2/2] Layout exports...")
    export_layouts(save_test, out_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
