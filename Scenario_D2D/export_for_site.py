"""Export D2D pickles to JSON for the static showcase site.

Run from this directory:
    python export_for_site.py [--out ../prototype/assets/data]

Reads:
    saves/simulation_results.pkl            (no-QoS sweep over K)
    saves_QoS/qos_results_K50.pkl           (optional, K=50 QoS study)

Writes to <out>/:
    d2d_sweep_K.json   (main D2D scaling finding — WMMSE / GNN / DNN across K)
    d2d_qos.json       (optional — CDFs for QoS variants)
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = (HERE / ".." / "prototype" / "assets" / "data").resolve()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}")


def _stub(path: Path, shape: dict, reason: str) -> None:
    obj = dict(shape)
    obj["_stub"] = True
    obj["_reason"] = reason
    _write_json(path, obj)


def export_sweep_K(saves: Path, out_dir: Path) -> None:
    """d2d_sweep_K.json — WMMSE / GNN / DNN vs K."""
    pkl = saves / "simulation_results.pkl"
    out = out_dir / "d2d_sweep_K.json"

    methods = ["WMMSE", "GNN", "DNN"]

    if not pkl.exists():
        _stub(
            out,
            {"sweep": "K", "x_key": "K", "x_label": "Number of transceiver pairs (K)",
             "methods": methods, "points": []},
            f"missing {pkl} — re-run Scenario_D2D/main.py to regenerate.",
        )
        print(f"  WARNING: {pkl} missing; wrote stub.")
        return

    with pkl.open("rb") as f:
        r = pickle.load(f)

    ks = r["K"]
    wmmse = r["WMMSE"]
    gnn   = r["GNN"]
    dnn   = r["DNN"]
    t_w   = r["Time_WMM"]
    t_g   = r["Time_GNN"]
    t_d   = r["Time_DNN"]

    points = []
    for i, k in enumerate(ks):
        wm = float(wmmse[i])
        point = {"x": int(k), "metrics": {}}
        for name, sr_arr, t_arr in (
            ("WMMSE", wmmse, t_w),
            ("GNN",   gnn,   t_g),
            ("DNN",   dnn,   t_d),
        ):
            sr = float(sr_arr[i])
            point["metrics"][name] = {
                "sum_rate":                 round(sr, 3),
                "sum_rate_ratio_vs_wmmse":  round(sr / wm, 4) if wm > 0 else 1.0,
                "inference_ms":             round(float(t_arr[i]) * 1000.0, 3),
            }
        points.append(point)

    _write_json(out, {
        "sweep": "K",
        "x_key": "K",
        "x_label": "Number of transceiver pairs (K)",
        "methods": methods,
        "points": points,
    })


def export_qos(saves_qos: Path, out_dir: Path) -> None:
    """d2d_qos.json — per-user rate CDFs for unconstrained vs constrained variants.

    Optional. If the pickle isn't present, write a stub so the UI can degrade gracefully.
    """
    pkl = saves_qos / "qos_results_K50.pkl"
    out = out_dir / "d2d_qos.json"

    methods = ["WMMSE", "WMMSE_QoS", "GNN_unconstrained", "GNN_QoS", "GNN_QoS_anneal"]

    if not pkl.exists():
        _stub(
            out,
            {"K": 50, "r_min": None, "methods": methods, "cdfs": {}, "metrics": {}},
            f"missing {pkl} — re-run Scenario_D2D/test_QoS.py to regenerate.",
        )
        print(f"  WARNING: {pkl} missing; wrote stub.")
        return

    with pkl.open("rb") as f:
        r = pickle.load(f)

    method_map = {
        "WMMSE":            r["metrics_wm"],
        "WMMSE_QoS":        r["metrics_wm_qos"],
        "GNN_unconstrained": r["metrics_old"],
        "GNN_QoS":          r["metrics_QoS"],
        "GNN_QoS_anneal":   r["metrics_QoS_anneal"],
    }

    # Build CDF summaries: sample quantiles so we don't dump 50k floats per method.
    quantile_pts = 200
    qs = [i / (quantile_pts - 1) for i in range(quantile_pts)]

    cdfs = {}
    summary = {}
    for name, m in method_map.items():
        rates = sorted(float(x) for x in m["rates_flat"])
        n = len(rates)
        def pct(q: float) -> float:
            if n == 0: return 0.0
            idx = min(n - 1, max(0, int(round(q * (n - 1)))))
            return round(rates[idx], 5)
        cdfs[name] = {"q": qs, "rate": [pct(q) for q in qs]}
        summary[name] = {
            "violation_rate":   round(float(m["violation_rate"]), 3),
            "sum_rate":         round(float(m["sum_rate"]), 3),
            "jains":            round(float(m["jains"]), 4),
            "p05":              round(float(m["p05"]), 5),
            "p10":              round(float(m["p10"]), 5),
            "p20":              round(float(m["p20"]), 5),
        }

    _write_json(out, {
        "K": int(r["K"]),
        "r_min": round(float(r["R_MIN"]), 4),
        "methods": methods,
        "cdfs": cdfs,
        "metrics": summary,
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"output directory (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {out_dir}")

    print("\n[1/2] K-sweep export...")
    export_sweep_K(HERE / "saves", out_dir)

    print("\n[2/2] QoS export (optional)...")
    export_qos(HERE / "saves_QoS", out_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
