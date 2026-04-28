"""Export D2D pickles to JSON (and one SVG figure) for the static showcase site.

Run from the repository root:
    python web_tools/export_d2d_for_site.py [--out web/assets/data]
                                            [--figures-out web/assets/images/figures]

Reads:
    saves/simulation_results.pkl            (no-QoS sweep over K)
    saves_QoS/qos_results_K50.pkl           (optional, K=50 QoS study)

Writes to <out>/:
    d2d_sweep_K.json   (main D2D scaling finding — WMMSE / GNN / DNN across K)
    d2d_qos.json       (optional — CDFs for QoS variants)

Writes to <figures-out>/:
    d2d_qos_cdf.svg    (optional — site-styled CDF figure, matches dark palette)
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCENARIO_DIR = ROOT / "Scenario_D2D"
DEFAULT_OUT = ROOT / "web" / "assets" / "data"
DEFAULT_FIGURES_OUT = ROOT / "web" / "assets" / "images" / "figures"

if str(SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(SCENARIO_DIR))

# Site palette (matches web/styles.css :root tokens).
SITE = {
    "bg":       "#0a0b0e",
    "surface":  "#14161c",
    "text":     "#e8e8ea",
    "text_dim": "#9a9aa0",
    "rule":     "#262830",   # solid stand-in for rgba(255,255,255,0.08)
    "blue":     "#4da3ff",
    "orange":   "#ff6a3d",
    "yellow":   "#f6c445",
    "violet":   "#b265d9",
    "grey":     "#888888",
    "red":      "#ff5252",
}


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
    """d2d_sweep_K.json — WMMSE / GNN / DNN / Greedy vs K."""
    pkl = saves / "simulation_results.pkl"
    out = out_dir / "d2d_sweep_K.json"

    methods = ["WMMSE", "GNN", "DNN", "Greedy"]

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

    ks    = r["K"]
    wmmse = r["WMMSE"]
    gnn   = r["GNN"]
    dnn   = r["DNN"]
    grd   = r["Greedy"]
    t_w   = r["Time_WMM"]
    t_g   = r["Time_GNN"]
    t_d   = r["Time_DNN"]
    t_gr  = r["Time_Grd"]

    points = []
    for i, k in enumerate(ks):
        wm = float(wmmse[i])
        point = {"x": int(k), "metrics": {}}
        for name, sr_arr, t_arr in (
            ("WMMSE",  wmmse, t_w),
            ("GNN",    gnn,   t_g),
            ("DNN",    dnn,   t_d),
            ("Greedy", grd,   t_gr),
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


def export_qos_figure_svg(saves_qos: Path, figures_dir: Path) -> None:
    """d2d_qos_cdf.svg — site-styled re-render of the QoS CDF.

    Mirrors main_supporter.plot_qos_cdf but uses the site palette and dark
    theme, exported as SVG so it slots into the showcase page next to the
    other dark-themed widgets.
    """
    pkl = saves_qos / "qos_results_K50.pkl"
    out_path = figures_dir / "d2d_qos_cdf.svg"

    if not pkl.exists():
        print(f"  WARNING: {pkl} missing; skipping SVG export.")
        return

    # Lazy import so the JSON-only path stays lightweight.
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with pkl.open("rb") as f:
        r = pickle.load(f)

    R_MIN = float(r["R_MIN"])
    series = [
        ("WMMSE (Baseline)",         r["metrics_wm"],         SITE["blue"],   ":",   1.5),
        ("WMMSE (QoS Constraint)",   r["metrics_wm_qos"],     SITE["blue"],   "-.",  1.5),
        ("Original GNN",             r["metrics_old"],        SITE["violet"], "--",  1.6),
        ("Modified GNN (Constraint)",            r["metrics_QoS"],        SITE["orange"], "-",   2.4),
        ("Modified GNN with Penalty Annealing", r["metrics_QoS_anneal"], SITE["yellow"], "-",   2.0),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(SITE["bg"])
    ax.set_facecolor(SITE["bg"])

    for label, m, color, ls, lw in series:
        sorted_rates = np.sort(m["rates_flat"])
        p = np.arange(len(sorted_rates)) / float(len(sorted_rates) - 1)
        ax.plot(sorted_rates, p, label=label, color=color, linestyle=ls, linewidth=lw)

    ax.axvline(x=R_MIN, color=SITE["red"], linestyle="-.", linewidth=1.4,
               label=f"r_min = {R_MIN}")

    # Axis range — keep matplotlib's framing (lower/middle of the distribution).
    sample = np.sort(r["metrics_wm"]["rates_flat"])
    ax.set_xlim(0, max(np.percentile(sample, 95), R_MIN * 3))
    ax.set_ylim(0, 1.0)

    ax.set_xlabel("Data Rate (bits/s/Hz)", color=SITE["text"])
    ax.set_ylabel("Cumulative Probability (CDF)", color=SITE["text"])
    ax.set_title("CDF of User Data Rates", color=SITE["text"], pad=12)

    for spine in ax.spines.values():
        spine.set_color(SITE["rule"])
    ax.tick_params(colors=SITE["text_dim"])
    ax.grid(True, color=SITE["rule"], alpha=0.6, linewidth=0.7)

    legend = ax.legend(loc="lower right", facecolor=SITE["surface"],
                       edgecolor=SITE["rule"], labelcolor=SITE["text"], framealpha=0.95)
    legend.get_frame().set_linewidth(0.7)

    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"JSON output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--figures-out", default=str(DEFAULT_FIGURES_OUT),
                    help=f"SVG figures output directory (default: {DEFAULT_FIGURES_OUT})")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_out).resolve()
    print(f"Exporting JSON to {out_dir}")
    print(f"Exporting figures to {figures_dir}")

    print("\n[1/3] K-sweep export...")
    export_sweep_K(SCENARIO_DIR / "saves", out_dir)

    print("\n[2/3] QoS JSON export (optional)...")
    export_qos(SCENARIO_DIR / "saves_QoS", out_dir)

    print("\n[3/3] QoS SVG figure export (optional)...")
    export_qos_figure_svg(SCENARIO_DIR / "saves_QoS", figures_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
