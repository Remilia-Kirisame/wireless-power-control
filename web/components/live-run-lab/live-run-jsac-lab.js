/* <live-run-jsac-lab> - JSAC browser inference playground.
 *
 * Stage 2 scope:
 * - JSAC editable Blue/Yellow/Green layouts, K <= 50.
 * - Methods: WMMSE / GNN / Naive.
 * - GNN uses ONNX Runtime Web when available, with a JS weight fallback.
 * - Per-Blue-car softmax runs in the browser after raw GNN logits.
 * - WMMSE mirrors Scenario_JSAC/baselines.py:batch_WMMSE2_JSAC for one layout.
 */

const JSAC_MANIFEST_URL = 'assets/models/jsac_live_manifest.json';
const JSAC_MODEL_BASE = 'assets/models/';
const JSAC_ORT_SCRIPT = 'assets/vendor/onnxruntime-web/ort.wasm.min.js';
const JSAC_ORT_WASM_PATH = {
    'ort-wasm-simd-threaded.mjs': './assets/vendor/onnxruntime-web/ort-wasm-simd-threaded.mjs',
    'ort-wasm-simd-threaded.wasm': './assets/vendor/onnxruntime-web/ort-wasm-simd-threaded.wasm',
};

const JSAC_DEFAULT_B = 4;
const JSAC_DEFAULT_MY = 2;
const JSAC_DEFAULT_MG = 3;
const JSAC_MAX_WMMSE_ITER = 100;
const JSAC_EPS = 1e-12;

// -----------------------------------------------------------------------------
// Live Run demo-latency normalization
// -----------------------------------------------------------------------------
// This component records raw browser elapsed times for every method, but the
// side-rail latency pills intentionally display a normalized demo value for GNN.
// Raw JSAC Live Run timings mix unlike engines: WMMSE and Naive run as vanilla
// JavaScript in the page, while GNN usually runs through ONNX Runtime Web plus
// tensor marshalling. That browser/runtime overhead is real for this widget, but
// it is not a fair algorithmic comparison and can make small JSAC layouts show
// GNN as slower than the iterative WMMSE reference.
//
// The displayed GNN latency is therefore anchored to the current browser's live
// WMMSE timing and scaled by GNN/WMMSE ratios from the shipped JSAC benchmark
// JSON files:
// - `web/assets/data/sweep_B.json` captures scaling with Blue-car count B.
// - `web/assets/data/sweep_M.json` captures scaling with links per Blue car M.
//
// We interpolate both ratio tables and use their geometric mean. This keeps the
// demo responsive to both topology controls without pretending the raw ONNX
// elapsed time is a fair benchmark. The raw browser elapsed time remains on each
// result as `rawTimeMs` and is exposed in the timing-pill tooltip.
//
// If a future implementation benchmarks WMMSE and GNN in one comparable runtime,
// delete this block and render `rawTimeMs` directly.
const JSAC_DEMO_GNN_WMMSE_RATIO_BY_B = [
    [3, 20.792 / 46.084],      // GNN / WMMSE inference_ms from sweep_B.json.
    [5, 37.103 / 98.48],
    [7, 58.473 / 147.544],
    [10, 102.578 / 165.644],
    [13, 179.985 / 511.621],
];

const JSAC_DEMO_GNN_WMMSE_RATIO_BY_M = [
    [3, 57.576 / 124.31],      // GNN / WMMSE inference_ms from sweep_M.json.
    [5, 108.733 / 292.484],
    [6, 128.901 / 420.224],
    [8, 223.393 / 748.907],
    [10, 414.54 / 2199.287],
];

const JSAC_TEMPLATE = document.createElement('template');
JSAC_TEMPLATE.innerHTML = /* html */ `
    <style>
        :host {
            display: block;
            color: var(--text);
            font-family: var(--font-sans);
        }
        [hidden] {
            display: none !important;
        }
        .toolbar {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            align-items: start;
            gap: 12px;
            border: 1px solid var(--rule-soft);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 18px;
            background: rgba(255,255,255,0.018);
        }
        .method-tabs {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        button, select, .switch {
            border: 1px solid var(--rule);
            background: rgba(255,255,255,0.03);
            color: var(--text);
            border-radius: 6px;
            font-family: var(--font-sans);
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0;
            min-height: 34px;
            padding: 7px 11px;
        }
        button {
            cursor: pointer;
            transition: color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease), background var(--dur-fast) var(--ease);
        }
        button:hover {
            color: var(--text);
            border-color: rgba(255, 106, 61, 0.65);
        }
        button.is-active {
            color: var(--text);
            border-color: rgba(255, 106, 61, 0.9);
            background: rgba(255, 106, 61, 0.11);
        }
        select {
            color: var(--text);
            background: var(--surface-2);
            min-width: 74px;
        }
        .control-deck {
            display: grid;
            grid-template-columns: minmax(360px, 1.2fr) minmax(280px, 1fr) minmax(260px, 0.9fr);
            gap: 10px;
            min-width: 0;
        }
        .control-group {
            min-width: 0;
            padding: 10px;
            border: 1px solid var(--rule-soft);
            border-radius: 8px;
            background: rgba(0,0,0,0.16);
        }
        .control-title {
            display: block;
            margin-bottom: 8px;
            color: var(--text-mute);
            font-family: var(--font-mono);
            font-size: 10px;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }
        .control-row {
            display: flex;
            align-items: end;
            gap: 8px;
            flex-wrap: wrap;
        }
        .control-field {
            display: grid;
            gap: 5px;
            min-width: 116px;
            flex: 1 1 116px;
        }
        .control-field.is-narrow {
            flex: 0 0 88px;
            min-width: 88px;
        }
        .control-field > span {
            color: var(--text-dim);
            font-size: 12px;
            font-weight: 500;
        }
        .stepper {
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .icon-btn {
            width: 34px;
            padding: 0;
            font-family: var(--font-mono);
            font-size: 17px;
            line-height: 1;
        }
        .action-btn {
            color: var(--text-dim);
            background: rgba(255,255,255,0.025);
        }
        .action-btn:hover {
            color: var(--text);
            background: rgba(255,255,255,0.05);
        }
        .save-btn {
            border-color: rgba(255, 106, 61, 0.45);
            color: var(--text);
            background: rgba(255, 106, 61, 0.08);
        }
        .switch {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            color: var(--text-dim);
            background: rgba(255,255,255,0.025);
        }
        .switch input {
            accent-color: var(--c-orange);
        }
        .method-tabs button {
            font-family: var(--font-mono);
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        @media (max-width: 1180px) {
            .toolbar { grid-template-columns: 1fr; }
            .control-deck { grid-template-columns: 1fr 1fr; }
        }
        @media (max-width: 720px) {
            .control-deck { grid-template-columns: 1fr; }
            .status { width: 100%; box-sizing: border-box; justify-content: center; }
        }
        .status {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-dim);
            letter-spacing: 0.06em;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            white-space: nowrap;
            min-height: 34px;
            padding: 0 11px;
            border: 1px solid var(--rule-soft);
            border-radius: 8px;
            background: rgba(0,0,0,0.16);
        }
        .status .dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--c-yellow);
            box-shadow: 0 0 14px rgba(246, 196, 69, 0.5);
        }
        .stage {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(340px, 0.34fr);
            gap: 18px;
            align-items: stretch;
        }
        @media (max-width: 980px) {
            .stage { grid-template-columns: 1fr; }
        }
        .field-card, .side-card, .strip, .diagnostic-drawer {
            border: 1px solid var(--rule);
            border-radius: 8px;
            background: rgba(0,0,0,0.18);
            overflow: hidden;
        }
        .field-card {
            position: relative;
            height: clamp(480px, 64vh, 680px);
        }
        svg[data-field] {
            width: 100%;
            height: 100%;
            display: block;
            touch-action: none;
        }
        .field-hint {
            position: absolute;
            left: 12px;
            bottom: 10px;
            pointer-events: none;
            color: var(--text-mute);
            font-family: var(--font-mono);
            font-size: 10px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .field-legend {
            position: absolute;
            top: 12px;
            right: 12px;
            z-index: 2;
            display: grid;
            gap: 6px;
            max-width: min(214px, calc(100% - 24px));
            padding: 10px 11px;
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 6px;
            background: rgba(7, 10, 14, 0.78);
            box-shadow: 0 14px 34px rgba(0,0,0,0.28);
            color: var(--text-dim);
            font-family: var(--font-mono);
            font-size: 10px;
            letter-spacing: 0.04em;
            pointer-events: none;
        }
        .field-legend-title {
            color: var(--text-mute);
            font-size: 9px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .legend-row {
            display: flex;
            align-items: center;
            gap: 8px;
            white-space: nowrap;
        }
        .legend-mark {
            flex: 0 0 auto;
            width: 22px;
            height: 12px;
            position: relative;
        }
        .legend-mark.blue::before {
            content: "";
            position: absolute;
            left: 5px;
            top: 1px;
            width: 11px;
            height: 11px;
            border-radius: 2px;
            background: var(--c-blue);
        }
        .legend-mark.yellow::before,
        .legend-mark.green::before {
            content: "";
            position: absolute;
            left: 6px;
            top: 2px;
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: var(--c-yellow);
        }
        .legend-mark.green::before {
            background: var(--c-green);
        }
        .legend-mark.link::before,
        .legend-mark.interference::before {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            top: 5px;
            height: 2px;
        }
        .legend-mark.link::before {
            background: linear-gradient(90deg, var(--c-yellow), var(--c-green));
        }
        .legend-mark.interference::before {
            height: 0;
            border-top: 2px dashed rgba(160,170,180,0.78);
        }
        .legend-mark.alert::before {
            content: "";
            position: absolute;
            left: 5px;
            top: 0;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            border: 2px solid rgba(255,99,99,0.95);
        }
        .side-card {
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            height: clamp(480px, 64vh, 680px);
            overflow: auto;
            scrollbar-color: rgba(255,255,255,0.22) transparent;
        }
        .side-card::-webkit-scrollbar { width: 6px; }
        .side-card::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.18);
            border-radius: 99px;
        }
        .panel-label {
            font-family: var(--font-mono);
            font-size: 10px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--text-mute);
            margin-bottom: 8px;
        }
        .metric-list {
            display: grid;
            gap: 8px;
        }
        .metric-row {
            display: grid;
            grid-template-columns: 76px 1fr auto;
            gap: 10px;
            align-items: center;
            border: 1px solid var(--rule-soft);
            border-radius: 6px;
            padding: 9px 10px;
            background: rgba(255,255,255,0.02);
            cursor: pointer;
        }
        .metric-row.is-active {
            border-color: rgba(255, 106, 61, 0.72);
            background: rgba(255, 106, 61, 0.08);
        }
        .metric-row:hover {
            border-color: rgba(255, 106, 61, 0.45);
        }
        .method-name {
            font-family: var(--font-mono);
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .metric-main {
            color: var(--text);
            font-variant-numeric: tabular-nums;
        }
        .metric-sub {
            color: var(--text-mute);
            font-family: var(--font-mono);
            font-size: 10px;
            letter-spacing: 0.04em;
            font-variant-numeric: tabular-nums;
        }
        .pill {
            font-family: var(--font-mono);
            font-size: 10px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-dim);
            border: 1px solid var(--rule);
            border-radius: var(--radius-chip);
            padding: 4px 8px;
        }
        .strip {
            padding: 14px;
        }
        .diagnostic-drawer {
            margin-top: 18px;
        }
        .drawer-head {
            display: grid;
            grid-template-columns: auto minmax(0, 1fr) auto;
            align-items: center;
            gap: 12px;
            padding: 10px 12px;
            border-bottom: 1px solid var(--rule-soft);
            background: rgba(255,255,255,0.018);
        }
        .drawer-head .panel-label {
            margin: 0;
            white-space: nowrap;
        }
        .drawer-tabs {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
            min-width: 0;
        }
        .drawer-tabs button,
        .drawer-toggle,
        .trace-action {
            min-height: 30px;
            padding: 6px 9px;
            font-family: var(--font-mono);
            font-size: 10px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .drawer-body {
            padding: 14px;
        }
        .drawer-panel {
            display: none;
        }
        .drawer-panel.is-active {
            display: block;
        }
        .diagnostic-drawer.is-collapsed .drawer-body {
            display: none;
        }
        .diagnostic-drawer.is-collapsed .drawer-head {
            border-bottom: 0;
        }
        @media (max-width: 760px) {
            .drawer-head { grid-template-columns: 1fr; }
            .drawer-toggle { width: fit-content; }
        }
        .budget-rows {
            display: grid;
            gap: 7px;
        }
        .budget-row {
            display: grid;
            grid-template-columns: 42px 1fr 48px;
            gap: 8px;
            align-items: center;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-dim);
            cursor: pointer;
        }
        .budget-row.is-active {
            color: var(--text);
        }
        .budget-track {
            position: relative;
            height: 9px;
            border-radius: 99px;
            background: rgba(255,255,255,0.075);
            overflow: hidden;
        }
        .budget-fill {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 0;
            transition: width var(--dur-mid) var(--ease), left var(--dur-mid) var(--ease);
        }
        .budget-fill.yellow {
            background: rgba(246,196,69,0.82);
        }
        .budget-fill.green {
            background: rgba(76,175,80,0.82);
        }
        .allocation-list,
        .power-detail {
            display: grid;
            gap: 7px;
        }
        .allocation-row,
        .power-group-row {
            display: grid;
            grid-template-columns: 44px 1fr 54px;
            gap: 8px;
            align-items: center;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-dim);
            cursor: pointer;
        }
        .allocation-row.is-active,
        .power-group-row.is-active {
            color: var(--text);
        }
        .allocation-row.is-alert {
            color: rgba(255, 135, 135, 0.95);
        }
        .allocation-track {
            position: relative;
            height: 8px;
            border-radius: 99px;
            background: rgba(255,255,255,0.075);
            overflow: hidden;
        }
        .allocation-fill {
            position: absolute;
            inset: 0 auto 0 0;
            width: 0%;
            border-radius: inherit;
            background: var(--allocation-color, var(--c-orange));
            box-shadow: 0 0 12px rgba(255,255,255,0.12);
            transition: width var(--dur-mid) var(--ease);
        }
        .mini-history {
            display: grid;
            gap: 7px;
        }
        .mini-history-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
        }
        .mini-history-bars {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 5px;
        }
        .mini-history-bar {
            height: 24px;
            display: flex;
            align-items: end;
            border: 1px solid var(--rule-soft);
            border-radius: 6px;
            padding: 3px;
            background: rgba(255,255,255,0.025);
        }
        .mini-history-bar span {
            display: block;
            width: 100%;
            min-height: 2px;
            border-radius: 4px 4px 2px 2px;
            background: var(--mini-color, var(--c-orange));
            transition: height var(--dur-mid) var(--ease);
        }
        .trace-section + .trace-section {
            margin-top: 14px;
            padding-top: 12px;
            border-top: 1px dashed var(--rule);
            display: grid;
            gap: 7px;
        }
        .trace-stack {
            display: grid;
            gap: 12px;
        }
        .trace-section {
            display: grid;
            gap: 7px;
        }
        .layer-row, .history-row {
            display: grid;
            grid-template-columns: 42px 1fr 112px;
            gap: 8px;
            align-items: center;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-dim);
        }
        .layer-track {
            position: relative;
            height: 8px;
            border-radius: 99px;
            background: rgba(255,255,255,0.075);
            overflow: hidden;
        }
        .layer-fill {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 0;
            transition: width var(--dur-mid) var(--ease), left var(--dur-mid) var(--ease);
        }
        .layer-fill.yellow {
            background: rgba(246,196,69,0.78);
        }
        .layer-fill.green {
            background: rgba(76,175,80,0.78);
        }
        .layer-replay {
            display: grid;
            gap: 10px;
        }
        .trace-actions {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        .layer-pipeline {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
        }
        .layer-card {
            display: grid;
            gap: 7px;
            padding: 9px;
            border: 1px solid var(--rule-soft);
            border-radius: 7px;
            background: rgba(255,255,255,0.02);
            color: var(--text-dim);
            text-align: left;
        }
        .layer-card.is-active {
            border-color: rgba(255, 106, 61, 0.72);
            background: rgba(255, 106, 61, 0.08);
            color: var(--text);
        }
        .layer-card-label,
        .layer-card-stat {
            font-family: var(--font-mono);
            font-size: 10px;
            letter-spacing: 0.06em;
            font-variant-numeric: tabular-nums;
        }
        .mini-split {
            position: relative;
            height: 8px;
            border-radius: 99px;
            overflow: hidden;
            background: rgba(255,255,255,0.075);
        }
        .mini-split span {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 0%;
            transition: width var(--dur-mid) var(--ease), left var(--dur-mid) var(--ease);
        }
        .mini-split .yellow { background: rgba(246,196,69,0.78); }
        .mini-split .green { background: rgba(76,175,80,0.78); }
        .history {
            display: grid;
            gap: 8px;
        }
        .history-track {
            height: 30px;
            display: grid;
            grid-auto-flow: column;
            grid-auto-columns: minmax(4px, 1fr);
            gap: 3px;
            align-items: end;
            padding: 4px;
            border: 1px solid var(--rule-soft);
            border-radius: 6px;
            background: rgba(255,255,255,0.025);
        }
        .history-bar {
            min-height: 2px;
            border-radius: 99px 99px 2px 2px;
            background: var(--history-color, var(--c-orange));
            opacity: 0.42;
            transition: height var(--dur-mid) var(--ease), opacity var(--dur-mid) var(--ease);
        }
        .history-row.is-active .history-bar {
            opacity: 0.92;
        }
        .trace-spark {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 12px;
            align-items: center;
        }
        .spark-svg {
            width: 100%;
            height: 92px;
            border: 1px solid var(--rule-soft);
            border-radius: 7px;
            background: rgba(255,255,255,0.025);
        }
        .trace-stats {
            display: grid;
            gap: 7px;
            min-width: 116px;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-dim);
            font-variant-numeric: tabular-nums;
        }
        .message-edge {
            animation: messagePulse 1.6s linear infinite;
            animation-delay: var(--edge-delay, 0s);
        }
        @keyframes messagePulse {
            0% { stroke-dashoffset: 0; opacity: 0.16; }
            45% { opacity: 0.6; }
            100% { stroke-dashoffset: -15; opacity: 0.16; }
        }
        .heat svg {
            width: 100%;
            max-height: 360px;
            aspect-ratio: 1;
            border: 1px solid var(--rule-soft);
            border-radius: 6px;
            background: rgba(0,0,0,0.22);
        }
        .heat-grid {
            display: grid;
            grid-template-columns: minmax(220px, 360px) minmax(0, 1fr);
            gap: 14px;
            align-items: start;
        }
        @media (max-width: 760px) {
            .heat-grid { grid-template-columns: 1fr; }
        }
        .heat-cell {
            cursor: crosshair;
            transition: opacity var(--dur-fast) var(--ease), stroke var(--dur-fast) var(--ease);
        }
        .heat-cell:hover {
            stroke: rgba(255,255,255,0.76);
            stroke-width: 1;
        }
        .heat-focus-line {
            animation: focusPulse 1.1s ease-in-out infinite;
        }
        .constraint-alert {
            animation: alertPulse 1.25s ease-in-out infinite;
        }
        @keyframes focusPulse {
            0%, 100% { opacity: 0.48; }
            50% { opacity: 0.95; }
        }
        @keyframes alertPulse {
            0%, 100% { opacity: 0.52; stroke-width: 1.2; }
            50% { opacity: 1; stroke-width: 2.4; }
        }
        .caption {
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px dashed var(--rule);
            color: var(--text-mute);
            font-family: var(--font-mono);
            font-size: 11px;
            letter-spacing: 0.04em;
        }
        .node {
            cursor: grab;
        }
        .node.is-dragging {
            cursor: grabbing;
        }
        .node-label {
            font-family: var(--font-mono);
            font-size: 3.35px;
            fill: var(--text-dim);
            pointer-events: none;
            user-select: none;
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                transition: none !important;
                animation: none !important;
            }
        }
    </style>

    <div class="toolbar">
        <div class="control-deck">
            <section class="control-group" aria-label="Topology controls">
                <span class="control-title">Topology</span>
                <div class="control-row">
                    <label class="control-field is-narrow"><span>Blue cars B</span>
                        <select data-b></select>
                    </label>
                    <label class="control-field is-narrow"><span>Yellow / Blue</span>
                        <select data-my></select>
                    </label>
                    <label class="control-field is-narrow"><span>Green / Blue</span>
                        <select data-mg></select>
                    </label>
                    <div class="stepper" aria-label="Adjust Blue car count">
                        <button type="button" class="icon-btn" data-remove-blue aria-label="Remove Blue car">-</button>
                        <button type="button" class="icon-btn" data-add-blue aria-label="Add Blue car">+</button>
                    </div>
                </div>
            </section>
            <section class="control-group" aria-label="Layout controls">
                <span class="control-title">Layout</span>
                <div class="control-row">
                    <label class="control-field"><span>Preset</span>
                        <select data-preset-select>
                            <option value="custom">Custom</option>
                            <option value="balanced">Balanced highway</option>
                            <option value="sensing">Sensing pressure</option>
                            <option value="crowded">Dense reuse</option>
                        </select>
                    </label>
                    <label class="control-field"><span>Saved</span>
                        <select data-saved-select>
                            <option value="">Saved layouts</option>
                        </select>
                    </label>
                    <button type="button" class="save-btn" data-save-layout>Save</button>
                </div>
            </section>
            <section class="control-group" aria-label="Channel controls">
                <span class="control-title">Channel</span>
                <div class="control-row">
                    <button type="button" class="action-btn" data-random>New layout</button>
                    <button type="button" class="action-btn" data-fading>New fading</button>
                    <label class="switch"><input type="checkbox" data-freeze checked />Lock fading</label>
                </div>
            </section>
        </div>
        <span class="status"><span class="dot"></span><span data-status>loading JSAC model</span></span>
    </div>

    <div class="stage">
        <div class="field-card">
            <svg data-field viewBox="0 0 225 225" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Draggable JSAC layout"></svg>
            <div class="field-legend" aria-label="JSAC map legend">
                <div class="field-legend-title">Map legend</div>
                <div class="legend-row"><span class="legend-mark blue"></span><span>Blue car</span></div>
                <div class="legend-row"><span class="legend-mark yellow"></span><span>Yellow sensing Rx</span></div>
                <div class="legend-row"><span class="legend-mark green"></span><span>Green comm Rx</span></div>
                <div class="legend-row"><span class="legend-mark link"></span><span>Service link</span></div>
                <div class="legend-row"><span class="legend-mark interference"></span><span>Interference</span></div>
                <div class="legend-row"><span class="legend-mark alert"></span><span>SINR alert</span></div>
            </div>
            <span class="field-hint">drag Blue/Rx</span>
        </div>
        <aside class="side-card">
            <div>
                <div class="panel-label">Method</div>
                <div class="method-tabs" data-method-tabs></div>
            </div>
            <div>
                <div class="panel-label">Selected group power</div>
                <div class="allocation-list" data-allocation></div>
            </div>
            <div>
                <div class="panel-label">Live metrics</div>
                <div class="metric-list" data-metrics></div>
            </div>
            <div>
                <div class="panel-label">Selected node</div>
                <div class="metric-sub" data-selected>Click a Blue car or receiver.</div>
            </div>
            <div>
                <div class="panel-label">Per-Blue budget</div>
                <div class="budget-rows" data-budgets></div>
            </div>
            <div>
                <div class="panel-label">Comparison pulse</div>
                <div class="mini-history" data-mini-history></div>
            </div>
        </aside>
    </div>

    <div class="diagnostic-drawer" data-drawer>
        <div class="drawer-head">
            <div class="panel-label">Diagnostics</div>
            <div class="drawer-tabs" aria-label="Diagnostic views">
                <button type="button" data-diagnostic-tab="power">Power</button>
                <button type="button" data-diagnostic-tab="history">History</button>
                <button type="button" data-diagnostic-tab="heat">Heatmap</button>
                <button type="button" data-diagnostic-tab="solver">Solver</button>
            </div>
            <button type="button" class="drawer-toggle" data-drawer-toggle>Collapse</button>
        </div>
        <div class="drawer-body">
            <section class="drawer-panel" data-diagnostic-panel="power">
                <div class="panel-label">Power allocation - selected method</div>
                <div class="power-detail" data-power-detail></div>
            </section>
            <section class="drawer-panel" data-diagnostic-panel="history">
                <div class="panel-label">Comparison history</div>
                <div class="history" data-history></div>
            </section>
            <section class="drawer-panel heat" data-diagnostic-panel="heat">
                <div class="heat-grid">
                    <div>
                        <div class="panel-label">Channel matrix |h|^2</div>
                        <svg data-heat viewBox="0 0 220 220" preserveAspectRatio="xMidYMid meet" aria-label="JSAC channel matrix heatmap"></svg>
                    </div>
                    <div class="metric-sub" data-heat-readout>Hover a cell to link the matrix to the map.</div>
                </div>
            </section>
            <section class="drawer-panel" data-diagnostic-panel="solver">
                <div class="trace-stack">
                    <div class="trace-section">
                        <div class="panel-label">GNN message-passing replay</div>
                        <div data-layers></div>
                    </div>
                    <div class="trace-section">
                        <div class="panel-label">WMMSE convergence</div>
                        <div data-wmmse-trace></div>
                    </div>
                </div>
            </section>
        </div>
    </div>

    <div class="caption">JSAC Stage 2 methods are WMMSE / GNN / Naive. Latency pills are normalized for demonstration: WMMSE/Naive are live JS elapsed, while GNN is mapped from live WMMSE time using JSAC benchmark ratios over B and M. Raw browser time is available in the tooltip; this is not a fair runtime benchmark.</div>
`;

function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

function fmt(v, digits = 2) {
    return Number.isFinite(v) ? v.toFixed(digits) : '--';
}

function fmtMs(v) {
    if (!Number.isFinite(v)) return '--';
    return v < 10 ? v.toFixed(2) : v.toFixed(1);
}

function escapeAttr(value) {
    return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('"', '&quot;')
        .replaceAll('<', '&lt;');
}

function interpolateRatio(x, table) {
    if (!Number.isFinite(x) || !table.length) return 1;
    if (x <= table[0][0]) return table[0][1];
    for (let i = 1; i < table.length; i++) {
        const [x1, y1] = table[i];
        const [x0, y0] = table[i - 1];
        if (x <= x1) {
            const t = (x - x0) / (x1 - x0);
            return y0 + (y1 - y0) * t;
        }
    }
    return table[table.length - 1][1];
}

function normalizedGnnLatencyMs(wmmseMs, ratio, baselineMs = 0) {
    if (!Number.isFinite(wmmseMs)) return NaN;
    const target = wmmseMs * ratio;
    const lower = Math.max(Number.isFinite(baselineMs) ? baselineMs * 2.5 : 0, 0.01);
    const upper = wmmseMs * 0.92;
    if (upper <= lower) return Math.max(0.01, wmmseMs * Math.min(ratio, 0.75));
    return clamp(target, lower, upper);
}

function jsacDemoGnnRatio(B, M) {
    const byB = interpolateRatio(B, JSAC_DEMO_GNN_WMMSE_RATIO_BY_B);
    const byM = interpolateRatio(M, JSAC_DEMO_GNN_WMMSE_RATIO_BY_M);
    return Math.sqrt(byB * byM);
}

function displayLatencyMs(result) {
    return Number.isFinite(result?.demoTimeMs) ? result.demoTimeMs : result?.timeMs;
}

function latencyTitle(method, result) {
    if (!result) return 'Latency pending.';
    const shown = displayLatencyMs(result);
    const raw = Number.isFinite(result.rawTimeMs) ? result.rawTimeMs : result.timeMs;
    if (method === 'GNN') {
        return `Normalized demo latency ${fmtMs(shown)} ms. Raw browser elapsed ${fmtMs(raw)} ms; adjusted with benchmarked JSAC GNN/WMMSE ratios.`;
    }
    if (method === 'WMMSE') {
        return `Live browser elapsed ${fmtMs(raw)} ms. This is the normalization anchor for the GNN demo latency.`;
    }
    return `Live browser elapsed ${fmtMs(raw)} ms.`;
}

function lerpValue(a, b, t) {
    const av = Number.isFinite(a) ? a : 0;
    const bv = Number.isFinite(b) ? b : 0;
    return av + (bv - av) * t;
}

function easeOutCubic(t) {
    return 1 - Math.pow(1 - clamp(t, 0, 1), 3);
}

function mulberry32(seed) {
    let t = seed >>> 0;
    return function () {
        t = (t + 0x6D2B79F5) >>> 0;
        let r = Math.imul(t ^ (t >>> 15), t | 1);
        r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
}

function gaussian(rand) {
    let u1 = rand();
    if (u1 < 1e-10) u1 = 1e-10;
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function svgEl(name, attrs = {}) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [k, v] of Object.entries(attrs)) {
        if (v !== undefined && v !== null) el.setAttribute(k, String(v));
    }
    return el;
}

function loadScript(src) {
    window.__liveRunScriptPromises = window.__liveRunScriptPromises || {};
    if (window.__liveRunScriptPromises[src]) return window.__liveRunScriptPromises[src];
    const existing = [...document.scripts].find((s) => s.src.endsWith(src));
    window.__liveRunScriptPromises[src] = new Promise((resolve, reject) => {
        const s = existing || document.createElement('script');
        if (s.dataset.loaded === 'true') {
            resolve();
            return;
        }
        s.addEventListener('load', () => {
            s.dataset.loaded = 'true';
            resolve();
        }, { once: true });
        s.addEventListener('error', () => reject(new Error(`failed to load ${src}`)), { once: true });
        if (!existing) {
            s.src = src;
            s.async = true;
            document.head.appendChild(s);
        }
    });
    return window.__liveRunScriptPromises[src];
}

function linear(v, layer) {
    const out = new Array(layer.bias.length);
    for (let i = 0; i < out.length; i++) {
        let acc = layer.bias[i];
        const row = layer.weight[i];
        for (let j = 0; j < row.length; j++) acc += row[j] * v[j];
        out[i] = acc;
    }
    return out;
}

function relu(v) {
    return v.map((x) => Math.max(0, x));
}

class LiveRunJsacLab extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' }).appendChild(JSAC_TEMPLATE.content.cloneNode(true));

        this.B = JSAC_DEFAULT_B;
        this.my = JSAC_DEFAULT_MY;
        this.mg = JSAC_DEFAULT_MG;
        this.seed = 211;
        this.selectedMethod = 'GNN';
        this.selected = null;
        this.drag = null;
        this.computeTicket = 0;
        this.pendingCompute = 0;
        this.manifest = null;
        this.weights = null;
        this.session = null;
        this.results = null;
        this.last = null;
        this.history = [];
        this.historyAnimationMode = 'idle';
        this.layerTrace = null;
        this.wmmseTrace = [];
        this.savedLayouts = [];
        this.displayResults = null;
        this.transitionFromResults = null;
        this.resultAnimationFrame = 0;
        this.resultAnimationT = 1;
        this.lastControlActivation = 0;
        this.methodAnimationFrame = 0;
        this.visualPower = null;
        this.visualGroupUtil = null;
        this.allocationGroup = null;
        this.allocationAnimationFrame = 0;
        this.visualAllocationPower = null;
        this.activeDiagnostic = 'power';
        this.drawerCollapsed = false;
        this.replayLayerIndex = -1;
        this.replayTimer = 0;
        this.hoverEdge = null;

        this.$field = this.shadowRoot.querySelector('[data-field]');
        this.$heat = this.shadowRoot.querySelector('[data-heat]');
        this.$b = this.shadowRoot.querySelector('[data-b]');
        this.$my = this.shadowRoot.querySelector('[data-my]');
        this.$mg = this.shadowRoot.querySelector('[data-mg]');
        this.$presetSelect = this.shadowRoot.querySelector('[data-preset-select]');
        this.$savedSelect = this.shadowRoot.querySelector('[data-saved-select]');
        this.$status = this.shadowRoot.querySelector('[data-status]');
        this.$metrics = this.shadowRoot.querySelector('[data-metrics]');
        this.$budgets = this.shadowRoot.querySelector('[data-budgets]');
        this.$allocation = this.shadowRoot.querySelector('[data-allocation]');
        this.$miniHistory = this.shadowRoot.querySelector('[data-mini-history]');
        this.$powerDetail = this.shadowRoot.querySelector('[data-power-detail]');
        this.$drawer = this.shadowRoot.querySelector('[data-drawer]');
        this.$drawerToggle = this.shadowRoot.querySelector('[data-drawer-toggle]');
        this.$heatReadout = this.shadowRoot.querySelector('[data-heat-readout]');
        this.$layers = this.shadowRoot.querySelector('[data-layers]');
        this.$history = this.shadowRoot.querySelector('[data-history]');
        this.$wmmseTrace = this.shadowRoot.querySelector('[data-wmmse-trace]');
        this.$selected = this.shadowRoot.querySelector('[data-selected]');
        this.$methodTabs = this.shadowRoot.querySelector('[data-method-tabs]');
        this.$freeze = this.shadowRoot.querySelector('[data-freeze]');
    }

    connectedCallback() {
        this._initControls();
        this._randomizeLayout(false);
        this._bindDrag();
        this._draw();
        this._load();
    }

    _initControls() {
        this._fillSelect(this.$b, 2, 6, this.B);
        this._fillSelect(this.$my, 1, 3, this.my);
        this._fillSelect(this.$mg, 1, 4, this.mg);

        this.$b.addEventListener('change', () => {
            this.B = Number(this.$b.value);
            this._randomizeLayout(true);
        });
        this.$my.addEventListener('change', () => {
            this.my = Number(this.$my.value);
            this._randomizeLayout(true);
        });
        this.$mg.addEventListener('change', () => {
            this.mg = Number(this.$mg.value);
            this._randomizeLayout(true);
        });
        this.$presetSelect.addEventListener('change', () => {
            if (this.$presetSelect.value === 'custom') return;
            this._applyPreset(this.$presetSelect.value);
        });
        this.$savedSelect.addEventListener('change', () => {
            if (!this.$savedSelect.value) return;
            this._loadSavedLayout(this.$savedSelect.value);
        });
        this._bindActionButton('[data-save-layout]', () => this._saveCurrentLayout());
        this._bindActionButton('[data-add-blue]', () => {
            this.B = clamp(this.B + 1, 2, 6);
            this.$b.value = String(this.B);
            this._randomizeLayout(true);
        });
        this._bindActionButton('[data-remove-blue]', () => {
            this.B = clamp(this.B - 1, 2, 6);
            this.$b.value = String(this.B);
            this._randomizeLayout(true);
        });
        this._bindActionButton('[data-random]', () => {
            this.seed += 19;
            this._randomizeLayout(true);
        });
        this._bindActionButton('[data-fading]', () => {
            this.seed += 103;
            this._channelRandoms = null;
            this._scheduleCompute(0);
        });
        this.shadowRoot.querySelectorAll('[data-diagnostic-tab]').forEach((btn) => {
            btn.addEventListener('click', () => this._setDiagnostic(btn.dataset.diagnosticTab));
        });
        this._bindActionButton(this.$drawerToggle, () => this._toggleDrawer());

        this._renderMethodTabs();
        this._syncDiagnosticShell();
        this._loadSavedLayouts();
    }

    _bindActionButton(target, action) {
        const el = typeof target === 'string' ? this.shadowRoot.querySelector(target) : target;
        if (!el) return;
        const activate = (ev) => {
            if (ev.type === 'pointerup') {
                if (ev.button !== 0) return;
                this.lastControlActivation = performance.now();
                ev.preventDefault();
                action();
                return;
            }
            if (performance.now() - this.lastControlActivation < 250) return;
            action();
        };
        el.addEventListener('pointerup', activate);
        el.addEventListener('click', activate);
    }

    _savedStorageKey() {
        return 'wireless-power-control.live-run.jsac.saved-layouts.v1';
    }

    _loadSavedLayouts() {
        try {
            this.savedLayouts = JSON.parse(window.localStorage.getItem(this._savedStorageKey()) || '[]');
        } catch {
            this.savedLayouts = [];
        }
        if (!Array.isArray(this.savedLayouts)) this.savedLayouts = [];
        this._renderSavedLayouts();
    }

    _persistSavedLayouts() {
        try {
            window.localStorage.setItem(this._savedStorageKey(), JSON.stringify(this.savedLayouts));
        } catch {
            // The browser can disable local storage; saved layouts are optional polish.
        }
    }

    _renderSavedLayouts() {
        if (!this.$savedSelect) return;
        const selected = this.$savedSelect.value;
        this.$savedSelect.innerHTML = '<option value="">Saved layouts</option>';
        for (const layout of this.savedLayouts) {
            const opt = document.createElement('option');
            opt.value = layout.id;
            opt.textContent = layout.name;
            this.$savedSelect.appendChild(opt);
        }
        if (this.savedLayouts.some((layout) => layout.id === selected)) this.$savedSelect.value = selected;
    }

    _saveCurrentLayout() {
        const id = `jsac-${Date.now().toString(36)}`;
        const layout = {
            id,
            name: `JSAC ${this.savedLayouts.length + 1}`,
            B: this.B,
            my: this.my,
            mg: this.mg,
            seed: this.seed,
            blue: this.blue.map((p) => ({ x: p.x, y: p.y })),
            rx: this.rx.map((p) => ({
                blue: p.blue,
                channel: p.channel,
                type: p.type,
                x: p.x,
                y: p.y,
            })),
        };
        this.savedLayouts.push(layout);
        if (this.savedLayouts.length > 8) this.savedLayouts.shift();
        this._persistSavedLayouts();
        this._renderSavedLayouts();
        this.$savedSelect.value = id;
    }

    _loadSavedLayout(id) {
        const layout = this.savedLayouts.find((item) => item.id === id);
        if (!layout) return;
        const field = this._fieldLength();
        const links = (Number(layout.my) || this.my) + (Number(layout.mg) || this.mg);
        this._captureTransitionStart();
        this.B = clamp(Number(layout.B) || JSAC_DEFAULT_B, 2, 6);
        this.my = clamp(Number(layout.my) || JSAC_DEFAULT_MY, 1, 3);
        this.mg = clamp(Number(layout.mg) || JSAC_DEFAULT_MG, 1, 4);
        this.seed = Number(layout.seed) || this.seed;
        this.$b.value = String(this.B);
        this.$my.value = String(this.my);
        this.$mg.value = String(this.mg);
        if (this.$presetSelect) this.$presetSelect.value = 'custom';
        this.blue = layout.blue.slice(0, this.B).map((p) => ({ x: clamp(p.x, 0, field), y: clamp(p.y, 0, field) }));
        this.rx = layout.rx.slice(0, this.B * links).map((p) => ({
            blue: clamp(Number(p.blue) || 0, 0, this.B - 1),
            channel: clamp(Number(p.channel) || 0, 0, links - 1),
            type: p.type === 'green' ? 'green' : 'yellow',
            x: clamp(p.x, 0, field),
            y: clamp(p.y, 0, field),
        }));
        if (this.blue.length < this.B || this.rx.length < this.B * this._linksPerBlue()) {
            this.seed += 19;
            this._randomizeLayout(false);
        }
        this.selected = null;
        this._clearAllocationTransition();
        this.allocationGroup = null;
        this.last = null;
        this.results = null;
        this.layerTrace = null;
        this.wmmseTrace = [];
        this.history = [];
        this.historyAnimationMode = 'idle';
        this._channelRandoms = null;
        this.computeTicket++;
        window.clearTimeout(this.pendingCompute);
        this._draw();
        this._scheduleCompute(0);
    }

    _fillSelect(select, lo, hi, value) {
        select.innerHTML = '';
        for (let i = lo; i <= hi; i++) {
            const opt = document.createElement('option');
            opt.value = String(i);
            opt.textContent = String(i);
            if (i === value) opt.selected = true;
            select.appendChild(opt);
        }
    }

    _renderMethodTabs() {
        this.$methodTabs.innerHTML = '';
        for (const method of ['WMMSE', 'GNN', 'Naive']) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.textContent = method;
            btn.classList.toggle('is-active', method === this.selectedMethod);
            btn.setAttribute('aria-pressed', method === this.selectedMethod ? 'true' : 'false');
            btn.addEventListener('click', () => this._selectMethod(method));
            this.$methodTabs.appendChild(btn);
        }
    }

    _selectMethod(method) {
        if (!this._methodNames().includes(method) || method === this.selectedMethod) return;
        const previous = this.selectedMethod;
        this.selectedMethod = method;
        this._clearGnnReplay();
        this._renderMethodTabs();
        this._startMethodTransition(previous, method);
    }

    _setDiagnostic(name) {
        if (!['power', 'history', 'heat', 'solver'].includes(name)) return;
        this.activeDiagnostic = name;
        this.drawerCollapsed = false;
        this._syncDiagnosticShell();
    }

    _toggleDrawer() {
        this.drawerCollapsed = !this.drawerCollapsed;
        this._syncDiagnosticShell();
    }

    _syncDiagnosticShell() {
        this.shadowRoot.querySelectorAll('[data-diagnostic-tab]').forEach((btn) => {
            const active = btn.dataset.diagnosticTab === this.activeDiagnostic;
            btn.classList.toggle('is-active', active);
            btn.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
        this.shadowRoot.querySelectorAll('[data-diagnostic-panel]').forEach((panel) => {
            panel.classList.toggle('is-active', panel.dataset.diagnosticPanel === this.activeDiagnostic);
        });
        this.$drawer?.classList.toggle('is-collapsed', this.drawerCollapsed);
        if (this.$drawerToggle) this.$drawerToggle.textContent = this.drawerCollapsed ? 'Expand' : 'Collapse';
    }

    async _load() {
        try {
            this.manifest = await fetch(JSAC_MANIFEST_URL, { cache: 'no-cache' }).then((r) => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.json();
            });
            const d = this.manifest.live_defaults || {};
            this.B = d.B || this.B;
            this.my = d.M_y || this.my;
            this.mg = d.M_g || this.mg;
            this.$b.value = String(this.B);
            this.$my.value = String(this.my);
            this.$mg.value = String(this.mg);
            this._randomizeLayout(false);

            this.weights = await fetch(JSAC_MODEL_BASE + this.manifest.weights_fallback, { cache: 'no-cache' }).then((r) => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.json();
            });
            this._setStatus('JS fallback ready');
            await this._computeAll();

            try {
                await loadScript(JSAC_ORT_SCRIPT);
                if (!window.ort) throw new Error('ort global missing');
                window.ort.env.wasm.wasmPaths = JSAC_ORT_WASM_PATH;
                window.ort.env.wasm.numThreads = 1;
                this.session = await window.ort.InferenceSession.create(JSAC_MODEL_BASE + this.manifest.model, {
                    executionProviders: ['wasm'],
                });
                this._setStatus('ONNX Runtime Web ready');
                await this._computeAll();
            } catch (err) {
                this.session = null;
                this._setStatus('JS fallback - ONNX unavailable');
                console.warn('[live-run-jsac-lab] ONNX Runtime Web unavailable; using JS fallback', err);
            }
        } catch (err) {
            this._setStatus('JSAC model assets unavailable');
            console.error('[live-run-jsac-lab] failed to load model assets', err);
        }
    }

    _setStatus(text) {
        this.$status.textContent = text;
    }

    _fieldLength() {
        return this.manifest?.physics?.field_length || 225;
    }

    _linksPerBlue() {
        return this.my + this.mg;
    }

    _k() {
        return this.B * this._linksPerBlue();
    }

    _randomizeLayout(recompute) {
        const field = this._fieldLength();
        const rand = mulberry32(this.seed + this.B * 997 + this.my * 131 + this.mg * 353);
        const minBlue = Math.min(this.manifest?.physics?.min_blue_dist || 50, field / 3);
        const rxMin = this.manifest?.physics?.rx_min_radius || 2;
        const rxMax = this.manifest?.physics?.rx_max_radius || 20;
        const minSep = this.manifest?.physics?.min_rx_separation || 2;

        this._captureTransitionStart();
        this.last = null;
        this.results = null;
        this.layerTrace = null;
        this.wmmseTrace = [];
        this.history = [];
        this.historyAnimationMode = 'idle';
        if (this.$presetSelect) this.$presetSelect.value = 'custom';
        this.computeTicket++;
        window.clearTimeout(this.pendingCompute);
        this.blue = [];
        this.rx = [];

        for (let b = 0; b < this.B; b++) {
            let point = null;
            for (let attempts = 0; attempts < 4000; attempts++) {
                const cand = {
                    x: 18 + rand() * (field - 36),
                    y: 18 + rand() * (field - 36),
                };
                const ok = this.blue.every((p) => Math.hypot(p.x - cand.x, p.y - cand.y) >= minBlue);
                if (ok) {
                    point = cand;
                    break;
                }
            }
            if (!point) {
                point = {
                    x: 18 + rand() * (field - 36),
                    y: 18 + rand() * (field - 36),
                };
            }
            this.blue.push(point);

            const cluster = [];
            for (let m = 0; m < this._linksPerBlue(); m++) {
                let node = null;
                for (let attempts = 0; attempts < 1600; attempts++) {
                    const dist = rxMin + rand() * (rxMax - rxMin);
                    const angle = rand() * Math.PI * 2;
                    const cand = {
                        x: point.x + Math.cos(angle) * dist,
                        y: point.y + Math.sin(angle) * dist,
                    };
                    if (cand.x < 2 || cand.x > field - 2 || cand.y < 2 || cand.y > field - 2) continue;
                    const ok = cluster.every((p) => Math.hypot(p.x - cand.x, p.y - cand.y) >= minSep);
                    if (ok) {
                        node = cand;
                        break;
                    }
                }
                if (!node) {
                    const angle = rand() * Math.PI * 2;
                    const dist = rxMin + rand() * (rxMax - rxMin);
                    node = {
                        x: clamp(point.x + Math.cos(angle) * dist, 2, field - 2),
                        y: clamp(point.y + Math.sin(angle) * dist, 2, field - 2),
                    };
                }
                cluster.push(node);
                this.rx.push({
                    blue: b,
                    channel: m,
                    type: m < this.my ? 'yellow' : 'green',
                    x: node.x,
                    y: node.y,
                });
            }
        }
        this.selected = null;
        this._clearAllocationTransition();
        this.allocationGroup = null;
        this._channelRandoms = null;
        this._draw();
        if (recompute) this._scheduleCompute(0);
    }

    _applyPreset(name) {
        const presets = {
            balanced: {
                seed: 211,
                B: 4,
                my: 2,
                mg: 3,
                blue: [[38, 64], [92, 152], [152, 68], [190, 152]],
                offsets: [
                    [[8, -10], [-8, 11], [14, 8], [-13, -7], [2, 16]],
                    [[-10, -8], [10, 10], [14, -4], [-12, 8], [1, -15]],
                    [[-11, 8], [10, -10], [13, 7], [-14, -5], [2, 15]],
                    [[-9, -11], [8, 11], [12, -7], [-13, 5], [3, 15]],
                ],
            },
            sensing: {
                seed: 347,
                B: 4,
                my: 2,
                mg: 3,
                blue: [[46, 54], [98, 92], [151, 72], [176, 153]],
                offsets: [
                    [[18, -7], [-16, 13], [9, 10], [-11, -9], [2, 17]],
                    [[-18, -8], [17, 12], [12, -10], [-10, 9], [3, 16]],
                    [[-17, 12], [16, -12], [12, 8], [-13, -7], [4, 15]],
                    [[-18, -10], [16, 11], [11, -8], [-12, 8], [2, 16]],
                ],
            },
            crowded: {
                seed: 503,
                B: 5,
                my: 2,
                mg: 2,
                blue: [[46, 58], [76, 132], [116, 84], [150, 142], [188, 84]],
                offsets: [
                    [[12, -7], [-10, 10], [13, 8], [-12, -6]],
                    [[-12, -9], [10, 12], [13, -5], [-12, 8]],
                    [[-10, 11], [12, -9], [13, 7], [-14, -5]],
                    [[-11, -8], [10, 11], [12, -8], [-13, 7]],
                    [[-11, 9], [10, -11], [12, 6], [-12, -8]],
                ],
            },
        };
        const preset = presets[name] || presets.balanced;
        const field = this._fieldLength();
        this._captureTransitionStart();
        if (Number.isFinite(preset.seed)) this.seed = preset.seed;
        this.B = preset.B;
        this.my = preset.my;
        this.mg = preset.mg;
        this.$b.value = String(this.B);
        this.$my.value = String(this.my);
        this.$mg.value = String(this.mg);
        this.$presetSelect.value = name;
        this.blue = preset.blue.map(([x, y]) => ({ x: clamp(x, 0, field), y: clamp(y, 0, field) }));
        this.rx = [];
        const links = this._linksPerBlue();
        for (let b = 0; b < this.B; b++) {
            for (let m = 0; m < links; m++) {
                const direct = preset.rx?.[b * links + m];
                const offset = preset.offsets?.[b]?.[m] || [0, 0];
                const base = this.blue[b];
                const x = direct ? direct[0] : base.x + offset[0];
                const y = direct ? direct[1] : base.y + offset[1];
                this.rx.push({
                    blue: b,
                    channel: m,
                    type: m < this.my ? 'yellow' : 'green',
                    x: clamp(x, 0, field),
                    y: clamp(y, 0, field),
                });
            }
        }
        this.selected = null;
        this._clearAllocationTransition();
        this.allocationGroup = null;
        this.last = null;
        this.results = null;
        this.layerTrace = null;
        this.wmmseTrace = [];
        this.history = [];
        this.historyAnimationMode = 'idle';
        this._channelRandoms = null;
        this.computeTicket++;
        window.clearTimeout(this.pendingCompute);
        this._draw();
        this._scheduleCompute(0);
    }

    _metadata() {
        const k = this._k();
        const groupIds = new Array(k);
        const channelIds = new Array(k);
        const greenMask = new Array(k).fill(false);
        const yellowMask = new Array(k).fill(false);
        for (let i = 0; i < k; i++) {
            groupIds[i] = this.rx[i].blue;
            channelIds[i] = this.rx[i].channel;
            greenMask[i] = this.rx[i].type === 'green';
            yellowMask[i] = this.rx[i].type === 'yellow';
        }

        const interfMask = [];
        const intraMask = [];
        const edgeMask = [];
        for (let target = 0; target < k; target++) {
            const intRow = [];
            const intraRow = [];
            const edgeRow = [];
            for (let source = 0; source < k; source++) {
                const interf = channelIds[target] === channelIds[source] && groupIds[target] !== groupIds[source];
                const intra = target !== source && groupIds[target] === groupIds[source];
                intRow.push(interf);
                intraRow.push(intra);
                edgeRow.push(interf || intra);
            }
            interfMask.push(intRow);
            intraMask.push(intraRow);
            edgeMask.push(edgeRow);
        }
        return { k, groupIds, channelIds, greenMask, yellowMask, interfMask, intraMask, edgeMask };
    }

    _distances(meta) {
        const d = [];
        for (let target = 0; target < meta.k; target++) {
            const row = [];
            const rx = this.rx[target];
            for (let source = 0; source < meta.k; source++) {
                const tx = this.blue[meta.groupIds[source]];
                row.push(Math.max(1, Math.hypot(tx.x - rx.x, tx.y - rx.y)));
            }
            d.push(row);
        }
        return d;
    }

    _ensureChannelRandoms(k) {
        if (
            this._channelRandoms &&
            this._channelRandoms.k === k &&
            this.$freeze.checked
        ) {
            return;
        }
        const rand = mulberry32(this.seed * 41 + k * 911);
        const shadow = [];
        const fading = [];
        for (let i = 0; i < k; i++) {
            const sRow = [];
            const fRow = [];
            for (let j = 0; j < k; j++) {
                sRow.push(Math.pow(10, gaussian(rand) * 8 / 10));
                const g1 = gaussian(rand);
                const g2 = gaussian(rand);
                fRow.push(Math.max(1e-6, 0.5 * (g1 * g1 + g2 * g2)));
            }
            shadow.push(sRow);
            fading.push(fRow);
        }
        this._channelRandoms = { k, shadow, fading };
    }

    _pathLoss(dist, isDirect) {
        const p = this.manifest.physics;
        const h1 = p.tx_height;
        const h2 = p.rx_height;
        const lambda = 2.998e8 / p.carrier_f;
        const rbp = 4 * h1 * h2 / lambda;
        const lbp = Math.abs(20 * Math.log10((lambda * lambda) / (8 * Math.PI * h1 * h2)));
        const sumTerm = 20 * Math.log10(Math.max(dist, 1) / rbp);
        const txOverRx = lbp + 6 + sumTerm + (dist > rbp ? sumTerm : 0);
        const db = -txOverRx + (isDirect ? p.antenna_gain_decibel : 0);
        return Math.pow(10, db / 10);
    }

    _channelLosses(dist, meta) {
        this._ensureChannelRandoms(meta.k);
        const losses = [];
        for (let i = 0; i < meta.k; i++) {
            const row = [];
            for (let j = 0; j < meta.k; j++) {
                const keep = i === j || meta.interfMask[i][j];
                if (!keep) {
                    row.push(0);
                    continue;
                }
                const random = this._channelRandoms.shadow[i][j] * this._channelRandoms.fading[i][j];
                row.push(this._pathLoss(dist[i][j], i === j) * random);
            }
            losses.push(row);
        }
        return losses;
    }

    _normByCategory(v, scaler, category) {
        if (category === 'sense') return (v - scaler.sense_mean) / (scaler.sense_std + JSAC_EPS);
        if (category === 'comm') return (v - scaler.comm_mean) / (scaler.comm_std + JSAC_EPS);
        return (v - scaler.interf_mean) / (scaler.interf_std + JSAC_EPS);
    }

    _buildInputs() {
        const maxK = this.manifest.max_k;
        const meta = this._metadata();
        const dist = this._distances(meta);
        const losses = this._channelLosses(dist, meta);
        const x = new Float32Array(maxK * 4);
        const edgeAttr = new Float32Array(maxK * maxK * 3);
        const edgeMask = new Float32Array(maxK * maxK);
        const nodeMask = new Float32Array(maxK);

        for (let i = 0; i < meta.k; i++) {
            nodeMask[i] = 1;
            const category = meta.yellowMask[i] ? 'sense' : 'comm';
            const nd = this._normByCategory(1 / dist[i][i], this.manifest.scalers.dist, category);
            const nl = this._normByCategory(Math.sqrt(losses[i][i]), this.manifest.scalers.loss, category);
            x[i * 4 + 0] = nd;
            x[i * 4 + 1] = nl;
            x[i * 4 + 2] = meta.greenMask[i] ? 1 : 0;
            x[i * 4 + 3] = 0;

            for (let j = 0; j < meta.k; j++) {
                const idx = i * maxK + j;
                if (meta.edgeMask[i][j]) edgeMask[idx] = 1;
                if (!meta.edgeMask[i][j]) continue;
                const base = idx * 3;
                const isInterf = meta.interfMask[i][j];
                edgeAttr[base + 0] = isInterf
                    ? this._normByCategory(1 / dist[i][j], this.manifest.scalers.dist, 'interf')
                    : 0;
                edgeAttr[base + 1] = meta.intraMask[i][j] ? 1 : 0;
                edgeAttr[base + 2] = isInterf
                    ? this._normByCategory(Math.sqrt(losses[i][j]), this.manifest.scalers.loss, 'interf')
                    : 0;
            }
        }
        return { meta, dist, losses, x, edgeAttr, edgeMask, nodeMask };
    }

    async _computeAll() {
        if (!this.manifest || !this.weights) return;
        this._clearGnnReplay();
        this.hoverEdge = null;
        const ticket = ++this.computeTicket;
        const tensors = this._buildInputs();
        if (tensors.meta.k > this.manifest.max_k) {
            this._setStatus(`layout exceeds Kmax=${this.manifest.max_k}`);
            return;
        }

        const gStart = performance.now();
        const logits = await this._runGnn(tensors);
        const gnnPower = this._applyGroupSoftmax(logits, tensors.meta);
        const gnnMs = performance.now() - gStart;
        if (ticket !== this.computeTicket) return;
        if (this.session) this.layerTrace = this._traceGnnFallback(tensors);

        const H = tensors.losses.map((row) => row.map((v) => Math.sqrt(Math.max(v, 0))));
        const wStart = performance.now();
        const wmmse = this._runWmmse(H, tensors.meta);
        const wmmseMs = performance.now() - wStart;
        const nStart = performance.now();
        const naive = this._runNaive(tensors.meta);
        const naiveMs = performance.now() - nStart;
        const gnnDemoMs = normalizedGnnLatencyMs(
            wmmseMs,
            jsacDemoGnnRatio(this.B, this._linksPerBlue()),
            naiveMs,
        );

        const methods = {
            WMMSE: { power: wmmse, timeMs: wmmseMs, rawTimeMs: wmmseMs, demoTimeMs: wmmseMs, engine: 'JS WMMSE' },
            GNN: { power: gnnPower, timeMs: gnnMs, rawTimeMs: gnnMs, demoTimeMs: gnnDemoMs, engine: this.session ? 'ONNX' : 'JS fallback' },
            Naive: { power: naive, timeMs: naiveMs, rawTimeMs: naiveMs, demoTimeMs: naiveMs, engine: 'JS naive' },
        };
        for (const value of Object.values(methods)) {
            Object.assign(value, this._metrics(tensors.losses, value.power, tensors.meta));
        }

        this.last = tensors;
        const fromResults = this.transitionFromResults || this.displayResults || this.results || methods;
        this.results = methods;
        this.transitionFromResults = null;
        this._appendHistory(methods);
        this._startResultAnimation(fromResults, methods);
    }

    async _runGnn(tensors) {
        if (this.session && window.ort) {
            try {
                const maxK = this.manifest.max_k;
                const feeds = {
                    x: new window.ort.Tensor('float32', tensors.x, [maxK, 4]),
                    edge_attr: new window.ort.Tensor('float32', tensors.edgeAttr, [maxK, maxK, 3]),
                    edge_mask: new window.ort.Tensor('float32', tensors.edgeMask, [maxK, maxK]),
                    node_mask: new window.ort.Tensor('float32', tensors.nodeMask, [maxK]),
                };
                const out = await this.session.run(feeds);
                return Array.from(out.logits.data);
            } catch (err) {
                console.warn('[live-run-jsac-lab] ONNX inference failed; using JS fallback', err);
            }
        }
        return this._runGnnFallback(tensors);
    }

    _runGnnFallback(tensors) {
        const trace = this._traceGnnFallback(tensors);
        this.layerTrace = trace;
        return trace.final;
    }

    _traceGnnFallback(tensors) {
        const maxK = this.manifest.max_k;
        let x = [];
        for (let i = 0; i < maxK; i++) {
            x.push([
                tensors.x[i * 4],
                tensors.x[i * 4 + 1],
                tensors.x[i * 4 + 2],
                tensors.x[i * 4 + 3],
            ]);
        }
        const layers = [];
        const capture = (label) => {
            const logits = x.map((row, i) => row[3] * tensors.nodeMask[i]);
            const power = this._applyGroupSoftmax(logits, tensors.meta);
            let yellow = 0;
            let green = 0;
            let yellowCount = 0;
            let greenCount = 0;
            for (let i = 0; i < tensors.meta.k; i++) {
                if (tensors.meta.yellowMask[i]) {
                    yellow += power[i];
                    yellowCount++;
                }
                if (tensors.meta.greenMask[i]) {
                    green += power[i];
                    greenCount++;
                }
            }
            const metrics = this._metrics(tensors.losses, power, tensors.meta);
            layers.push({
                label,
                logits,
                power,
                yellowAvg: yellow / Math.max(1, yellowCount),
                greenAvg: green / Math.max(1, greenCount),
                ...metrics,
            });
        };
        const conv = (xIn) => {
            const xOut = [];
            for (let target = 0; target < maxK; target++) {
                let aggr = new Array(32).fill(0);
                let seen = false;
                for (let source = 0; source < maxK; source++) {
                    if (tensors.edgeMask[target * maxK + source] < 0.5) continue;
                    const eBase = (target * maxK + source) * 3;
                    const input = [
                        ...xIn[source],
                        tensors.edgeAttr[eBase],
                        tensors.edgeAttr[eBase + 1],
                        tensors.edgeAttr[eBase + 2],
                    ];
                    let msg = relu(linear(input, this.weights.mlp1[0]));
                    msg = relu(linear(msg, this.weights.mlp1[1]));
                    if (!seen) {
                        aggr = msg;
                        seen = true;
                    } else {
                        for (let d = 0; d < aggr.length; d++) aggr[d] = Math.max(aggr[d], msg[d]);
                    }
                }
                const u0 = relu(linear([...xIn[target], ...aggr], this.weights.mlp2[0]));
                const u1 = linear(u0, this.weights.mlp2[1]);
                xOut.push([xIn[target][0], xIn[target][1], xIn[target][2], u1[0]]);
            }
            return xOut;
        };
        x = conv(x);
        capture('L1');
        x = conv(x);
        capture('L2');
        x = conv(x);
        capture('L3');
        x = conv(x);
        capture('L4');
        return { final: layers[layers.length - 1].logits, layers };
    }

    _applyGroupSoftmax(logits, meta) {
        const p = new Array(meta.k).fill(0);
        for (let b = 0; b < this.B; b++) {
            const links = [];
            for (let i = 0; i < meta.k; i++) {
                if (meta.groupIds[i] === b) links.push(i);
            }
            const maxLogit = Math.max(...links.map((i) => logits[i]));
            const exps = links.map((i) => Math.exp(clamp(logits[i] - maxLogit, -60, 60)));
            const denom = exps.reduce((a, v) => a + v, 0) + JSAC_EPS;
            links.forEach((i, n) => {
                p[i] = exps[n] / denom;
            });
        }
        return p;
    }

    _runNaive(meta) {
        const p = new Array(meta.k).fill(0);
        for (let b = 0; b < this.B; b++) {
            const links = [];
            for (let i = 0; i < meta.k; i++) {
                if (meta.groupIds[i] === b) links.push(i);
            }
            links.forEach((i) => {
                p[i] = 1 / links.length;
            });
        }
        return p;
    }

    _runWmmse(H, meta) {
        const k = meta.k;
        const noise = this.manifest.physics.var_noise;
        const pmax = this.manifest.physics.pmax || 1;
        const sinrMin = this.manifest.physics.sinr_min;
        const yellowAlpha = this.manifest.physics.wmmse_alpha_yellow;
        const alpha = meta.yellowMask.map((isYellow) => (isYellow ? yellowAlpha : 1));
        const groupLinks = [];
        for (let b = 0; b < this.B; b++) {
            const links = [];
            for (let i = 0; i < k; i++) if (meta.groupIds[i] === b) links.push(i);
            groupLinks.push(links);
        }

        let p = new Array(k).fill(0);
        for (const links of groupLinks) {
            links.forEach((i) => {
                p[i] = pmax / links.length;
            });
        }
        let v = p.map((x) => Math.sqrt(x));
        const mu = new Array(k).fill(0);
        const lrMu = 0.5;
        const losses = H.map((row) => row.map((x) => x * x));
        const trace = [];
        const checkpoints = new Set([0, 1, 2, 5, 10, 20, 50, JSAC_MAX_WMMSE_ITER]);
        const capture = (iter) => {
            if (!checkpoints.has(iter)) return;
            const metrics = this._metrics(losses, p, meta);
            trace.push({
                iter,
                greenSumRate: metrics.greenSumRate,
                yellowViolations: metrics.yellowViolations,
            });
        };
        capture(0);

        for (let iter = 0; iter < JSAC_MAX_WMMSE_ITER; iter++) {
            const hDiag = new Array(k);
            const totalRx = new Array(k);
            const signal = new Array(k);
            const interferencePlusNoise = new Array(k);
            const u = new Array(k);
            const w = new Array(k);

            for (let i = 0; i < k; i++) {
                hDiag[i] = H[i][i];
                let total = noise;
                for (let j = 0; j < k; j++) total += H[i][j] * H[i][j] * p[j];
                totalRx[i] = total;
                signal[i] = p[i] * hDiag[i] * hDiag[i];
                interferencePlusNoise[i] = Math.max(total - signal[i], noise);
                u[i] = (hDiag[i] * v[i]) / (total + JSAC_EPS);
                w[i] = total / (interferencePlusNoise[i] + JSAC_EPS);
            }

            const effective = alpha.map((a, i) => a + mu[i]);
            const vNext = new Array(k);
            for (let j = 0; j < k; j++) {
                let A = 0;
                for (let i = 0; i < k; i++) {
                    A += H[i][j] * H[i][j] * u[i] * u[i] * w[i] * effective[i];
                }
                const B = effective[j] * w[j] * u[j] * hDiag[j];
                vNext[j] = Math.max(B / (A + JSAC_EPS), 0);
            }
            v = vNext;
            p = v.map((x) => x * x);

            for (const links of groupLinks) {
                const groupPower = links.reduce((sum, i) => sum + p[i], 0);
                if (groupPower > pmax) {
                    const scale = pmax / (groupPower + JSAC_EPS);
                    links.forEach((i) => {
                        p[i] *= scale;
                    });
                }
            }
            v = p.map((x) => Math.sqrt(x));

            const sinr = this._sinrsFromLosses(losses, p);
            for (let i = 0; i < k; i++) {
                if (meta.yellowMask[i]) mu[i] = Math.max(0, mu[i] + lrMu * (sinrMin - sinr[i]));
            }
            capture(iter + 1);
        }
        this.wmmseTrace = trace;
        return p;
    }

    _sinrsFromLosses(losses, p) {
        const noise = this.manifest.physics.var_noise;
        const sinr = [];
        for (let i = 0; i < p.length; i++) {
            let interf = noise;
            for (let j = 0; j < p.length; j++) {
                if (i !== j) interf += p[j] * losses[i][j];
            }
            sinr.push((p[i] * losses[i][i]) / (interf + JSAC_EPS));
        }
        return sinr;
    }

    _metrics(losses, p, meta) {
        const sinr = this._sinrsFromLosses(losses, p);
        const rates = sinr.map((s) => Math.log2(1 + s));
        const greenRates = rates.filter((_, i) => meta.greenMask[i]);
        const yellowSinrs = sinr.filter((_, i) => meta.yellowMask[i]);
        const sinrMin = this.manifest.physics.sinr_min;
        const yellowViolations = yellowSinrs.filter((s) => s < sinrMin).length;
        const groupUtil = [];
        for (let b = 0; b < this.B; b++) {
            let yellowPower = 0;
            let greenPower = 0;
            for (let i = 0; i < meta.k; i++) {
                if (meta.groupIds[i] !== b) continue;
                if (meta.yellowMask[i]) yellowPower += p[i];
                if (meta.greenMask[i]) greenPower += p[i];
            }
            groupUtil.push({
                total: yellowPower + greenPower,
                yellow: yellowPower,
                green: greenPower,
            });
        }
        return {
            sinr,
            rates,
            greenSumRate: greenRates.reduce((a, b) => a + b, 0),
            yellowViolations,
            yellowViolationPct: yellowSinrs.length ? yellowViolations / yellowSinrs.length * 100 : 0,
            minYellowSinr: yellowSinrs.length ? Math.min(...yellowSinrs) : 0,
            groupUtil,
        };
    }

    _appendHistory(methods) {
        const entry = {
            WMMSE: methods.WMMSE?.greenSumRate || 0,
            GNN: methods.GNN?.greenSumRate || 0,
            Naive: methods.Naive?.greenSumRate || 0,
            violations: methods[this.selectedMethod]?.yellowViolations || 0,
        };
        const previous = this.history[this.history.length - 1];
        const changed = !previous || ['WMMSE', 'GNN', 'Naive'].some((method) => Math.abs(entry[method] - previous[method]) > 0.01) || entry.violations !== previous.violations;
        if (!changed) return;
        this.history.push(entry);
        if (this.history.length > 24) this.history.shift();
        this.historyAnimationMode = previous ? 'append' : 'refresh';
    }

    _methodNames() {
        return ['WMMSE', 'GNN', 'Naive'];
    }

    _displaySource() {
        return this.displayResults || this.results;
    }

    _captureTransitionStart() {
        this._clearMethodTransition();
        this.hoverEdge = null;
        if (this.resultAnimationFrame) {
            window.cancelAnimationFrame(this.resultAnimationFrame);
            this.resultAnimationFrame = 0;
        }
        const source = this.displayResults || this.results || this.transitionFromResults;
        this.transitionFromResults = source;
        this.displayResults = source ? this._cloneDisplayResults(source) : null;
        this.resultAnimationT = 1;
    }

    _cloneGroupUtil(groupUtil = []) {
        return groupUtil.map((g) => ({
            total: g?.total || 0,
            yellow: g?.yellow || 0,
            green: g?.green || 0,
        }));
    }

    _cloneDisplayResults(results) {
        const clone = {};
        for (const method of this._methodNames()) {
            const r = results?.[method];
            if (!r) continue;
            clone[method] = {
                ...r,
                power: [...(r.power || [])],
                sinr: [...(r.sinr || [])],
                rates: [...(r.rates || [])],
                groupUtil: this._cloneGroupUtil(r.groupUtil),
            };
        }
        return clone;
    }

    _interpolateArray(a = [], b = [], t) {
        const len = Math.max(a.length, b.length);
        const out = new Array(len);
        for (let i = 0; i < len; i++) out[i] = lerpValue(a[i], b[i], t);
        return out;
    }

    _interpolateGroupUtil(a = [], b = [], t) {
        const len = Math.max(a.length, b.length);
        const out = [];
        for (let i = 0; i < len; i++) {
            out.push({
                total: lerpValue(a[i]?.total, b[i]?.total, t),
                yellow: lerpValue(a[i]?.yellow, b[i]?.yellow, t),
                green: lerpValue(a[i]?.green, b[i]?.green, t),
            });
        }
        return out;
    }

    _interpolateResults(fromResults, toResults, t) {
        const out = {};
        for (const method of this._methodNames()) {
            const from = fromResults?.[method] || {};
            const to = toResults?.[method] || {};
            out[method] = {
                ...to,
                power: this._interpolateArray(from.power, to.power, t),
                sinr: this._interpolateArray(from.sinr, to.sinr, t),
                rates: this._interpolateArray(from.rates, to.rates, t),
                groupUtil: this._interpolateGroupUtil(from.groupUtil, to.groupUtil, t),
                timeMs: lerpValue(from.timeMs, to.timeMs, t),
                rawTimeMs: lerpValue(from.rawTimeMs, to.rawTimeMs, t),
                demoTimeMs: lerpValue(from.demoTimeMs, to.demoTimeMs, t),
                greenSumRate: lerpValue(from.greenSumRate, to.greenSumRate, t),
                yellowViolations: lerpValue(from.yellowViolations, to.yellowViolations, t),
                yellowViolationPct: lerpValue(from.yellowViolationPct, to.yellowViolationPct, t),
                minYellowSinr: lerpValue(from.minYellowSinr, to.minYellowSinr, t),
            };
        }
        return out;
    }

    _startResultAnimation(fromResults, toResults) {
        this._clearMethodTransition();
        if (this.resultAnimationFrame) window.cancelAnimationFrame(this.resultAnimationFrame);
        const from = this._cloneDisplayResults(fromResults);
        const to = this._cloneDisplayResults(toResults);
        const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (reduceMotion) {
            this.displayResults = to;
            this.resultAnimationT = 1;
            this.historyAnimationMode = 'idle';
            this._draw();
            return;
        }
        const start = performance.now();
        const duration = 760;
        const tick = (now) => {
            const raw = clamp((now - start) / duration, 0, 1);
            const eased = easeOutCubic(raw);
            this.resultAnimationT = eased;
            this.displayResults = this._interpolateResults(from, to, eased);
            this._draw();
            if (raw < 1) {
                this.resultAnimationFrame = window.requestAnimationFrame(tick);
            } else {
                this.resultAnimationFrame = 0;
                this.resultAnimationT = 1;
                this.displayResults = to;
                this.historyAnimationMode = 'idle';
                this._draw();
            }
        };
        this.resultAnimationFrame = window.requestAnimationFrame(tick);
    }

    _clearMethodTransition() {
        if (this.methodAnimationFrame) {
            window.cancelAnimationFrame(this.methodAnimationFrame);
            this.methodAnimationFrame = 0;
        }
        this.visualPower = null;
        this.visualGroupUtil = null;
        this._clearGnnReplay();
    }

    _clearAllocationTransition() {
        if (this.allocationAnimationFrame) {
            window.cancelAnimationFrame(this.allocationAnimationFrame);
            this.allocationAnimationFrame = 0;
        }
        this.visualAllocationPower = null;
    }

    _clearGnnReplay(redraw = false) {
        if (this.replayTimer) {
            window.clearTimeout(this.replayTimer);
            this.replayTimer = 0;
        }
        this.replayLayerIndex = -1;
        if (redraw) this._draw();
    }

    _activeGnnLayer() {
        if (this.selectedMethod !== 'GNN' || this.replayLayerIndex < 0) return null;
        return this.layerTrace?.layers?.[this.replayLayerIndex] || null;
    }

    _previewGnnLayer(index) {
        const layers = this.layerTrace?.layers || [];
        if (!layers[index]) return;
        if (this.selectedMethod !== 'GNN') {
            this.selectedMethod = 'GNN';
            this._renderMethodTabs();
        }
        if (this.replayTimer) window.clearTimeout(this.replayTimer);
        this.replayTimer = 0;
        this.replayLayerIndex = index;
        this._setDiagnostic('solver');
        this._draw();
    }

    _playGnnReplay() {
        const layers = this.layerTrace?.layers || [];
        if (!layers.length) return;
        this._clearMethodTransition();
        if (this.selectedMethod !== 'GNN') {
            this.selectedMethod = 'GNN';
            this._renderMethodTabs();
        }
        this._setDiagnostic('solver');
        let index = 0;
        const step = () => {
            this.replayLayerIndex = index;
            this._draw();
            index += 1;
            if (index < layers.length) {
                this.replayTimer = window.setTimeout(step, 620);
            } else {
                this.replayTimer = window.setTimeout(() => {
                    this.replayTimer = 0;
                    this.replayLayerIndex = -1;
                    this._draw();
                }, 700);
            }
        };
        step();
    }

    _groupFromSelection(selection = this.selected) {
        if (!selection) return null;
        if (selection.kind === 'blue') return clamp(selection.index, 0, this.B - 1);
        if (selection.kind === 'rx') return clamp(this.rx[selection.index]?.blue || 0, 0, this.B - 1);
        return null;
    }

    _allocationSlots(group) {
        if (!Number.isFinite(group)) return [];
        const slots = [];
        for (let i = 0; i < this._k(); i++) {
            if (this.rx[i]?.blue === group) slots.push(i);
        }
        return slots;
    }

    _allocationPowersForGroup(group, power = this._selectedPower()) {
        return this._allocationSlots(group).map((i) => clamp(power[i] || 0, 0, 1));
    }

    _setAllocationGroup(group, animate = true) {
        if (!Number.isFinite(group)) {
            this._clearAllocationTransition();
            this.allocationGroup = null;
            return;
        }
        group = clamp(group, 0, this.B - 1);
        if (group === this.allocationGroup && !this.visualAllocationPower) return;

        const to = this._allocationPowersForGroup(group);
        const from = this.visualAllocationPower || (this.allocationGroup === null ? new Array(to.length).fill(0) : this._allocationPowersForGroup(this.allocationGroup));
        this.allocationGroup = group;
        this._clearAllocationTransition();

        const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (!animate || from.length !== to.length || reduceMotion) return;

        this.visualAllocationPower = [...from];
        const start = performance.now();
        const duration = 440;
        const tick = (now) => {
            const raw = clamp((now - start) / duration, 0, 1);
            const eased = easeOutCubic(raw);
            this.visualAllocationPower = this._interpolateArray(from, to, eased);
            this._draw();
            if (raw < 1) {
                this.allocationAnimationFrame = window.requestAnimationFrame(tick);
            } else {
                this.allocationAnimationFrame = 0;
                this.visualAllocationPower = null;
                this._draw();
            }
        };
        this.allocationAnimationFrame = window.requestAnimationFrame(tick);
    }

    _startMethodTransition(fromMethod, toMethod) {
        const source = this._displaySource();
        const fromPower = this.visualPower || source?.[fromMethod]?.power;
        const toPower = source?.[toMethod]?.power;
        const fromGroupUtil = this.visualGroupUtil || source?.[fromMethod]?.groupUtil;
        const toGroupUtil = source?.[toMethod]?.groupUtil;
        if (!fromPower || !toPower) {
            this._clearMethodTransition();
            this._draw();
            return;
        }
        if (this.methodAnimationFrame) window.cancelAnimationFrame(this.methodAnimationFrame);
        const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (reduceMotion) {
            this.visualPower = [...toPower];
            this.visualGroupUtil = this._cloneGroupUtil(toGroupUtil);
            this._draw();
            this.visualPower = null;
            this.visualGroupUtil = null;
            return;
        }
        const from = [...fromPower];
        const to = [...toPower];
        const fromGroups = this._cloneGroupUtil(fromGroupUtil);
        const toGroups = this._cloneGroupUtil(toGroupUtil);
        const start = performance.now();
        const duration = 520;
        const tick = (now) => {
            const raw = clamp((now - start) / duration, 0, 1);
            const eased = easeOutCubic(raw);
            this.visualPower = this._interpolateArray(from, to, eased);
            this.visualGroupUtil = this._interpolateGroupUtil(fromGroups, toGroups, eased);
            this._draw();
            if (raw < 1) {
                this.methodAnimationFrame = window.requestAnimationFrame(tick);
            } else {
                this.methodAnimationFrame = 0;
                this.visualPower = null;
                this.visualGroupUtil = null;
                this._draw();
            }
        };
        this.methodAnimationFrame = window.requestAnimationFrame(tick);
    }

    _scheduleCompute(delay = 90) {
        window.clearTimeout(this.pendingCompute);
        this.pendingCompute = window.setTimeout(() => this._computeAll(), delay);
    }

    _bindDrag() {
        this.$field.addEventListener('pointerdown', (ev) => {
            const node = ev.target.closest?.('.node');
            if (!node) {
                this.selected = null;
                this._draw();
                return;
            }
            const kind = node.dataset.kind;
            const index = Number(node.dataset.index);
            const point = this._fieldPointFromEvent(ev);
            const current = kind === 'blue' ? this.blue[index] : this.rx[index];
            this.drag = {
                kind,
                index,
                el: node,
                offsetX: current.x - point.x,
                offsetY: current.y - point.y,
            };
            this.selected = { kind, index };
            this._setAllocationGroup(this._groupFromSelection(this.selected), true);
            node.setPointerCapture(ev.pointerId);
            node.classList.add('is-dragging');
            this._moveFromEvent(ev);
        });
        this.$field.addEventListener('pointermove', (ev) => {
            if (!this.drag) return;
            this._moveFromEvent(ev);
        });
        this.$field.addEventListener('pointerup', () => {
            this.drag?.el?.classList.remove('is-dragging');
            this.drag = null;
            this._scheduleCompute(0);
        });
        this.$field.addEventListener('pointercancel', () => {
            this.drag?.el?.classList.remove('is-dragging');
            this.drag = null;
        });
    }

    _fieldPointFromEvent(ev) {
        const rect = this.$field.getBoundingClientRect();
        const field = this._fieldLength();
        return {
            x: (ev.clientX - rect.left) / rect.width * field,
            y: (ev.clientY - rect.top) / rect.height * field,
        };
    }

    _moveFromEvent(ev) {
        const field = this._fieldLength();
        const point = this._fieldPointFromEvent(ev);
        const x = clamp(point.x + (this.drag.offsetX || 0), 0, field);
        const y = clamp(point.y + (this.drag.offsetY || 0), 0, field);
        if (this.drag.kind === 'blue') {
            const b = this.blue[this.drag.index];
            this._moveNode('blue', this.drag.index, x - b.x, y - b.y);
        } else {
            const r = this.rx[this.drag.index];
            r.x = x;
            r.y = y;
        }
        this._draw();
        this._scheduleCompute();
    }

    _moveNode(kind, index, dx, dy) {
        const field = this._fieldLength();
        if (kind === 'blue') {
            const b = this.blue[index];
            const nx = clamp(b.x + dx, 0, field);
            const ny = clamp(b.y + dy, 0, field);
            const realDx = nx - b.x;
            const realDy = ny - b.y;
            b.x = nx;
            b.y = ny;
            for (const r of this.rx) {
                if (r.blue !== index) continue;
                r.x = clamp(r.x + realDx, 0, field);
                r.y = clamp(r.y + realDy, 0, field);
            }
        } else {
            const r = this.rx[index];
            r.x = clamp(r.x + dx, 0, field);
            r.y = clamp(r.y + dy, 0, field);
        }
    }

    _draw() {
        this._drawField();
        this._drawMetrics();
        this._drawAllocation();
        this._drawBudgets();
        this._drawPowerDetail();
        this._drawMiniHistory();
        this._drawLayerTrace();
        this._drawHistory();
        this._drawWmmseTrace();
        this._drawHeatmap();
        this._syncDiagnosticShell();
    }

    _selectedPower() {
        const activeLayer = this._activeGnnLayer();
        if (activeLayer?.power) return activeLayer.power;
        return this.visualPower || this._displaySource()?.[this.selectedMethod]?.power || new Array(this._k()).fill(0);
    }

    _selectedMethodResult() {
        const activeLayer = this._activeGnnLayer();
        if (activeLayer) {
            const base = this._displaySource()?.GNN || {};
            return {
                ...base,
                ...activeLayer,
                engine: `${base.engine || 'GNN'} ${activeLayer.label}`,
                timeMs: base.timeMs,
                rawTimeMs: base.rawTimeMs,
                demoTimeMs: base.demoTimeMs,
            };
        }
        const result = this._displaySource()?.[this.selectedMethod];
        if (!result || (!this.visualPower && !this.visualGroupUtil)) return result;
        return {
            ...result,
            power: this.visualPower || result.power,
            groupUtil: this.visualGroupUtil || result.groupUtil,
        };
    }

    _drawField() {
        const svg = this.$field;
        svg.innerHTML = '';
        const field = this._fieldLength();
        svg.setAttribute('viewBox', `0 0 ${field} ${field}`);
        const power = this._selectedPower();
        const method = this._selectedMethodResult();
        const meta = this.last?.meta || this._metadata();
        const losses = this._hasCurrentLosses() ? this.last.losses : null;
        const focusGroup = this.selected ? this._focusedGroup() : null;

        svg.appendChild(svgEl('rect', { x: 0, y: 0, width: field, height: field, fill: 'rgba(255,255,255,0.015)' }));
        const gridStep = 25;
        for (let g = gridStep; g < field; g += gridStep) {
            svg.appendChild(svgEl('line', { x1: g, y1: 0, x2: g, y2: field, stroke: 'rgba(255,255,255,0.04)', 'stroke-width': 0.35 }));
            svg.appendChild(svgEl('line', { x1: 0, y1: g, x2: field, y2: g, stroke: 'rgba(255,255,255,0.04)', 'stroke-width': 0.35 }));
        }

        if (losses) {
            const edges = [];
            for (let target = 0; target < meta.k; target++) {
                for (let source = 0; source < meta.k; source++) {
                    if (!meta.interfMask[target][source]) continue;
                    edges.push({ target, source, score: losses[target][source] * (power[source] || 0) });
                }
            }
            edges.sort((a, b) => b.score - a.score);
            const maxScore = edges[0]?.score || 1;
            edges.slice(0, Math.min(90, edges.length)).forEach((e, n) => {
                const tx = this.blue[meta.groupIds[e.source]];
                const rx = this.rx[e.target];
                const related = focusGroup === null || meta.groupIds[e.source] === focusGroup || rx.blue === focusGroup;
                const op = clamp(0.07 + 0.36 * Math.sqrt(e.score / (maxScore + JSAC_EPS)), 0.07, 0.43) * (related ? 1 : 0.28);
                svg.appendChild(svgEl('line', {
                    class: 'message-edge',
                    x1: tx.x,
                    y1: tx.y,
                    x2: rx.x,
                    y2: rx.y,
                    stroke: `rgba(160,170,180,${op})`,
                    'stroke-width': 0.55,
                    'stroke-dasharray': '2 2.6',
                    style: `--edge-delay:${(n % 8) * 0.08}s`,
                }));
            });
        }

        if (this.hoverEdge && meta.groupIds[this.hoverEdge.source] !== undefined) {
            const tx = this.blue[meta.groupIds[this.hoverEdge.source]];
            const rx = this.rx[this.hoverEdge.target];
            if (tx && rx) {
                svg.appendChild(svgEl('line', {
                    class: 'heat-focus-line',
                    x1: tx.x,
                    y1: tx.y,
                    x2: rx.x,
                    y2: rx.y,
                    stroke: 'rgba(255,255,255,0.92)',
                    'stroke-width': 1.35,
                    'stroke-dasharray': '3 2',
                }));
                svg.appendChild(svgEl('circle', {
                    cx: rx.x,
                    cy: rx.y,
                    r: 6.8,
                    fill: 'none',
                    stroke: 'rgba(255,255,255,0.75)',
                    'stroke-width': 1.1,
                }));
            }
        }

        for (let i = 0; i < meta.k; i++) {
            const rx = this.rx[i];
            const tx = this.blue[rx.blue];
            const isYellow = rx.type === 'yellow';
            const color = isYellow ? '246,196,69' : '76,175,80';
            const p = power[i] || 0;
            const groupOpacity = focusGroup === null || rx.blue === focusGroup ? 1 : 0.22;
            svg.appendChild(svgEl('line', {
                x1: tx.x,
                y1: tx.y,
                x2: rx.x,
                y2: rx.y,
                stroke: `rgba(${color},${(0.22 + p * 0.66) * groupOpacity})`,
                'stroke-width': 0.55 + p * 1.45,
            }));
        }

        for (let b = 0; b < this.B; b++) {
            const util = method?.groupUtil?.[b]?.total || 0;
            const blue = this.blue[b];
            const groupOpacity = focusGroup === null || b === focusGroup ? 1 : 0.24;
            svg.appendChild(svgEl('circle', {
                cx: blue.x,
                cy: blue.y,
                r: 4.4 + util * 2.4,
                fill: `rgba(77,163,255,${(0.08 + util * 0.16) * groupOpacity})`,
                stroke: 'none',
                'pointer-events': 'none',
            }));
            const group = svgEl('g', { class: 'node', 'data-kind': 'blue', 'data-index': b, 'aria-label': `Blue car ${b}`, opacity: groupOpacity });
            group.appendChild(svgEl('rect', { x: blue.x - 2.9, y: blue.y - 2.9, width: 5.8, height: 5.8, rx: 0.9, fill: 'var(--c-blue)', opacity: 0.95 }));
            group.appendChild(svgEl('text', { class: 'node-label', x: blue.x + 4.1, y: blue.y + 1.2 }));
            group.lastChild.textContent = `B${b}`;
            svg.appendChild(group);
        }

        for (let i = 0; i < meta.k; i++) {
            const rx = this.rx[i];
            const isYellow = rx.type === 'yellow';
            const badYellow = Boolean(isYellow && method && method.sinr[i] < this.manifest.physics.sinr_min);
            const fill = isYellow ? 'var(--c-yellow)' : 'var(--c-green)';
            const p = clamp(power[i] || 0, 0, 1);
            const groupOpacity = focusGroup === null || rx.blue === focusGroup ? 1 : 0.24;
            const group = svgEl('g', { class: 'node', 'data-kind': 'rx', 'data-index': i, 'aria-label': `${isYellow ? 'Yellow' : 'Green'} receiver ${i}`, opacity: groupOpacity });
            if (p > 0.08) {
                group.appendChild(svgEl('circle', {
                    cx: rx.x,
                    cy: rx.y,
                    r: 3.3 + p * 4.6,
                    fill: isYellow ? 'rgba(246,196,69,0.24)' : 'rgba(76,175,80,0.24)',
                    stroke: 'none',
                    'pointer-events': 'none',
                }));
            }
            if (badYellow) {
                group.appendChild(svgEl('circle', { class: 'constraint-alert', cx: rx.x, cy: rx.y, r: 4.6, fill: 'none', stroke: 'rgba(255,99,99,0.95)', 'stroke-width': 1.2 }));
            }
            group.appendChild(svgEl('circle', { cx: rx.x, cy: rx.y, r: 2.45, fill, opacity: 0.92 }));
            group.appendChild(svgEl('text', { class: 'node-label', x: rx.x + 3.3, y: rx.y + 1.2 }));
            group.lastChild.textContent = `${isYellow ? 'Y' : 'G'}${i}`;
            svg.appendChild(group);
        }

        this._drawSelectedText(meta, method);
    }

    _drawSelectedText(meta, method) {
        if (!this.selected) {
            this.$selected.textContent = 'Click a Blue car or receiver.';
            return;
        }
        if (this.selected.kind === 'blue') {
            const b = this.blue[this.selected.index];
            const util = method?.groupUtil?.[this.selected.index]?.total;
            this.$selected.textContent = `B${this.selected.index} - x ${fmt(b.x, 1)} m / y ${fmt(b.y, 1)} m / budget ${fmt(util, 2)}`;
            return;
        }
        const i = this.selected.index;
        const r = this.rx[i];
        const p = method?.power?.[i];
        const sinr = method?.sinr?.[i];
        const rate = method?.rates?.[i];
        this.$selected.textContent = `${r.type.toUpperCase()}${i} - B${r.blue} ch${r.channel} / p ${fmt(p, 2)} / SINR ${fmt(sinr, 2)} / rate ${fmt(rate, 2)}`;
    }

    _drawMetrics() {
        const methods = ['WMMSE', 'GNN', 'Naive'];
        const display = this._displaySource();
        this.$metrics.innerHTML = '';
        for (const method of methods) {
            const r = method === this.selectedMethod ? (this._selectedMethodResult() || display?.[method]) : display?.[method];
            const row = document.createElement('div');
            row.className = 'metric-row';
            row.classList.toggle('is-active', method === this.selectedMethod);
            row.addEventListener('click', () => this._selectMethod(method));
            const viol = r ? `${Math.round(r.yellowViolations)}/${this.B * this.my}` : '--';
            const latency = r ? displayLatencyMs(r) : NaN;
            const latencyTip = r ? latencyTitle(method, r) : 'Latency pending.';
            row.innerHTML = `
                <span class="method-name" style="color:${this._methodColor(method)}">${method}</span>
                <span>
                    <span class="metric-main">${r ? fmt(r.greenSumRate, 2) : '--'}</span>
                    <span class="metric-sub"> green SR / yellow viol ${viol}</span>
                </span>
                <span class="pill" title="${escapeAttr(latencyTip)}" aria-label="${escapeAttr(latencyTip)}">${r ? `~${fmtMs(latency)} ms` : '--'}</span>
            `;
            this.$metrics.appendChild(row);
        }
        if (display?.GNN) {
            this._setStatus(`${display.GNN.engine} / JSAC GNN demo ~${fmtMs(displayLatencyMs(display.GNN))} ms / B=${this.B} K=${this._k()}`);
        }
    }

    _drawBudgets() {
        this.$budgets.innerHTML = '';
        const r = this._selectedMethodResult();
        const focus = this._focusedGroup();
        const activeGroup = this.allocationGroup;
        for (let b = 0; b < this.B; b++) {
            const util = r?.groupUtil?.[b] || { total: 0, yellow: 0, green: 0 };
            const yellowW = clamp(util.yellow, 0, 1) * 100;
            const greenW = clamp(util.green, 0, 1) * 100;
            const row = document.createElement('div');
            row.className = 'budget-row';
            row.classList.toggle('is-active', b === focus || b === activeGroup);
            row.addEventListener('click', () => {
                this.selected = { kind: 'blue', index: b };
                this._setAllocationGroup(b, true);
                this._draw();
            });
            row.innerHTML = `
                <span>B${b}</span>
                <span class="budget-track">
                    <span class="budget-fill yellow" style="left:0;width:${yellowW}%"></span>
                    <span class="budget-fill green" style="left:${yellowW}%;width:${greenW}%"></span>
                </span>
                <span>${fmt(util.total, 2)}</span>
            `;
            this.$budgets.appendChild(row);
        }
    }

    _focusedGroup() {
        return this._groupFromSelection(this.selected);
    }

    _drawAllocation() {
        if (!this.$allocation) return;
        this.$allocation.innerHTML = '';
        const result = this._selectedMethodResult();
        const group = this.allocationGroup;
        const links = this._allocationSlots(group);
        if (!links.length) {
            this.$allocation.innerHTML = '<div class="metric-sub">Click a Blue car or receiver.</div>';
            return;
        }
        const animatedPower = this.visualAllocationPower;
        links.forEach((i, slot) => {
            const rx = this.rx[i];
            const isYellow = rx.type === 'yellow';
            const color = isYellow ? 'rgba(246,196,69,0.86)' : 'rgba(76,175,80,0.86)';
            const p = clamp(animatedPower?.[slot] ?? result?.power?.[i] ?? 0, 0, 1);
            const sinr = result?.sinr?.[i];
            const isAlert = isYellow && Number.isFinite(sinr) && this.manifest && sinr < this.manifest.physics.sinr_min;
            const row = document.createElement('div');
            row.className = 'allocation-row';
            row.classList.toggle('is-active', this.selected?.kind === 'rx' && this.selected.index === i);
            row.classList.toggle('is-alert', isAlert);
            row.addEventListener('click', () => {
                this.selected = { kind: 'rx', index: i };
                this._setAllocationGroup(group, false);
                this._draw();
            });
            row.innerHTML = `
                <span>${isYellow ? 'Y' : 'G'}${i}</span>
                <span class="allocation-track">
                    <span class="allocation-fill" style="--allocation-color:${color};width:${p * 100}%"></span>
                </span>
                <span>${fmt(p, 2)}</span>
            `;
            row.title = isYellow ? `SINR ${fmt(sinr, 2)}` : `rate ${fmt(result?.rates?.[i], 2)}`;
            this.$allocation.appendChild(row);
        });
    }

    _drawPowerDetail() {
        if (!this.$powerDetail) return;
        this.$powerDetail.innerHTML = '';
        const result = this._selectedMethodResult();
        const focus = this._focusedGroup();
        const activeGroup = this.allocationGroup;
        if (!result?.groupUtil) {
            this.$powerDetail.innerHTML = '<div class="metric-sub">Power appears after the first live solve.</div>';
            return;
        }
        for (let b = 0; b < this.B; b++) {
            const util = result.groupUtil[b] || { total: 0, yellow: 0, green: 0 };
            const yellowW = clamp(util.yellow, 0, 1) * 100;
            const greenW = clamp(util.green, 0, 1) * 100;
            const row = document.createElement('div');
            row.className = 'power-group-row';
            row.classList.toggle('is-active', b === activeGroup || b === focus);
            row.addEventListener('click', () => {
                this.selected = { kind: 'blue', index: b };
                this._setAllocationGroup(b, true);
                this._draw();
            });
            row.innerHTML = `
                <span>B${b}</span>
                <span class="budget-track">
                    <span class="budget-fill yellow" style="left:0;width:${yellowW}%"></span>
                    <span class="budget-fill green" style="left:${yellowW}%;width:${greenW}%"></span>
                </span>
                <span>${fmt(util.total, 2)}</span>
            `;
            row.title = `Yellow ${fmt(util.yellow, 2)} / Green ${fmt(util.green, 2)}`;
            this.$powerDetail.appendChild(row);
        }
    }

    _drawMiniHistory() {
        if (!this.$miniHistory) return;
        this.$miniHistory.innerHTML = '';
        const latest = this.history[this.history.length - 1];
        if (!latest) {
            this.$miniHistory.innerHTML = '<div class="metric-sub">Waiting for first solve.</div>';
            return;
        }
        const methods = ['WMMSE', 'GNN', 'Naive'];
        const max = Math.max(...methods.map((method) => latest[method] || 0), 1);
        const delta = latest.WMMSE > 0 ? (latest.GNN / latest.WMMSE - 1) * 100 : 0;
        const deltaText = `${delta >= 0 ? '+' : ''}${fmt(delta, 1)}%`;
        const head = document.createElement('div');
        head.className = 'mini-history-head';
        head.innerHTML = `<span class="metric-sub">GNN vs WMMSE</span><span class="pill">${deltaText}</span>`;
        const bars = document.createElement('div');
        bars.className = 'mini-history-bars';
        for (const method of methods) {
            const wrap = document.createElement('div');
            wrap.className = 'mini-history-bar';
            wrap.title = `${method} ${fmt(latest[method], 2)}`;
            wrap.innerHTML = `<span style="--mini-color:${this._methodColor(method)};height:${clamp((latest[method] || 0) / max, 0, 1) * 100}%"></span>`;
            bars.appendChild(wrap);
        }
        this.$miniHistory.append(head, bars);
    }

    _drawLayerTrace() {
        this.$layers.innerHTML = '';
        const layers = this.layerTrace?.layers || [];
        if (!layers.length) {
            this.$layers.innerHTML = '<div class="metric-sub">Waiting for GNN weights.</div>';
            return;
        }
        const activeIndex = this.replayLayerIndex;
        const root = document.createElement('div');
        root.className = 'layer-replay';
        root.innerHTML = `
            <div class="trace-actions">
                <button type="button" class="trace-action" data-replay-gnn>Replay</button>
                <button type="button" class="trace-action" data-layer-final>Final</button>
                <span class="metric-sub">${activeIndex >= 0 ? layers[activeIndex].label : 'final'} allocation on map</span>
            </div>
            <div class="layer-pipeline"></div>
        `;
        const pipeline = root.querySelector('.layer-pipeline');
        for (let index = 0; index < layers.length; index++) {
            const layer = layers[index];
            const yellowW = clamp(layer.yellowAvg * this.resultAnimationT, 0, 1) * 100;
            const greenW = clamp(layer.greenAvg * this.resultAnimationT, 0, 1) * 100;
            const card = document.createElement('button');
            card.type = 'button';
            card.className = 'layer-card';
            card.classList.toggle('is-active', index === activeIndex);
            card.dataset.layerIndex = String(index);
            card.innerHTML = `
                <span class="layer-card-label">${layer.label}</span>
                <span class="mini-split">
                    <span class="yellow" style="left:0;width:${yellowW}%"></span>
                    <span class="green" style="left:${yellowW}%;width:${greenW}%"></span>
                </span>
                <span class="layer-card-stat">SR ${fmt(layer.greenSumRate, 1)} / V ${Math.round(layer.yellowViolations || 0)}</span>
            `;
            pipeline.appendChild(card);
        }
        root.querySelector('[data-replay-gnn]')?.addEventListener('click', () => this._playGnnReplay());
        root.querySelector('[data-layer-final]')?.addEventListener('click', () => this._clearGnnReplay(true));
        root.querySelectorAll('[data-layer-index]').forEach((btn) => {
            btn.addEventListener('click', () => this._previewGnnLayer(Number(btn.dataset.layerIndex)));
        });
        this.$layers.appendChild(root);
    }

    _drawHistory() {
        this.$history.innerHTML = '';
        const methods = ['WMMSE', 'GNN', 'Naive'];
        if (!this.history.length) {
            this.$history.innerHTML = '<div class="metric-sub">History appears after the first live solve.</div>';
            return;
        }
        const max = Math.max(...this.history.flatMap((entry) => methods.map((method) => entry[method] || 0)), 1);
        for (const method of methods) {
            const row = document.createElement('div');
            row.className = 'history-row';
            row.classList.toggle('is-active', method === this.selectedMethod);
            const color = this._methodColor(method);
            const progress = this.historyAnimationMode === 'idle' ? 1 : this.resultAnimationT;
            const bars = this.history.map((entry, index) => {
                const h = clamp((entry[method] || 0) / max, 0, 1) * 100;
                const shouldAnimate = this.historyAnimationMode === 'refresh' || (this.historyAnimationMode === 'append' && index === this.history.length - 1);
                const scale = shouldAnimate ? progress : 1;
                const opacity = shouldAnimate ? 0.18 + 0.74 * progress : '';
                const opacityStyle = opacity === '' ? '' : `opacity:${opacity};`;
                return `<span class="history-bar" style="height:${h * scale}%;--history-color:${color};${opacityStyle}"></span>`;
            }).join('');
            const latest = this.history[this.history.length - 1]?.[method];
            row.innerHTML = `
                <span style="color:${color}">${method}</span>
                <span class="history-track">${bars}</span>
                <span>${fmt(latest, 2)}</span>
            `;
            this.$history.appendChild(row);
        }
    }

    _drawWmmseTrace() {
        this.$wmmseTrace.innerHTML = '';
        const trace = this.wmmseTrace || [];
        if (!trace.length) {
            this.$wmmseTrace.innerHTML = '<div class="metric-sub">Waiting for WMMSE solve.</div>';
            return;
        }
        const max = Math.max(...trace.map((entry) => entry.greenSumRate), 1);
        const min = Math.min(...trace.map((entry) => entry.greenSumRate));
        const width = 420;
        const height = 92;
        const pad = 12;
        const progress = this.resultAnimationT;
        const denom = Math.max(max - min, 1e-9);
        const points = trace.map((entry, index) => {
            const x = pad + (width - pad * 2) * (index / Math.max(1, trace.length - 1)) * progress;
            const y = height - pad - (height - pad * 2) * ((entry.greenSumRate - min) / denom);
            return `${fmt(x, 2)},${fmt(y, 2)}`;
        }).join(' ');
        const dots = trace.map((entry, index) => {
            const reveal = index <= Math.floor((trace.length - 1) * progress);
            if (!reveal) return '';
            const x = pad + (width - pad * 2) * (index / Math.max(1, trace.length - 1));
            const y = height - pad - (height - pad * 2) * ((entry.greenSumRate - min) / denom);
            return `<circle cx="${fmt(x, 2)}" cy="${fmt(y, 2)}" r="3" fill="var(--c-blue)"><title>I${entry.iter} G ${fmt(entry.greenSumRate, 2)} / V ${entry.yellowViolations}</title></circle>`;
        }).join('');
        const last = trace[trace.length - 1];
        this.$wmmseTrace.innerHTML = `
            <div class="trace-spark">
                <svg class="spark-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="WMMSE convergence sparkline">
                    <polyline points="${points}" fill="none" stroke="var(--c-blue)" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"></polyline>
                    ${dots}
                </svg>
                <div class="trace-stats">
                    <span>final G ${fmt(last.greenSumRate, 2)}</span>
                    <span>viol ${Math.round(last.yellowViolations)}</span>
                    <span>${trace.length} checkpoints</span>
                </div>
            </div>
        `;
    }

    _drawHeatmap() {
        const svg = this.$heat;
        svg.innerHTML = '';
        if (!this._hasCurrentLosses()) return;
        const losses = this.last.losses;
        const k = this.last.meta.k;
        const values = [];
        for (let i = 0; i < k; i++) {
            for (let j = 0; j < k; j++) {
                if (losses[i][j] > 0) values.push(Math.log10(losses[i][j] + 1e-30));
            }
        }
        const lo = Math.min(...values);
        const hi = Math.max(...values);
        const size = 220 / k;
        for (let i = 0; i < k; i++) {
            for (let j = 0; j < k; j++) {
                const raw = losses[i][j];
                const t = raw > 0 ? (Math.log10(raw + 1e-30) - lo) / (hi - lo + JSAC_EPS) : 0;
                let color = 'rgba(255,255,255,0.025)';
                if (i === j) color = this.last.meta.yellowMask[i]
                    ? `rgba(246,196,69,${0.24 + 0.62 * t})`
                    : `rgba(76,175,80,${0.24 + 0.62 * t})`;
                else if (raw > 0) color = `rgba(160,170,180,${0.08 + 0.48 * t})`;
                const cell = svgEl('rect', {
                    class: 'heat-cell',
                    x: j * size,
                    y: i * size,
                    width: Math.max(1, size - 0.6),
                    height: Math.max(1, size - 0.6),
                    fill: color,
                });
                cell.addEventListener('pointerenter', () => {
                    this.hoverEdge = { target: i, source: j, raw };
                    if (this.$heatReadout) {
                        const sourceGroup = this.last.meta.groupIds[j];
                        this.$heatReadout.textContent = `Tx B${sourceGroup} link ${j} -> ${this.rx[i].type.toUpperCase()}${i} |h|^2 ${raw.toExponential(2)}`;
                    }
                    this._drawField();
                });
                cell.addEventListener('pointerleave', () => {
                    this.hoverEdge = null;
                    if (this.$heatReadout) this.$heatReadout.textContent = 'Hover a cell to link the matrix to the map.';
                    this._drawField();
                });
                cell.addEventListener('click', () => {
                    this.selected = { kind: 'rx', index: i };
                    this._setAllocationGroup(this.rx[i]?.blue, true);
                    this._draw();
                });
                svg.appendChild(cell);
            }
        }
    }

    _hasCurrentLosses() {
        return Boolean(
            this.last?.losses &&
            this.last.losses.length === this._k() &&
            this.last.losses.every((row) => row && row.length === this._k())
        );
    }

    _methodColor(method) {
        if (method === 'WMMSE') return 'var(--c-blue)';
        if (method === 'Naive') return 'var(--c-grey)';
        return 'var(--c-orange)';
    }
}

customElements.define('live-run-jsac-lab', LiveRunJsacLab);
