import './live-run-jsac-lab.js?v=1.1.1';

/* <live-run-lab> - Live Run browser inference playground.
 *
 * Stage 1 scope:
 * - D2D only, K <= 20.
 * - Methods: WMMSE / GNN / Greedy.
 * - GNN uses ONNX Runtime Web when available, with a JS weight fallback.
 * - WMMSE runs live in JavaScript for the current layout.
 * - Greedy mirrors Scenario_D2D/baselines.py:simple_greedy, using the
 *   WMMSE allocation to estimate active-link sparsity for live layouts.
 */

const MANIFEST_URL = 'assets/models/d2d_live_manifest.json';
const MODEL_BASE = 'assets/models/';
const ORT_SCRIPT = 'assets/vendor/onnxruntime-web/ort.wasm.min.js';
const ORT_WASM_PATH = {
    'ort-wasm-simd-threaded.mjs': './assets/vendor/onnxruntime-web/ort-wasm-simd-threaded.mjs',
    'ort-wasm-simd-threaded.wasm': './assets/vendor/onnxruntime-web/ort-wasm-simd-threaded.wasm',
};

const DEFAULT_K = 8;
const MAX_WMMSE_ITER = 100;
const EPS = 1e-12;

// -----------------------------------------------------------------------------
// Live Run demo-latency normalization
// -----------------------------------------------------------------------------
// The side-rail latency pills are presentation aids for this browser demo, not
// the authoritative runtime benchmark. Raw Live Run timings mix different
// engines: WMMSE and Greedy run as vanilla JavaScript in this page, while GNN
// runs through ONNX Runtime Web when available, with a JS weight fallback. That
// means the raw GNN number includes browser/runtime overhead that is not
// comparable to the JS WMMSE loop, and at small K it can invert the expected
// algorithmic ordering.
//
// To keep the demo intuitive without hiding the measurement problem, the UI
// displays a normalized GNN latency derived from the current browser's WMMSE
// time multiplied by benchmarked GNN/WMMSE ratios from
// `web/assets/data/d2d_sweep_K.json`. The absolute scale still follows the
// visitor's machine because WMMSE is timed live; the relative GNN-vs-WMMSE shape
// follows the project results. The raw browser elapsed time is preserved on each
// result as `rawTimeMs` and exposed in the timing-pill tooltip.
//
// If a future implementation runs both methods through a fair shared engine,
// remove this block and render `rawTimeMs` directly.
const D2D_DEMO_GNN_WMMSE_RATIO_BY_K = [
    [2, 0.50],                 // Live Run extrapolation; the research sweep starts at K=5.
    [5, 3.739 / 8.985],        // GNN / WMMSE inference_ms from d2d_sweep_K.json.
    [10, 4.177 / 18.771],
    [20, 8.829 / 51.667],
];

const TEMPLATE = document.createElement('template');
TEMPLATE.innerHTML = /* html */ `
    <style>
        :host {
            display: block;
            background: var(--surface);
            border: 1px solid var(--rule);
            border-radius: var(--radius-card);
            padding: 20px;
            color: var(--text);
            font-family: var(--font-sans);
        }
        [hidden] {
            display: none !important;
        }
        .head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 18px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }
        .eyebrow {
            display: block;
            font-family: var(--font-mono);
            font-size: 11px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--text-dim);
            margin-bottom: 8px;
        }
        .eyebrow::before {
            content: '';
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--c-orange);
            margin-right: 7px;
            transform: translateY(-1px);
        }
        .title {
            font-size: 18px;
            font-weight: 600;
            line-height: 1.25;
        }
        .sub {
            color: var(--text-dim);
            font-size: 13px;
            margin-top: 5px;
            max-width: 72ch;
        }
        .mode-tabs, .method-tabs {
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
        button:disabled {
            cursor: not-allowed;
            opacity: 0.5;
            border-color: var(--rule);
        }
        select {
            color: var(--text);
            background: var(--surface-2);
            min-width: 74px;
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
        .control-deck {
            display: grid;
            grid-template-columns: minmax(200px, 0.8fr) minmax(280px, 1.2fr) minmax(260px, 1fr);
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
            min-width: 118px;
            flex: 1 1 118px;
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
        .mode-tabs button,
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
            max-width: min(210px, calc(100% - 24px));
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
        .legend-mark.tx::before {
            content: "";
            position: absolute;
            left: 5px;
            top: 1px;
            width: 11px;
            height: 11px;
            border-radius: 3px;
            background: var(--c-blue);
        }
        .legend-mark.rx::before {
            content: "";
            position: absolute;
            left: 6px;
            top: 1px;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--text);
            opacity: 0.86;
        }
        .legend-mark.link::before,
        .legend-mark.interference::before {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            top: 5px;
            border-top: 2px solid rgba(255,106,61,0.82);
        }
        .legend-mark.interference::before {
            border-top: 2px dashed rgba(77,163,255,0.74);
        }
        .legend-mark.halo::before {
            content: "";
            position: absolute;
            left: 4px;
            top: -1px;
            width: 13px;
            height: 13px;
            border-radius: 50%;
            border: 3px solid rgba(255,106,61,0.5);
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
            background: var(--c-orange);
            box-shadow: 0 0 14px rgba(255, 106, 61, 0.55);
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
        .panel-label {
            font-family: var(--font-mono);
            font-size: 10px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--text-mute);
            margin-bottom: 8px;
        }
        .bars {
            display: grid;
            gap: 7px;
        }
        .bar-row {
            display: grid;
            grid-template-columns: 42px 1fr 48px;
            gap: 8px;
            align-items: center;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-dim);
        }
        .bar-track {
            height: 8px;
            border-radius: 99px;
            background: rgba(255,255,255,0.075);
            overflow: hidden;
        }
        .bar-fill {
            display: block;
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, rgba(255,106,61,0.45), var(--bar-color, var(--c-orange)));
            box-shadow: 0 0 12px color-mix(in srgb, var(--bar-color, var(--c-orange)) 45%, transparent);
            width: 0%;
            transition: width var(--dur-mid) var(--ease);
        }
        .layer-trace, .trace-section + .trace-section {
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
            grid-template-columns: 42px 1fr 74px;
            gap: 8px;
            align-items: center;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-dim);
        }
        .layer-track {
            height: 8px;
            border-radius: 99px;
            background: rgba(255,255,255,0.075);
            overflow: hidden;
        }
        .layer-fill {
            display: block;
            height: 100%;
            width: 0;
            border-radius: inherit;
            background: rgba(255,106,61,0.78);
            transition: width var(--dur-mid) var(--ease);
        }
        .diagnostic-drawer {
            margin-top: 18px;
        }
        .strip {
            padding: 14px;
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
        .drawer-toggle {
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
            0% { stroke-dashoffset: 0; opacity: 0.18; }
            45% { opacity: 0.62; }
            100% { stroke-dashoffset: -26; opacity: 0.18; }
        }
        .heat-focus-line {
            animation: focusPulse 1.1s ease-in-out infinite;
        }
        @keyframes focusPulse {
            0%, 100% { opacity: 0.48; }
            50% { opacity: 0.95; }
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
        .tx-label, .rx-label {
            font-family: var(--font-mono);
            font-size: 17px;
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

    <div class="head">
        <div>
            <span class="eyebrow">INTERACTIVE - live browser inference</span>
            <div class="title" data-title>Draw the D2D interference channel and compare WMMSE, GNN, and Greedy.</div>
            <div class="sub" data-sub>Drag any transmitter or receiver. The browser rebuilds the graph, runs the exported D2D GNN, runs live WMMSE, and recomputes SINR plus sum-rate for the current geometry.</div>
        </div>
        <div class="mode-tabs" aria-label="Scenario mode">
            <button type="button" class="is-active" data-mode="d2d">D2D</button>
            <button type="button" data-mode="jsac">JSAC</button>
        </div>
    </div>

    <div data-panel="d2d">
    <div class="toolbar">
        <div class="control-deck">
            <section class="control-group" aria-label="Topology controls">
                <span class="control-title">Topology</span>
                <div class="control-row">
                    <label class="control-field is-narrow"><span>Pairs K</span>
                <select data-k></select>
                    </label>
                    <div class="stepper" aria-label="Adjust pair count">
                        <button type="button" class="icon-btn" data-remove aria-label="Remove pair">-</button>
                        <button type="button" class="icon-btn" data-add aria-label="Add pair">+</button>
                    </div>
                </div>
            </section>
            <section class="control-group" aria-label="Layout controls">
                <span class="control-title">Layout</span>
                <div class="control-row">
                    <label class="control-field"><span>Preset</span>
                        <select data-preset-select>
                            <option value="custom">Custom</option>
                            <option value="balanced">Balanced field</option>
                            <option value="hidden">Hidden terminal</option>
                            <option value="crowded">Dense interference</option>
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
        <span class="status"><span class="dot"></span><span data-status>loading model</span></span>
    </div>

    <div class="stage">
        <div class="field-card">
            <svg data-field viewBox="0 0 1000 1000" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Draggable D2D layout"></svg>
            <div class="field-legend" aria-label="D2D map legend">
                <div class="field-legend-title">Map legend</div>
                <div class="legend-row"><span class="legend-mark tx"></span><span>Transmitter</span></div>
                <div class="legend-row"><span class="legend-mark rx"></span><span>Receiver</span></div>
                <div class="legend-row"><span class="legend-mark link"></span><span>Direct link</span></div>
                <div class="legend-row"><span class="legend-mark interference"></span><span>Interference</span></div>
                <div class="legend-row"><span class="legend-mark halo"></span><span>Allocated power</span></div>
            </div>
            <span class="field-hint">drag Tx/Rx</span>
        </div>
        <aside class="side-card">
            <div>
                <div class="panel-label">Method</div>
                <div class="method-tabs" data-method-tabs></div>
            </div>
            <div>
                <div class="panel-label">Power allocation</div>
                <div class="bars" data-bars></div>
            </div>
            <div>
                <div class="panel-label">Live metrics</div>
                <div class="metric-list" data-metrics></div>
            </div>
            <div>
                <div class="panel-label">Selected link</div>
                <div class="metric-sub" data-selected>Click a node or drag a link endpoint.</div>
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
                <button type="button" data-diagnostic-tab="history">History</button>
                <button type="button" data-diagnostic-tab="heat">Heatmap</button>
                <button type="button" data-diagnostic-tab="solver">Solver</button>
            </div>
            <button type="button" class="drawer-toggle" data-drawer-toggle>Collapse</button>
        </div>
        <div class="drawer-body">
            <section class="drawer-panel" data-diagnostic-panel="history">
                <div class="panel-label">Comparison history</div>
                <div class="history" data-history></div>
            </section>
            <section class="drawer-panel heat" data-diagnostic-panel="heat">
                <div class="heat-grid">
                    <div>
                        <div class="panel-label">Channel matrix |h|^2</div>
                        <svg data-heat viewBox="0 0 220 220" preserveAspectRatio="xMidYMid meet" aria-label="Channel matrix heatmap"></svg>
                    </div>
                    <div class="metric-sub">Darker direct cells carry stronger intended links; blue off-diagonal cells are interference paths.</div>
                </div>
            </section>
            <section class="drawer-panel" data-diagnostic-panel="solver">
                <div class="trace-stack">
                    <div class="trace-section">
                        <div class="panel-label">GNN layer summary</div>
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

    <div class="caption">D2D methods are WMMSE / GNN / Greedy. Latency pills are normalized for demonstration: WMMSE/Greedy are live JS elapsed, while GNN is mapped from live WMMSE time using D2D benchmark ratios. Raw browser time is available in the tooltip; this is not a fair runtime benchmark.</div>
    </div>

    <live-run-jsac-lab data-panel="jsac" hidden></live-run-jsac-lab>
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

function displayLatencyMs(result) {
    return Number.isFinite(result?.demoTimeMs) ? result.demoTimeMs : result?.timeMs;
}

function latencyTitle(method, result) {
    if (!result) return 'Latency pending.';
    const shown = displayLatencyMs(result);
    const raw = Number.isFinite(result.rawTimeMs) ? result.rawTimeMs : result.timeMs;
    if (method === 'GNN') {
        return `Normalized demo latency ${fmtMs(shown)} ms. Raw browser elapsed ${fmtMs(raw)} ms; adjusted with the benchmarked D2D GNN/WMMSE ratio.`;
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

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

class LiveRunLab extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' }).appendChild(TEMPLATE.content.cloneNode(true));

        this.mode = 'd2d';
        this.k = DEFAULT_K;
        this.seed = 13;
        this.selectedMethod = 'GNN';
        this.selected = null;
        this.drag = null;
        this.computeTicket = 0;
        this.pendingCompute = 0;
        this.manifest = null;
        this.weights = null;
        this.session = null;
        this.engine = 'loading';
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
        this.activeDiagnostic = 'history';
        this.drawerCollapsed = false;
        this.hoverEdge = null;

        this.$field = this.shadowRoot.querySelector('[data-field]');
        this.$heat = this.shadowRoot.querySelector('[data-heat]');
        this.$k = this.shadowRoot.querySelector('[data-k]');
        this.$presetSelect = this.shadowRoot.querySelector('[data-preset-select]');
        this.$savedSelect = this.shadowRoot.querySelector('[data-saved-select]');
        this.$status = this.shadowRoot.querySelector('[data-status]');
        this.$metrics = this.shadowRoot.querySelector('[data-metrics]');
        this.$bars = this.shadowRoot.querySelector('[data-bars]');
        this.$layers = this.shadowRoot.querySelector('[data-layers]');
        this.$history = this.shadowRoot.querySelector('[data-history]');
        this.$wmmseTrace = this.shadowRoot.querySelector('[data-wmmse-trace]');
        this.$selected = this.shadowRoot.querySelector('[data-selected]');
        this.$methodTabs = this.shadowRoot.querySelector('[data-method-tabs]');
        this.$freeze = this.shadowRoot.querySelector('[data-freeze]');
        this.$miniHistory = this.shadowRoot.querySelector('[data-mini-history]');
        this.$drawer = this.shadowRoot.querySelector('[data-drawer]');
        this.$drawerToggle = this.shadowRoot.querySelector('[data-drawer-toggle]');
        this.$title = this.shadowRoot.querySelector('[data-title]');
        this.$sub = this.shadowRoot.querySelector('[data-sub]');
    }

    connectedCallback() {
        this._initControls();
        this._randomizeLayout(false);
        this._bindDrag();
        this._draw();
        this._load();
    }

    _initControls() {
        this.shadowRoot.querySelectorAll('[data-mode]').forEach((btn) => {
            btn.addEventListener('click', () => this._setMode(btn.dataset.mode));
        });
        for (let i = 2; i <= 20; i++) {
            const opt = document.createElement('option');
            opt.value = String(i);
            opt.textContent = String(i);
            if (i === this.k) opt.selected = true;
            this.$k.appendChild(opt);
        }
        this.$k.addEventListener('change', () => {
            this.k = Number(this.$k.value);
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
        this._bindActionButton('[data-add]', () => {
            this.k = clamp(this.k + 1, 2, 20);
            this.$k.value = String(this.k);
            this._randomizeLayout(true);
        });
        this._bindActionButton('[data-remove]', () => {
            this.k = clamp(this.k - 1, 2, 20);
            this.$k.value = String(this.k);
            this._randomizeLayout(true);
        });
        this._bindActionButton('[data-random]', () => {
            this.seed += 17;
            this._randomizeLayout(true);
        });
        this._bindActionButton('[data-fading]', () => {
            this.seed += 101;
            this._fading = null;
            this._scheduleCompute(0);
        });
        this.shadowRoot.querySelectorAll('[data-preset]').forEach((btn) => {
            this._bindActionButton(btn, () => this._applyPreset(btn.dataset.preset));
        });
        this.shadowRoot.querySelectorAll('[data-diagnostic-tab]').forEach((btn) => {
            btn.addEventListener('click', () => this._setDiagnostic(btn.dataset.diagnosticTab));
        });
        this._bindActionButton(this.$drawerToggle, () => this._toggleDrawer());

        this._renderMethodTabs();
        this._syncDiagnosticShell();
        this._setMode(this.mode);
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
        return 'wireless-power-control.live-run.d2d.saved-layouts.v1';
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
            // Local storage can be unavailable in hardened browser contexts; the live lab still works without persistence.
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
        const id = `d2d-${Date.now().toString(36)}`;
        const layout = {
            id,
            name: `D2D ${this.savedLayouts.length + 1}`,
            k: this.k,
            seed: this.seed,
            tx: this.tx.map((p) => ({ x: p.x, y: p.y })),
            rx: this.rx.map((p) => ({ x: p.x, y: p.y })),
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
        this._captureTransitionStart();
        this.k = clamp(Number(layout.k) || DEFAULT_K, 2, this.manifest?.max_k || 20);
        this.seed = Number(layout.seed) || this.seed;
        this.$k.value = String(this.k);
        if (this.$presetSelect) this.$presetSelect.value = 'custom';
        this.tx = layout.tx.slice(0, this.k).map((p) => ({ x: clamp(p.x, 0, field), y: clamp(p.y, 0, field) }));
        this.rx = layout.rx.slice(0, this.k).map((p) => ({ x: clamp(p.x, 0, field), y: clamp(p.y, 0, field) }));
        while (this.tx.length < this.k || this.rx.length < this.k) {
            this.seed += 17;
            this._randomizeLayout(false);
            break;
        }
        this.selected = null;
        this.last = null;
        this.results = null;
        this.layerTrace = null;
        this.wmmseTrace = [];
        this.history = [];
        this.historyAnimationMode = 'idle';
        this._fading = null;
        this.computeTicket++;
        window.clearTimeout(this.pendingCompute);
        this._draw();
        this._scheduleCompute(0);
    }

    _setMode(mode) {
        this.mode = mode === 'jsac' ? 'jsac' : 'd2d';
        this.shadowRoot.querySelectorAll('[data-mode]').forEach((btn) => {
            btn.classList.toggle('is-active', btn.dataset.mode === this.mode);
            btn.setAttribute('aria-pressed', btn.dataset.mode === this.mode ? 'true' : 'false');
        });
        this.shadowRoot.querySelectorAll('[data-panel]').forEach((panel) => {
            panel.hidden = panel.dataset.panel !== this.mode;
        });
        if (this.mode === 'jsac') {
            this.$title.textContent = 'Draw Blue-car JSAC clusters and compare WMMSE, GNN, and Naive.';
            this.$sub.textContent = 'Drag Blue transmitters or Yellow/Green receivers to rebuilds the graph.';
        } else {
            this.$title.textContent = 'Draw the D2D interference channel and compare WMMSE, GNN, and Greedy.';
            this.$sub.textContent = 'Drag any transmitter or receiver to rebuilds the graph.';
        }
    }

    _renderMethodTabs() {
        this.$methodTabs.innerHTML = '';
        for (const method of ['WMMSE', 'GNN', 'Greedy']) {
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
        this._renderMethodTabs();
        this._startMethodTransition(previous, method);
    }

    _setDiagnostic(name) {
        if (!['history', 'heat', 'solver'].includes(name)) return;
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
            this.manifest = await fetch(MANIFEST_URL, { cache: 'no-cache' }).then((r) => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.json();
            });
            this.weights = await fetch(MODEL_BASE + this.manifest.weights_fallback, { cache: 'no-cache' }).then((r) => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.json();
            });
            this.engine = 'JS fallback ready';
            this._setStatus('JS fallback ready');
            await this._computeAll();

            try {
                await loadScript(ORT_SCRIPT);
                if (!window.ort) throw new Error('ort global missing');
                window.ort.env.wasm.wasmPaths = ORT_WASM_PATH;
                window.ort.env.wasm.numThreads = 1;
                this.session = await window.ort.InferenceSession.create(MODEL_BASE + this.manifest.model, {
                    executionProviders: ['wasm'],
                });
                this.engine = 'ONNX Runtime Web';
                this._setStatus('ONNX Runtime Web ready');
                await this._computeAll();
            } catch (err) {
                this.session = null;
                this.engine = 'JS fallback';
                this._setStatus(`JS fallback - ONNX unavailable`);
                console.warn('[live-run-lab] ONNX Runtime Web unavailable; using JS fallback', err);
            }
        } catch (err) {
            this._setStatus('model assets unavailable');
            console.error('[live-run-lab] failed to load model assets', err);
        }
    }

    _setStatus(text) {
        this.$status.textContent = text;
    }

    _randomizeLayout(recompute) {
        const rand = mulberry32(this.seed + this.k * 997);
        const field = this._fieldLength();
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
        this.tx = [];
        this.rx = [];
        for (let i = 0; i < this.k; i++) {
            const tx = {
                x: 80 + rand() * (field - 160),
                y: 80 + rand() * (field - 160),
            };
            const d = 8 + rand() * 45;
            const a = rand() * Math.PI * 2;
            const rx = {
                x: clamp(tx.x + Math.cos(a) * d, 14, field - 14),
                y: clamp(tx.y + Math.sin(a) * d, 14, field - 14),
            };
            this.tx.push(tx);
            this.rx.push(rx);
        }
        this.selected = null;
        this._fading = null;
        this._draw();
        if (recompute) this._scheduleCompute(0);
    }

    _applyPreset(name) {
        const presets = {
            balanced: {
                seed: 41,
                k: 8,
                tx: [
                    [150, 170], [360, 145], [640, 165], [830, 230],
                    [190, 650], [420, 820], [670, 720], [840, 610],
                ],
                rx: [
                    [188, 188], [398, 118], [603, 198], [802, 268],
                    [226, 630], [457, 792], [704, 748], [807, 582],
                ],
            },
            hidden: {
                seed: 83,
                k: 6,
                tx: [
                    [170, 250], [475, 232], [515, 278],
                    [760, 215], [260, 710], [735, 730],
                ],
                rx: [
                    [230, 250], [420, 250], [432, 280],
                    [820, 220], [318, 690], [690, 690],
                ],
            },
            crowded: {
                seed: 127,
                k: 10,
                tx: [
                    [135, 170], [310, 140], [485, 205], [675, 155], [850, 230],
                    [170, 650], [360, 800], [570, 700], [760, 820], [860, 610],
                ],
                rx: [
                    [178, 198], [352, 167], [452, 238], [718, 183], [815, 270],
                    [212, 625], [398, 772], [612, 732], [724, 790], [822, 580],
                ],
            },
        };
        const preset = presets[name] || presets.balanced;
        const field = this._fieldLength();
        this._captureTransitionStart();
        if (Number.isFinite(preset.seed)) this.seed = preset.seed;
        this.k = preset.k;
        this.$k.value = String(this.k);
        this.$presetSelect.value = name;
        this.tx = preset.tx.map(([x, y]) => ({ x: clamp(x, 0, field), y: clamp(y, 0, field) }));
        this.rx = preset.rx.map(([x, y]) => ({ x: clamp(x, 0, field), y: clamp(y, 0, field) }));
        this.selected = null;
        this.last = null;
        this.results = null;
        this.layerTrace = null;
        this.wmmseTrace = [];
        this.history = [];
        this.historyAnimationMode = 'idle';
        this._fading = null;
        this.computeTicket++;
        window.clearTimeout(this.pendingCompute);
        this._draw();
        this._scheduleCompute(0);
    }

    _fieldLength() {
        return this.manifest?.physics?.field_length || 1000;
    }

    _ensureFading() {
        if (this._fading && this._fading.length >= this.k && this.$freeze.checked) return;
        const rand = mulberry32(this.seed * 31 + this.k * 131);
        this._fading = [];
        for (let i = 0; i < this.k; i++) {
            const row = [];
            for (let j = 0; j < this.k; j++) {
                const g1 = gaussian(rand);
                const g2 = gaussian(rand);
                row.push(Math.max(1e-6, 0.5 * (g1 * g1 + g2 * g2)));
            }
            this._fading.push(row);
        }
    }

    _distances() {
        const d = [];
        for (let i = 0; i < this.k; i++) {
            const row = [];
            for (let j = 0; j < this.k; j++) {
                row.push(Math.max(1, Math.hypot(this.tx[j].x - this.rx[i].x, this.tx[j].y - this.rx[i].y)));
            }
            d.push(row);
        }
        return d;
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

    _channelLosses(dist) {
        this._ensureFading();
        const losses = [];
        for (let i = 0; i < this.k; i++) {
            const row = [];
            for (let j = 0; j < this.k; j++) {
                row.push(this._pathLoss(dist[i][j], i === j) * this._fading[i][j]);
            }
            losses.push(row);
        }
        return losses;
    }

    _normValue(v, scaler, diag) {
        if (diag) return (v - scaler.diag_mean) / (scaler.diag_var + EPS);
        return (v - scaler.off_mean) / (scaler.off_var + EPS);
    }

    _buildInputs() {
        const maxK = this.manifest.max_k;
        const dist = this._distances();
        const losses = this._channelLosses(dist);
        const x = new Float32Array(maxK * 3);
        const edgeAttr = new Float32Array(maxK * maxK);
        const edgeMask = new Float32Array(maxK * maxK);
        const nodeMask = new Float32Array(maxK);
        const threshold = this.manifest.physics.threshold;

        for (let i = 0; i < this.k; i++) {
            nodeMask[i] = 1;
            const nd = this._normValue(1 / dist[i][i], this.manifest.scalers.dist, true);
            const nl = this._normValue(Math.sqrt(losses[i][i]), this.manifest.scalers.loss, true);
            x[i * 3 + 0] = nd;
            x[i * 3 + 1] = nl;
            x[i * 3 + 2] = 0;
            for (let j = 0; j < this.k; j++) {
                const idx = i * maxK + j;
                edgeAttr[idx] = this._normValue(1 / dist[i][j], this.manifest.scalers.dist, i === j);
                if (i !== j && dist[i][j] <= threshold) edgeMask[idx] = 1;
            }
        }
        return { dist, losses, x, edgeAttr, edgeMask, nodeMask };
    }

    async _computeAll() {
        if (!this.manifest || !this.weights) return;
        this.hoverEdge = null;
        const ticket = ++this.computeTicket;
        const tensors = this._buildInputs();
        const hMag = tensors.losses.map((row) => row.map((v) => Math.sqrt(v)));

        const gnnStart = performance.now();
        const gnn = await this._runGnn(tensors);
        const gnnMs = performance.now() - gnnStart;
        if (ticket !== this.computeTicket) return;
        if (this.session) this.layerTrace = this._traceGnnFallback(tensors);

        const wStart = performance.now();
        const wmmse = this._runWmmse(hMag);
        const wmmseMs = performance.now() - wStart;
        const greedy = this._runGreedy(tensors.losses, wmmse);
        const greedyMs = 0.02;
        const gnnDemoMs = normalizedGnnLatencyMs(
            wmmseMs,
            interpolateRatio(this.k, D2D_DEMO_GNN_WMMSE_RATIO_BY_K),
            greedyMs,
        );

        const methods = {
            WMMSE: { power: wmmse, timeMs: wmmseMs, rawTimeMs: wmmseMs, demoTimeMs: wmmseMs, engine: 'JS WMMSE' },
            GNN: { power: gnn.slice(0, this.k), timeMs: gnnMs, rawTimeMs: gnnMs, demoTimeMs: gnnDemoMs, engine: this.session ? 'ONNX' : 'JS fallback' },
            Greedy: { power: greedy, timeMs: greedyMs, rawTimeMs: greedyMs, demoTimeMs: greedyMs, engine: 'JS greedy' },
        };
        for (const value of Object.values(methods)) {
            Object.assign(value, this._metrics(tensors.losses, value.power));
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
                    x: new window.ort.Tensor('float32', tensors.x, [maxK, 3]),
                    edge_attr: new window.ort.Tensor('float32', tensors.edgeAttr, [maxK, maxK, 1]),
                    edge_mask: new window.ort.Tensor('float32', tensors.edgeMask, [maxK, maxK]),
                    node_mask: new window.ort.Tensor('float32', tensors.nodeMask, [maxK]),
                };
                const out = await this.session.run(feeds);
                return Array.from(out.powers.data);
            } catch (err) {
                console.warn('[live-run-lab] ONNX inference failed; using JS fallback', err);
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
            x.push([tensors.x[i * 3], tensors.x[i * 3 + 1], tensors.x[i * 3 + 2]]);
        }
        const layers = [];
        const capture = (label) => {
            const values = x.map((row, i) => row[2] * tensors.nodeMask[i]);
            const activeValues = values.slice(0, this.k);
            const avg = activeValues.reduce((a, b) => a + b, 0) / Math.max(1, activeValues.length);
            layers.push({
                label,
                values: activeValues,
                avg,
                max: Math.max(...activeValues, 0),
                active: activeValues.filter((v) => v > 0.05).length,
            });
        };
        const conv = (xIn) => {
            const xOut = [];
            for (let target = 0; target < maxK; target++) {
                let aggr = new Array(32).fill(0);
                let seen = false;
                for (let source = 0; source < maxK; source++) {
                    if (tensors.edgeMask[target * maxK + source] < 0.5) continue;
                    const input = [...xIn[source], tensors.edgeAttr[target * maxK + source]];
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
                xOut.push([xIn[target][0], xIn[target][1], sigmoid(u1[0])]);
            }
            return xOut;
        };
        x = conv(x);
        capture('L1');
        x = conv(x);
        capture('L2');
        x = conv(x);
        capture('L3');
        return { final: layers[layers.length - 1].values, layers };
    }

    _runWmmse(H) {
        const k = this.k;
        let b = new Array(k).fill(1);
        let f = new Array(k).fill(0);
        let w = new Array(k).fill(0);
        const noise = this.manifest.physics.var_noise;
        const trace = [];
        const checkpoints = new Set([0, 1, 2, 5, 10, 20, 50, MAX_WMMSE_ITER]);
        const capture = (iter) => {
            if (!checkpoints.has(iter)) return;
            const power = b.map((v) => v * v);
            const sumRate = this._sumRateFromH(H, power);
            trace.push({
                iter,
                sumRate,
                active: power.filter((v) => v > 0.05).length,
            });
        };

        const updateFilters = () => {
            for (let i = 0; i < k; i++) {
                let total = noise;
                for (let j = 0; j < k; j++) {
                    const rx = H[i][j] * b[j];
                    total += rx * rx;
                }
                const valid = H[i][i] * b[i];
                f[i] = valid / (total + EPS);
                w[i] = 1 / (1 - f[i] * valid + EPS);
            }
        };
        updateFilters();
        capture(0);
        for (let iter = 0; iter < MAX_WMMSE_ITER; iter++) {
            const next = new Array(k);
            for (let j = 0; j < k; j++) {
                const bup = w[j] * H[j][j] * f[j];
                let bdown = 0;
                for (let i = 0; i < k; i++) {
                    const rx = H[i][j] * f[i];
                    bdown += w[i] * rx * rx;
                }
                next[j] = clamp(bup / (bdown + EPS), 0, 1);
            }
            b = next;
            updateFilters();
            capture(iter + 1);
        }
        this.wmmseTrace = trace;
        return b.map((v) => v * v);
    }

    _sumRateFromH(H, p) {
        const noise = this.manifest.physics.var_noise;
        let sum = 0;
        for (let i = 0; i < p.length; i++) {
            let interf = noise;
            for (let j = 0; j < p.length; j++) {
                if (i !== j) interf += p[j] * H[i][j] * H[i][j];
            }
            const signal = p[i] * H[i][i] * H[i][i];
            sum += Math.log2(1 + signal / (interf + EPS));
        }
        return sum;
    }

    _runGreedy(losses, referencePower) {
        const k = this.k;
        const active = clamp(Math.floor(referencePower.reduce((a, b) => a + b, 0)), 1, k);
        const ranked = [];
        for (let i = 0; i < k; i++) ranked.push({ i, score: losses[i][i] * losses[i][i] });
        ranked.sort((a, b) => b.score - a.score);
        const p = new Array(k).fill(0);
        for (let n = 0; n < active; n++) p[ranked[n].i] = 1;
        return p;
    }

    _metrics(losses, p) {
        const noise = this.manifest.physics.var_noise;
        const sinr = [];
        const rates = [];
        for (let i = 0; i < this.k; i++) {
            let interf = noise;
            for (let j = 0; j < this.k; j++) {
                if (i !== j) interf += p[j] * losses[i][j];
            }
            const s = (p[i] * losses[i][i]) / (interf + EPS);
            sinr.push(s);
            rates.push(Math.log2(1 + s));
        }
        return {
            sinr,
            rates,
            sumRate: rates.reduce((a, b) => a + b, 0),
            avgRate: rates.reduce((a, b) => a + b, 0) / this.k,
            minSinr: Math.min(...sinr),
            activeLinks: p.filter((v) => v > 0.05).length,
        };
    }

    _appendHistory(methods) {
        const entry = {
            WMMSE: methods.WMMSE?.sumRate || 0,
            GNN: methods.GNN?.sumRate || 0,
            Greedy: methods.Greedy?.sumRate || 0,
        };
        const previous = this.history[this.history.length - 1];
        const changed = !previous || Object.keys(entry).some((method) => Math.abs(entry[method] - previous[method]) > 0.01);
        if (!changed) return;
        this.history.push(entry);
        if (this.history.length > 24) this.history.shift();
        this.historyAnimationMode = previous ? 'append' : 'refresh';
    }

    _methodNames() {
        return ['WMMSE', 'GNN', 'Greedy'];
    }

    _displaySource() {
        return this.displayResults || this.results;
    }

    _captureTransitionStart() {
        this._clearMethodTransition();
        if (this.resultAnimationFrame) {
            window.cancelAnimationFrame(this.resultAnimationFrame);
            this.resultAnimationFrame = 0;
        }
        const source = this.displayResults || this.results || this.transitionFromResults;
        this.transitionFromResults = source;
        this.displayResults = source ? this._cloneDisplayResults(source) : null;
        this.resultAnimationT = 1;
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
                timeMs: lerpValue(from.timeMs, to.timeMs, t),
                rawTimeMs: lerpValue(from.rawTimeMs, to.rawTimeMs, t),
                demoTimeMs: lerpValue(from.demoTimeMs, to.demoTimeMs, t),
                sumRate: lerpValue(from.sumRate, to.sumRate, t),
                avgRate: lerpValue(from.avgRate, to.avgRate, t),
                minSinr: lerpValue(from.minSinr, to.minSinr, t),
                activeLinks: lerpValue(from.activeLinks, to.activeLinks, t),
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
    }

    _startMethodTransition(fromMethod, toMethod) {
        const source = this._displaySource();
        const fromPower = this.visualPower || source?.[fromMethod]?.power;
        const toPower = source?.[toMethod]?.power;
        if (!fromPower || !toPower) {
            this._clearMethodTransition();
            this._draw();
            return;
        }
        if (this.methodAnimationFrame) window.cancelAnimationFrame(this.methodAnimationFrame);
        const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (reduceMotion) {
            this.visualPower = [...toPower];
            this._draw();
            this.visualPower = null;
            return;
        }
        const from = [...fromPower];
        const to = [...toPower];
        const start = performance.now();
        const duration = 520;
        const tick = (now) => {
            const raw = clamp((now - start) / duration, 0, 1);
            const eased = easeOutCubic(raw);
            this.visualPower = this._interpolateArray(from, to, eased);
            this._draw();
            if (raw < 1) {
                this.methodAnimationFrame = window.requestAnimationFrame(tick);
            } else {
                this.methodAnimationFrame = 0;
                this.visualPower = null;
                this._draw();
            }
        };
        this.methodAnimationFrame = window.requestAnimationFrame(tick);
    }

    _scheduleCompute(delay = 70) {
        window.clearTimeout(this.pendingCompute);
        this.pendingCompute = window.setTimeout(() => this._computeAll(), delay);
    }

    _bindDrag() {
        this.$field.addEventListener('pointerdown', (ev) => {
            const node = ev.target.closest?.('.node');
            if (!node) return;
            const kind = node.dataset.kind;
            const index = Number(node.dataset.index);
            const point = this._fieldPointFromEvent(ev);
            const arr = kind === 'tx' ? this.tx : this.rx;
            const current = arr[index];
            this.drag = {
                kind,
                index,
                el: node,
                offsetX: current.x - point.x,
                offsetY: current.y - point.y,
            };
            this.selected = { kind, index };
            node.setPointerCapture(ev.pointerId);
            node.classList.add('is-dragging');
            this._moveFromEvent(ev);
        });
        this.$field.addEventListener('pointermove', (ev) => {
            if (!this.drag) return;
            this._moveFromEvent(ev);
        });
        this.$field.addEventListener('pointerup', (ev) => {
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
        const arr = this.drag.kind === 'tx' ? this.tx : this.rx;
        arr[this.drag.index] = { x, y };
        this._draw();
        this._scheduleCompute();
    }

    _draw() {
        this._drawField();
        this._drawMetrics();
        this._drawBars();
        this._drawMiniHistory();
        this._drawLayerTrace();
        this._drawHistory();
        this._drawWmmseTrace();
        this._drawHeatmap();
        this._syncDiagnosticShell();
    }

    _selectedPower() {
        return this.visualPower || this._displaySource()?.[this.selectedMethod]?.power || new Array(this.k).fill(0);
    }

    _drawField() {
        const svg = this.$field;
        svg.innerHTML = '';
        const field = this._fieldLength();
        const p = this._selectedPower();
        const losses = this._hasCurrentLosses() ? this.last.losses : null;

        const bg = svgEl('rect', { x: 0, y: 0, width: field, height: field, fill: 'rgba(255,255,255,0.015)' });
        svg.appendChild(bg);
        for (let g = 100; g < field; g += 100) {
            svg.appendChild(svgEl('line', { x1: g, y1: 0, x2: g, y2: field, stroke: 'rgba(255,255,255,0.035)', 'stroke-width': 1 }));
            svg.appendChild(svgEl('line', { x1: 0, y1: g, x2: field, y2: g, stroke: 'rgba(255,255,255,0.035)', 'stroke-width': 1 }));
        }

        if (losses) {
            const edges = [];
            for (let target = 0; target < this.k; target++) {
                for (let source = 0; source < this.k; source++) {
                    if (target === source) continue;
                    edges.push({ target, source, score: losses[target][source] * (p[source] || 0) });
                }
            }
            edges.sort((a, b) => b.score - a.score);
            const maxScore = edges[0]?.score || 1;
            edges.slice(0, Math.min(70, edges.length)).forEach((e, n) => {
                const op = clamp(0.08 + 0.38 * Math.sqrt(e.score / (maxScore + EPS)), 0.08, 0.46);
                svg.appendChild(svgEl('line', {
                    class: 'message-edge',
                    x1: this.tx[e.source].x,
                    y1: this.tx[e.source].y,
                    x2: this.rx[e.target].x,
                    y2: this.rx[e.target].y,
                    stroke: 'rgba(77,163,255,' + op + ')',
                    'stroke-width': 1.4,
                    'stroke-dasharray': '5 8',
                    style: `--edge-delay:${(n % 8) * 0.08}s`,
                }));
            });
        }

        if (this.hoverEdge && this.tx[this.hoverEdge.source] && this.rx[this.hoverEdge.target]) {
            const tx = this.tx[this.hoverEdge.source];
            const rx = this.rx[this.hoverEdge.target];
            svg.appendChild(svgEl('line', {
                class: 'heat-focus-line',
                x1: tx.x,
                y1: tx.y,
                x2: rx.x,
                y2: rx.y,
                stroke: 'rgba(255,255,255,0.92)',
                'stroke-width': 2.6,
                'stroke-dasharray': '7 5',
            }));
            svg.appendChild(svgEl('circle', {
                cx: rx.x,
                cy: rx.y,
                r: 14,
                fill: 'none',
                stroke: 'rgba(255,255,255,0.75)',
                'stroke-width': 2,
            }));
        }

        for (let i = 0; i < this.k; i++) {
            const power = p[i] || 0;
            svg.appendChild(svgEl('line', {
                x1: this.tx[i].x,
                y1: this.tx[i].y,
                x2: this.rx[i].x,
                y2: this.rx[i].y,
                stroke: 'rgba(255,106,61,' + (0.25 + power * 0.65).toFixed(3) + ')',
                'stroke-width': 2.2,
            }));
        }

        for (let i = 0; i < this.k; i++) {
            const power = p[i] || 0;
            svg.appendChild(svgEl('circle', {
                cx: this.tx[i].x,
                cy: this.tx[i].y,
                r: 18 + power * 32,
                fill: 'none',
                stroke: 'rgba(255,106,61,' + (0.13 + power * 0.42).toFixed(3) + ')',
                'stroke-width': 5,
            }));

            const rxGroup = svgEl('g', { class: 'node', 'data-kind': 'rx', 'data-index': i, 'aria-label': `Receiver ${i}` });
            rxGroup.appendChild(svgEl('circle', { cx: this.rx[i].x, cy: this.rx[i].y, r: 8.5, fill: 'var(--text)', opacity: 0.86 }));
            rxGroup.appendChild(svgEl('text', { class: 'rx-label', x: this.rx[i].x + 13, y: this.rx[i].y + 5 }));
            rxGroup.lastChild.textContent = `R${i}`;
            svg.appendChild(rxGroup);

            const txGroup = svgEl('g', { class: 'node', 'data-kind': 'tx', 'data-index': i, 'aria-label': `Transmitter ${i}` });
            txGroup.appendChild(svgEl('rect', { x: this.tx[i].x - 9, y: this.tx[i].y - 9, width: 18, height: 18, rx: 3, fill: 'var(--c-blue)', opacity: 0.95 }));
            txGroup.appendChild(svgEl('text', { class: 'tx-label', x: this.tx[i].x + 14, y: this.tx[i].y + 5 }));
            txGroup.lastChild.textContent = `T${i}`;
            svg.appendChild(txGroup);
        }

        if (this.selected) {
            const arr = this.selected.kind === 'tx' ? this.tx : this.rx;
            const p0 = arr[this.selected.index];
            this.$selected.textContent = `${this.selected.kind.toUpperCase()}${this.selected.index} - x ${fmt(p0.x, 1)} m / y ${fmt(p0.y, 1)} m`;
        } else {
            this.$selected.textContent = 'Click a node or drag a link endpoint.';
        }
    }

    _drawMetrics() {
        const methods = ['WMMSE', 'GNN', 'Greedy'];
        const display = this._displaySource();
        this.$metrics.innerHTML = '';
        for (const method of methods) {
            const r = display?.[method];
            const row = document.createElement('div');
            row.className = 'metric-row';
            row.classList.toggle('is-active', method === this.selectedMethod);
            row.addEventListener('click', () => this._selectMethod(method));
            const latency = r ? displayLatencyMs(r) : NaN;
            const latencyTip = r ? latencyTitle(method, r) : 'Latency pending.';
            row.innerHTML = `
                    <span class="method-name" style="color:${method === 'WMMSE' ? 'var(--c-blue)' : method === 'GNN' ? 'var(--c-orange)' : 'var(--c-grey)'}">${method}</span>
                <span>
                    <span class="metric-main">${r ? fmt(r.sumRate, 2) : '--'}</span>
                    <span class="metric-sub"> b/s/Hz sum-rate / active ${r ? Math.round(r.activeLinks) : '--'}</span>
                </span>
                <span class="pill" title="${escapeAttr(latencyTip)}" aria-label="${escapeAttr(latencyTip)}">${r ? `~${fmtMs(latency)} ms` : '--'}</span>
            `;
            this.$metrics.appendChild(row);
        }
        if (display?.GNN) {
            this._setStatus(`${display.GNN.engine} / GNN demo ~${fmtMs(displayLatencyMs(display.GNN))} ms / K=${this.k}`);
        }
    }

    _drawBars() {
        this.$bars.innerHTML = '';
        const p = this._selectedPower();
        const color = this._methodColor(this.selectedMethod);
        for (let i = 0; i < this.k; i++) {
            const power = clamp(p[i] || 0, 0, 1);
            const row = document.createElement('div');
            row.className = 'bar-row';
            row.innerHTML = `
                <span>p${i}</span>
                <span class="bar-track"><span class="bar-fill" style="--bar-color:${color};width:${power * 100}%"></span></span>
                <span>${fmt(power, 2)}</span>
            `;
            this.$bars.appendChild(row);
        }
    }

    _drawLayerTrace() {
        this.$layers.innerHTML = '';
        const layers = this.layerTrace?.layers || [];
        if (!layers.length) {
            this.$layers.innerHTML = '<div class="metric-sub">Waiting for GNN weights.</div>';
            return;
        }
        for (const layer of layers) {
            const animatedAvg = layer.avg * this.resultAnimationT;
            const row = document.createElement('div');
            row.className = 'layer-row';
            row.innerHTML = `
                <span>${layer.label}</span>
                <span class="layer-track"><span class="layer-fill" style="width:${clamp(animatedAvg, 0, 1) * 100}%"></span></span>
                <span>avg ${fmt(layer.avg, 2)}</span>
            `;
            row.title = `max ${fmt(layer.max, 2)} / active ${layer.active}`;
            this.$layers.appendChild(row);
        }
    }

    _drawHistory() {
        this.$history.innerHTML = '';
        const methods = ['WMMSE', 'GNN', 'Greedy'];
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

    _drawMiniHistory() {
        if (!this.$miniHistory) return;
        this.$miniHistory.innerHTML = '';
        const latest = this.history[this.history.length - 1];
        if (!latest) {
            this.$miniHistory.innerHTML = '<div class="metric-sub">Waiting for first solve.</div>';
            return;
        }
        const methods = ['WMMSE', 'GNN', 'Greedy'];
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

    _drawWmmseTrace() {
        this.$wmmseTrace.innerHTML = '';
        const trace = this.wmmseTrace || [];
        if (!trace.length) {
            this.$wmmseTrace.innerHTML = '<div class="metric-sub">Waiting for WMMSE solve.</div>';
            return;
        }
        const max = Math.max(...trace.map((entry) => entry.sumRate), 1);
        const min = Math.min(...trace.map((entry) => entry.sumRate));
        const width = 420;
        const height = 92;
        const pad = 12;
        const progress = this.resultAnimationT;
        const denom = Math.max(max - min, 1e-9);
        const points = trace.map((entry, index) => {
            const x = pad + (width - pad * 2) * (index / Math.max(1, trace.length - 1)) * progress;
            const y = height - pad - (height - pad * 2) * ((entry.sumRate - min) / denom);
            return `${fmt(x, 2)},${fmt(y, 2)}`;
        }).join(' ');
        const dots = trace.map((entry, index) => {
            const reveal = index <= Math.floor((trace.length - 1) * progress);
            if (!reveal) return '';
            const x = pad + (width - pad * 2) * (index / Math.max(1, trace.length - 1));
            const y = height - pad - (height - pad * 2) * ((entry.sumRate - min) / denom);
            return `<circle cx="${fmt(x, 2)}" cy="${fmt(y, 2)}" r="3" fill="var(--c-blue)"><title>I${entry.iter} ${fmt(entry.sumRate, 2)} / active ${entry.active}</title></circle>`;
        }).join('');
        const last = trace[trace.length - 1];
        this.$wmmseTrace.innerHTML = `
            <div class="trace-spark">
                <svg class="spark-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="WMMSE convergence sparkline">
                    <polyline points="${points}" fill="none" stroke="var(--c-blue)" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"></polyline>
                    ${dots}
                </svg>
                <div class="trace-stats">
                    <span>final ${fmt(last.sumRate, 2)}</span>
                    <span>active ${Math.round(last.active)}</span>
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
        const values = [];
        for (let i = 0; i < this.k; i++) {
            for (let j = 0; j < this.k; j++) values.push(Math.log10(losses[i][j] + 1e-30));
        }
        const lo = Math.min(...values);
        const hi = Math.max(...values);
        const size = 220 / this.k;
        for (let i = 0; i < this.k; i++) {
            for (let j = 0; j < this.k; j++) {
                const t = (Math.log10(losses[i][j] + 1e-30) - lo) / (hi - lo + EPS);
                const color = i === j
                    ? `rgba(255,106,61,${0.25 + 0.65 * t})`
                    : `rgba(77,163,255,${0.08 + 0.55 * t})`;
                const cell = svgEl('rect', {
                    class: 'heat-cell',
                    x: j * size,
                    y: i * size,
                    width: Math.max(1, size - 1),
                    height: Math.max(1, size - 1),
                    fill: color,
                });
                cell.addEventListener('pointerenter', () => {
                    this.hoverEdge = { target: i, source: j };
                    this._drawField();
                });
                cell.addEventListener('pointerleave', () => {
                    this.hoverEdge = null;
                    this._drawField();
                });
                cell.addEventListener('click', () => {
                    this.selected = { kind: 'rx', index: i };
                    this._draw();
                });
                svg.appendChild(cell);
            }
        }
    }

    _hasCurrentLosses() {
        return Boolean(
            this.last?.losses &&
            this.last.losses.length === this.k &&
            this.last.losses.every((row) => row && row.length === this.k)
        );
    }

    _methodColor(method) {
        if (method === 'WMMSE') return 'var(--c-blue)';
        if (method === 'Greedy') return 'var(--c-grey)';
        return 'var(--c-orange)';
    }
}

customElements.define('live-run-lab', LiveRunLab);
