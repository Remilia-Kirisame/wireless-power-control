import './live-run-jsac-lab.js?v=1.2.4';

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
        .mode-tabs, .method-tabs, .actions {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        button, select, label.toggle {
            border: 1px solid var(--rule);
            background: rgba(255,255,255,0.03);
            color: var(--text-dim);
            border-radius: 6px;
            font-family: var(--font-mono);
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 7px 10px;
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
        }
        label.toggle {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            cursor: pointer;
        }
        label.toggle input {
            accent-color: var(--c-orange);
        }
        .toolbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            flex-wrap: wrap;
            border-top: 1px solid var(--rule-soft);
            border-bottom: 1px solid var(--rule-soft);
            padding: 12px 0;
            margin-bottom: 18px;
        }
        .stage {
            display: grid;
            grid-template-columns: minmax(0, 1.3fr) minmax(320px, 0.7fr);
            gap: 18px;
            align-items: stretch;
        }
        @media (max-width: 980px) {
            .stage { grid-template-columns: 1fr; }
        }
        .field-card, .side-card, .strip {
            border: 1px solid var(--rule);
            border-radius: 8px;
            background: rgba(0,0,0,0.18);
            overflow: hidden;
        }
        .field-card {
            position: relative;
            min-height: 520px;
        }
        svg[data-field] {
            width: 100%;
            height: 100%;
            min-height: 520px;
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
        .status {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-dim);
            letter-spacing: 0.06em;
            display: inline-flex;
            align-items: center;
            gap: 8px;
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
        .diagnostics {
            display: grid;
            grid-template-columns: 1fr 220px;
            gap: 18px;
            margin-top: 18px;
        }
        @media (max-width: 760px) {
            .diagnostics { grid-template-columns: 1fr; }
        }
        .strip {
            padding: 14px;
        }
        .history-strip {
            grid-column: 1 / -1;
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
        .message-edge {
            animation: messagePulse 1.6s linear infinite;
            animation-delay: var(--edge-delay, 0s);
        }
        @keyframes messagePulse {
            0% { stroke-dashoffset: 0; opacity: 0.18; }
            45% { opacity: 0.62; }
            100% { stroke-dashoffset: -26; opacity: 0.18; }
        }
        .heat svg {
            width: 100%;
            aspect-ratio: 1;
            border: 1px solid var(--rule-soft);
            border-radius: 6px;
            background: rgba(0,0,0,0.22);
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
        <div class="actions">
            <label class="toggle">K
                <select data-k></select>
            </label>
            <label class="toggle">Preset
                <select data-preset-select>
                    <option value="custom">Custom</option>
                    <option value="balanced">Balanced</option>
                    <option value="hidden">Hidden terminal</option>
                    <option value="crowded">Crowded corner</option>
                </select>
            </label>
            <label class="toggle">Saved
                <select data-saved-select>
                    <option value="">Saved layouts</option>
                </select>
            </label>
            <button type="button" data-save-layout>Save layout</button>
            <button type="button" data-add>Add pair</button>
            <button type="button" data-remove>Remove pair</button>
            <button type="button" data-random>Randomize</button>
            <button type="button" data-fading>Shuffle fading</button>
            <button type="button" data-preset="balanced">Balanced</button>
            <button type="button" data-preset="hidden">Hidden terminal</button>
            <button type="button" data-preset="crowded">Crowded corner</button>
            <label class="toggle"><input type="checkbox" data-freeze checked />Freeze fading</label>
        </div>
        <span class="status"><span class="dot"></span><span data-status>loading model</span></span>
    </div>

    <div class="stage">
        <div class="field-card">
            <svg data-field viewBox="0 0 1000 1000" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Draggable D2D layout"></svg>
            <span class="field-hint">drag Tx/Rx</span>
        </div>
        <aside class="side-card">
            <div>
                <div class="panel-label">Method</div>
                <div class="method-tabs" data-method-tabs></div>
            </div>
            <div>
                <div class="panel-label">Live metrics</div>
                <div class="metric-list" data-metrics></div>
            </div>
            <div>
                <div class="panel-label">Selected link</div>
                <div class="metric-sub" data-selected>Click a node or drag a link endpoint.</div>
            </div>
        </aside>
    </div>

    <div class="diagnostics">
        <div class="strip">
            <div class="panel-label">Power allocation - selected method</div>
            <div class="bars" data-bars></div>
            <div class="layer-trace">
                <div class="panel-label">Comparison history</div>
                <div class="history" data-history></div>
            </div>
        </div>
        <div class="strip heat">
            <div class="panel-label">Channel matrix |h|^2</div>
            <svg data-heat viewBox="0 0 220 220" preserveAspectRatio="xMidYMid meet" aria-label="Channel matrix heatmap"></svg>
        </div>
        <div class="strip history-strip">
            <div class="panel-label">Solver traces</div>
            <div class="trace-stack">
                <div class="trace-section">
                    <div class="panel-label">GNN layer trace</div>
                    <div data-layers></div>
                </div>
                <div class="trace-section">
                    <div class="panel-label">WMMSE iteration overlay</div>
                    <div data-wmmse-trace></div>
                </div>
            </div>
        </div>
    </div>

    <div class="caption">D2D methods are WMMSE / GNN / Greedy. Equal power is intentionally not used here; JSAC uses the Naive equal-power baseline instead.</div>
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

        this._renderMethodTabs();
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
            this.$sub.textContent = 'Drag Blue transmitters or Yellow/Green receivers. The browser rebuilds same-channel interference, runs the exported JSAC GNN, applies per-Blue softmax, and tracks Green rate plus Yellow SINR constraints.';
        } else {
            this.$title.textContent = 'Draw the D2D interference channel and compare WMMSE, GNN, and Greedy.';
            this.$sub.textContent = 'Drag any transmitter or receiver. The browser rebuilds the graph, runs the exported D2D GNN, runs live WMMSE, and recomputes SINR plus sum-rate for the current geometry.';
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
                k: 10,
                tx: [
                    [130, 160], [205, 135], [280, 190], [190, 250], [335, 270],
                    [710, 170], [820, 230], [745, 330], [520, 760], [640, 835],
                ],
                rx: [
                    [175, 190], [242, 165], [235, 220], [236, 290], [296, 310],
                    [758, 198], [780, 270], [790, 366], [566, 728], [596, 804],
                ],
            },
        };
        const preset = presets[name] || presets.balanced;
        const field = this._fieldLength();
        this._captureTransitionStart();
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

        const methods = {
            WMMSE: { power: wmmse, timeMs: wmmseMs, engine: 'JS WMMSE' },
            GNN: { power: gnn.slice(0, this.k), timeMs: gnnMs, engine: this.session ? 'ONNX' : 'JS fallback' },
            Greedy: { power: greedy, timeMs: 0.02, engine: 'JS greedy' },
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
            this.drag = { kind, index };
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
            const node = ev.target.closest?.('.node');
            node?.classList.remove('is-dragging');
            this.drag = null;
            this._scheduleCompute(0);
        });
    }

    _moveFromEvent(ev) {
        const rect = this.$field.getBoundingClientRect();
        const field = this._fieldLength();
        const x = clamp((ev.clientX - rect.left) / rect.width * field, 0, field);
        const y = clamp((ev.clientY - rect.top) / rect.height * field, 0, field);
        const arr = this.drag.kind === 'tx' ? this.tx : this.rx;
        arr[this.drag.index] = { x, y };
        this._draw();
        this._scheduleCompute();
    }

    _draw() {
        this._drawField();
        this._drawMetrics();
        this._drawBars();
        this._drawLayerTrace();
        this._drawHistory();
        this._drawWmmseTrace();
        this._drawHeatmap();
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
            row.innerHTML = `
                    <span class="method-name" style="color:${method === 'WMMSE' ? 'var(--c-blue)' : method === 'GNN' ? 'var(--c-orange)' : 'var(--c-grey)'}">${method}</span>
                <span>
                    <span class="metric-main">${r ? fmt(r.sumRate, 2) : '--'}</span>
                    <span class="metric-sub"> b/s/Hz sum-rate / active ${r ? Math.round(r.activeLinks) : '--'}</span>
                </span>
                <span class="pill">${r ? fmtMs(r.timeMs) : '--'} ms</span>
            `;
            this.$metrics.appendChild(row);
        }
        if (display?.GNN) {
            this._setStatus(`${display.GNN.engine} / GNN ${fmtMs(display.GNN.timeMs)} ms / K=${this.k}`);
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

    _drawWmmseTrace() {
        this.$wmmseTrace.innerHTML = '';
        const trace = this.wmmseTrace || [];
        if (!trace.length) {
            this.$wmmseTrace.innerHTML = '<div class="metric-sub">Waiting for WMMSE solve.</div>';
            return;
        }
        const max = Math.max(...trace.map((entry) => entry.sumRate), 1);
        for (const entry of trace) {
            const width = clamp(entry.sumRate / max, 0, 1) * this.resultAnimationT * 100;
            const row = document.createElement('div');
            row.className = 'layer-row';
            row.innerHTML = `
                <span>I${entry.iter}</span>
                <span class="layer-track"><span class="layer-fill" style="background:var(--c-blue);width:${width}%"></span></span>
                <span>${fmt(entry.sumRate, 1)}</span>
            `;
            row.title = `active ${entry.active}`;
            this.$wmmseTrace.appendChild(row);
        }
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
                svg.appendChild(svgEl('rect', {
                    x: j * size,
                    y: i * size,
                    width: Math.max(1, size - 1),
                    height: Math.max(1, size - 1),
                    fill: color,
                }));
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
