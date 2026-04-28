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
        .actions, .method-tabs {
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
            background: var(--c-yellow);
            box-shadow: 0 0 14px rgba(246, 196, 69, 0.5);
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
        .side-card {
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 14px;
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
            outline: none;
        }
        .node.is-dragging {
            cursor: grabbing;
        }
        .node:focus-visible .focus-ring {
            opacity: 1;
        }
        .node-label {
            font-family: var(--font-mono);
            font-size: 4px;
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
        <div class="actions">
            <label class="toggle">B
                <select data-b></select>
            </label>
            <label class="toggle">Y
                <select data-my></select>
            </label>
            <label class="toggle">G
                <select data-mg></select>
            </label>
            <button type="button" data-add-blue>Add Blue</button>
            <button type="button" data-remove-blue>Remove Blue</button>
            <button type="button" data-random>Randomize</button>
            <button type="button" data-fading>Shuffle fading</button>
            <label class="toggle"><input type="checkbox" data-freeze checked />Freeze fading</label>
        </div>
        <span class="status"><span class="dot"></span><span data-status>loading JSAC model</span></span>
    </div>

    <div class="stage">
        <div class="field-card">
            <svg data-field viewBox="0 0 225 225" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Draggable JSAC layout"></svg>
            <span class="field-hint">drag Blue/Rx - arrow keys nudge - shift = 10m</span>
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
                <div class="panel-label">Selected node</div>
                <div class="metric-sub" data-selected>Click a Blue car or receiver.</div>
            </div>
        </aside>
    </div>

    <div class="diagnostics">
        <div class="strip">
            <div class="panel-label">Per-Blue budget - selected method</div>
            <div class="budget-rows" data-budgets></div>
        </div>
        <div class="strip heat">
            <div class="panel-label">Channel matrix |h|^2</div>
            <svg data-heat viewBox="0 0 220 220" preserveAspectRatio="xMidYMid meet" aria-label="JSAC channel matrix heatmap"></svg>
        </div>
    </div>

    <div class="caption">JSAC Stage 2 methods are WMMSE / GNN / Naive. Naive splits each Blue car's budget equally across its Yellow and Green receivers.</div>
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

        this.$field = this.shadowRoot.querySelector('[data-field]');
        this.$heat = this.shadowRoot.querySelector('[data-heat]');
        this.$b = this.shadowRoot.querySelector('[data-b]');
        this.$my = this.shadowRoot.querySelector('[data-my]');
        this.$mg = this.shadowRoot.querySelector('[data-mg]');
        this.$status = this.shadowRoot.querySelector('[data-status]');
        this.$metrics = this.shadowRoot.querySelector('[data-metrics]');
        this.$budgets = this.shadowRoot.querySelector('[data-budgets]');
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
        this.shadowRoot.querySelector('[data-add-blue]').addEventListener('click', () => {
            this.B = clamp(this.B + 1, 2, 6);
            this.$b.value = String(this.B);
            this._randomizeLayout(true);
        });
        this.shadowRoot.querySelector('[data-remove-blue]').addEventListener('click', () => {
            this.B = clamp(this.B - 1, 2, 6);
            this.$b.value = String(this.B);
            this._randomizeLayout(true);
        });
        this.shadowRoot.querySelector('[data-random]').addEventListener('click', () => {
            this.seed += 19;
            this._randomizeLayout(true);
        });
        this.shadowRoot.querySelector('[data-fading]').addEventListener('click', () => {
            this.seed += 103;
            this._channelRandoms = null;
            this._scheduleCompute(0);
        });

        this._renderMethodTabs();
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
            btn.addEventListener('click', () => {
                this.selectedMethod = method;
                this._renderMethodTabs();
                this._draw();
            });
            this.$methodTabs.appendChild(btn);
        }
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

        this.last = null;
        this.results = null;
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
        this._channelRandoms = null;
        this._draw();
        if (recompute) this._scheduleCompute(0);
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

        const H = tensors.losses.map((row) => row.map((v) => Math.sqrt(Math.max(v, 0))));
        const wStart = performance.now();
        const wmmse = this._runWmmse(H, tensors.meta);
        const wmmseMs = performance.now() - wStart;
        const nStart = performance.now();
        const naive = this._runNaive(tensors.meta);
        const naiveMs = performance.now() - nStart;

        const methods = {
            WMMSE: { power: wmmse, timeMs: wmmseMs, engine: 'JS WMMSE' },
            GNN: { power: gnnPower, timeMs: gnnMs, engine: this.session ? 'ONNX' : 'JS fallback' },
            Naive: { power: naive, timeMs: naiveMs, engine: 'JS naive' },
        };
        for (const value of Object.values(methods)) {
            Object.assign(value, this._metrics(tensors.losses, value.power, tensors.meta));
        }

        this.last = tensors;
        this.results = methods;
        this._draw();
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
        x = conv(x);
        x = conv(x);
        x = conv(x);
        return x.map((row, i) => row[3] * tensors.nodeMask[i]);
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

            const sinr = this._sinrsFromLosses(H.map((row) => row.map((x) => x * x)), p);
            for (let i = 0; i < k; i++) {
                if (meta.yellowMask[i]) mu[i] = Math.max(0, mu[i] + lrMu * (sinrMin - sinr[i]));
            }
        }
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

    _scheduleCompute(delay = 90) {
        window.clearTimeout(this.pendingCompute);
        this.pendingCompute = window.setTimeout(() => this._computeAll(), delay);
    }

    _bindDrag() {
        this.$field.addEventListener('pointerdown', (ev) => {
            const node = ev.target.closest?.('.node');
            if (!node) return;
            const kind = node.dataset.kind;
            const index = Number(node.dataset.index);
            this.drag = { kind, index, el: node };
            this.selected = { kind, index };
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
        this.$field.addEventListener('keydown', (ev) => {
            const node = ev.target.closest?.('.node');
            if (!node) return;
            const keyMap = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
            if (!(ev.key in keyMap)) return;
            ev.preventDefault();
            const index = Number(node.dataset.index);
            const kind = node.dataset.kind;
            const step = ev.shiftKey ? 10 : 2;
            const [dx, dy] = keyMap[ev.key];
            this._moveNode(kind, index, dx * step, dy * step);
            this.selected = { kind, index };
            this._draw();
            this._scheduleCompute(0);
        });
    }

    _moveFromEvent(ev) {
        const rect = this.$field.getBoundingClientRect();
        const field = this._fieldLength();
        const x = clamp((ev.clientX - rect.left) / rect.width * field, 0, field);
        const y = clamp((ev.clientY - rect.top) / rect.height * field, 0, field);
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
        this._drawBudgets();
        this._drawHeatmap();
    }

    _selectedPower() {
        return this.results?.[this.selectedMethod]?.power || new Array(this._k()).fill(0);
    }

    _drawField() {
        const svg = this.$field;
        svg.innerHTML = '';
        const field = this._fieldLength();
        svg.setAttribute('viewBox', `0 0 ${field} ${field}`);
        const power = this._selectedPower();
        const method = this.results?.[this.selectedMethod];
        const meta = this.last?.meta || this._metadata();
        const losses = this._hasCurrentLosses() ? this.last.losses : null;

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
            for (const e of edges.slice(0, Math.min(90, edges.length))) {
                const tx = this.blue[meta.groupIds[e.source]];
                const rx = this.rx[e.target];
                const op = clamp(0.07 + 0.36 * Math.sqrt(e.score / (maxScore + JSAC_EPS)), 0.07, 0.43);
                svg.appendChild(svgEl('line', {
                    x1: tx.x,
                    y1: tx.y,
                    x2: rx.x,
                    y2: rx.y,
                    stroke: `rgba(160,170,180,${op})`,
                    'stroke-width': 0.55,
                    'stroke-dasharray': '2 2.6',
                }));
            }
        }

        for (let i = 0; i < meta.k; i++) {
            const rx = this.rx[i];
            const tx = this.blue[rx.blue];
            const isYellow = rx.type === 'yellow';
            const color = isYellow ? '246,196,69' : '76,175,80';
            const p = power[i] || 0;
            svg.appendChild(svgEl('line', {
                x1: tx.x,
                y1: tx.y,
                x2: rx.x,
                y2: rx.y,
                stroke: `rgba(${color},${0.25 + p * 0.72})`,
                'stroke-width': 0.9 + p * 2.2,
            }));
        }

        for (let b = 0; b < this.B; b++) {
            const util = method?.groupUtil?.[b]?.total || 0;
            const blue = this.blue[b];
            svg.appendChild(svgEl('circle', {
                cx: blue.x,
                cy: blue.y,
                r: 7 + util * 7,
                fill: 'none',
                stroke: `rgba(77,163,255,${0.14 + util * 0.35})`,
                'stroke-width': 2,
            }));
            const group = svgEl('g', { class: 'node', tabindex: 0, 'data-kind': 'blue', 'data-index': b, role: 'button', 'aria-label': `Blue car ${b}` });
            group.appendChild(svgEl('circle', { class: 'focus-ring', cx: blue.x, cy: blue.y, r: 8.6, fill: 'none', stroke: 'var(--c-orange)', 'stroke-width': 1.5, opacity: 0 }));
            group.appendChild(svgEl('rect', { x: blue.x - 4.2, y: blue.y - 4.2, width: 8.4, height: 8.4, rx: 1.2, fill: 'var(--c-blue)', opacity: 0.95 }));
            group.appendChild(svgEl('text', { class: 'node-label', x: blue.x + 5.8, y: blue.y + 1.6 }));
            group.lastChild.textContent = `B${b}`;
            svg.appendChild(group);
        }

        for (let i = 0; i < meta.k; i++) {
            const rx = this.rx[i];
            const isYellow = rx.type === 'yellow';
            const badYellow = Boolean(isYellow && method && method.sinr[i] < this.manifest.physics.sinr_min);
            const fill = isYellow ? 'var(--c-yellow)' : 'var(--c-green)';
            const group = svgEl('g', { class: 'node', tabindex: 0, 'data-kind': 'rx', 'data-index': i, role: 'button', 'aria-label': `${isYellow ? 'Yellow' : 'Green'} receiver ${i}` });
            group.appendChild(svgEl('circle', { class: 'focus-ring', cx: rx.x, cy: rx.y, r: 5.3, fill: 'none', stroke: 'var(--c-orange)', 'stroke-width': 1.4, opacity: 0 }));
            if (badYellow) {
                group.appendChild(svgEl('circle', { cx: rx.x, cy: rx.y, r: 5.5, fill: 'none', stroke: 'rgba(255,99,99,0.95)', 'stroke-width': 1.5 }));
            }
            group.appendChild(svgEl('circle', { cx: rx.x, cy: rx.y, r: 3.2, fill, opacity: 0.92 }));
            group.appendChild(svgEl('text', { class: 'node-label', x: rx.x + 4.4, y: rx.y + 1.4 }));
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
        const p = this.results?.[this.selectedMethod]?.power?.[i];
        const sinr = method?.sinr?.[i];
        const rate = method?.rates?.[i];
        this.$selected.textContent = `${r.type.toUpperCase()}${i} - B${r.blue} ch${r.channel} / p ${fmt(p, 2)} / SINR ${fmt(sinr, 2)} / rate ${fmt(rate, 2)}`;
    }

    _drawMetrics() {
        const methods = ['WMMSE', 'GNN', 'Naive'];
        this.$metrics.innerHTML = '';
        for (const method of methods) {
            const r = this.results?.[method];
            const row = document.createElement('div');
            row.className = 'metric-row';
            row.classList.toggle('is-active', method === this.selectedMethod);
            row.addEventListener('click', () => {
                this.selectedMethod = method;
                this._renderMethodTabs();
                this._draw();
            });
            const viol = r ? `${r.yellowViolations}/${this.B * this.my}` : '--';
            row.innerHTML = `
                <span class="method-name" style="color:${this._methodColor(method)}">${method}</span>
                <span>
                    <span class="metric-main">${r ? fmt(r.greenSumRate, 2) : '--'}</span>
                    <span class="metric-sub"> green SR / yellow viol ${viol}</span>
                </span>
                <span class="pill">${r ? fmtMs(r.timeMs) : '--'} ms</span>
            `;
            this.$metrics.appendChild(row);
        }
        if (this.results?.GNN) {
            this._setStatus(`${this.results.GNN.engine} / JSAC GNN ${fmtMs(this.results.GNN.timeMs)} ms / B=${this.B} K=${this._k()}`);
        }
    }

    _drawBudgets() {
        this.$budgets.innerHTML = '';
        const r = this.results?.[this.selectedMethod];
        for (let b = 0; b < this.B; b++) {
            const util = r?.groupUtil?.[b] || { total: 0, yellow: 0, green: 0 };
            const yellowW = clamp(util.yellow, 0, 1) * 100;
            const greenW = clamp(util.green, 0, 1) * 100;
            const row = document.createElement('div');
            row.className = 'budget-row';
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
                svg.appendChild(svgEl('rect', {
                    x: j * size,
                    y: i * size,
                    width: Math.max(1, size - 0.6),
                    height: Math.max(1, size - 0.6),
                    fill: color,
                }));
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
