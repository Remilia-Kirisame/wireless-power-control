/* <wmmse-iter> — animated WMMSE iteration visualizer.
 *
 * Lives at the bottom of section 01 Problem. Three coordinated views of the
 * same algorithm state: (a) a tiny K-pair layout map where direct-link
 * brightness encodes current power, (b) a per-link power bar strip, and
 * (c) a sum-rate-vs-iteration convergence curve with a dashed asymptote
 * showing the converged target. Transport controls (Play / Step / Reset),
 * speed slider, K dropdown, and seed shuffle let the reader feel how the
 * iterative cost grows with K.
 *
 * Algorithm port of `Scenario_D2D/baselines.py:batch_WMMSE2` for one
 * topology (no batch dimension). Channels are sampled in JS via a seeded
 * PRNG using the same path-loss / Rayleigh-fading family as the codebase.
 */

// ───────────────────────────────────────────────────────────────────────
// Tunables
// ───────────────────────────────────────────────────────────────────────

// Maximum WMMSE iterations to run / display.
const MAX_ITER = 100;

// Allowed K values in the dropdown.
const K_CHOICES = [5, 8, 10, 15, 20];
const DEFAULT_K = 8;
const DEFAULT_SEED = 42;

// Pacing: at speed=1, one WMMSE iteration plays per BASE_MS_PER_ITER ms.
// Slow enough that early-iteration drama (links being silenced) is legible.
const BASE_MS_PER_ITER = 90;

// Wall-clock per-iteration estimates (ms), calibrated to the table in the
// existing problem section. Real WMMSE in NumPy at K=20 is ~52ms, K=40 ~250ms;
// we extrapolate inwards. These drive the "wall clock" readout — it advances
// by msPerIter[K] for each iteration completed, regardless of playback speed.
const MS_PER_ITER_BY_K = {
    5:  0.09,
    8:  0.20,
    10: 0.32,
    15: 0.55,
    20: 0.78,
};

// GNN single-shot inference cost (ms) by K, shown alongside the WMMSE
// wall-clock readout. The GNN scales with K too (more nodes / edges) but
// far more slowly than iterative WMMSE, so the contrast widens with K.
// Calibration is rough — anchored to the ● LIVE chip in the sidebar.
const GNN_MS_BY_K = {
    5:  3,
    8:  4,
    10: 5,
    15: 7,
    20: 9,
};

// Channel-sampling parameters (mirror interference-sandbox conventions).
const FIELD_SIZE  = 1000;   // m
const D0          = 200;    // m, reference distance
const PATHLOSS_G  = 3;      // exponent γ
const RX_RADIUS_LO = 30;    // m, min Tx-Rx distance
const RX_RADIUS_HI = 80;    // m, max Tx-Rx distance
const SHADOW_DB   = 0;      // log-normal σ (dB); 0 = disabled for clarity
const VAR_NOISE   = 1.0;    // normalized; tuned with PMAX so mid-K SINRs land sensibly
const PMAX        = 1.0;

// Speed slider: log-spaced multipliers.
const SPEED_LEVELS = [0.25, 0.5, 1, 2, 4, 8];
const DEFAULT_SPEED_INDEX = 2; // → 1×

// Layout panel rendering.
const LAYOUT_W = 380;
const LAYOUT_H = 260;
const LAYOUT_PAD = 20;
const TX_SIZE = 6;          // square half-side
const RX_RADIUS = 4.5;
const DIRECT_LINK_W = 1.6;
const INTERFERER_PER_RX = 2; // number of strongest interferers to draw per Rx
const INTERFERER_LINE_W = 0.8;

// Convergence chart rendering.
const CURVE_W = 380;
const CURVE_H = 260;
const CURVE_PAD_L = 44;
const CURVE_PAD_R = 14;
const CURVE_PAD_T = 14;
const CURVE_PAD_B = 32;

// Power-bar strip rendering. The SVG fills the host width; bars and text
// are recomputed from the live `clientWidth` on every render and a
// ResizeObserver re-renders the strip when the host resizes.
const STRIP_H = 92;
const STRIP_PAD_X = 14;
const STRIP_PAD_T = 12;
const STRIP_PAD_B = 22;
const BAR_GAP_RATIO = 0.32; // gap as fraction of slot width

// Anti-divide-by-zero.
const EPS = 1e-12;


// ───────────────────────────────────────────────────────────────────────
// Helpers — PRNG + Gaussian + channel sampling
// ───────────────────────────────────────────────────────────────────────

function mulberry32(seed) {
    let t = seed >>> 0;
    return function () {
        t = (t + 0x6D2B79F5) >>> 0;
        let r = Math.imul(t ^ (t >>> 15), t | 1);
        r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
}

// Box-Muller; consumes two uniforms per sample.
function gaussian(rand) {
    let u1 = rand();
    if (u1 < 1e-10) u1 = 1e-10;
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/* Sample K Tx/Rx pairs and the K×K channel-magnitude matrix.
 *
 * Returns { tx, rx, H } where:
 *   tx[i] = [x, y]     transmitter i in [0, FIELD_SIZE]^2
 *   rx[i] = [x, y]     receiver i, near tx[i]
 *   H[i][j]            magnitude of channel from Tx j to Rx i (sqrt of |h|^2)
 */
function sampleChannels(K, seed) {
    // Decorrelate K and seed so neighboring seeds at the same K give
    // visibly different layouts.
    const rand = mulberry32(seed * 0x9E3779B9 + K * 0x85EBCA6B);

    const tx = [];
    const rx = [];
    for (let i = 0; i < K; i++) {
        const txX = 50 + rand() * (FIELD_SIZE - 100);
        const txY = 50 + rand() * (FIELD_SIZE - 100);
        tx.push([txX, txY]);

        const r = RX_RADIUS_LO + rand() * (RX_RADIUS_HI - RX_RADIUS_LO);
        const theta = rand() * 2 * Math.PI;
        rx.push([txX + r * Math.cos(theta), txY + r * Math.sin(theta)]);
    }

    const H = [];
    for (let i = 0; i < K; i++) {
        const row = [];
        for (let j = 0; j < K; j++) {
            const dx = tx[j][0] - rx[i][0];
            const dy = tx[j][1] - rx[i][1];
            const d = Math.max(1, Math.sqrt(dx * dx + dy * dy));
            const pl = Math.pow(D0 / d, PATHLOSS_G);

            let shadow = 1;
            if (SHADOW_DB > 0) {
                shadow = Math.pow(10, gaussian(rand) * SHADOW_DB / 10);
            }

            const x = gaussian(rand);
            const y = gaussian(rand);
            const fading = 0.5 * (x * x + y * y); // |h|^2 ~ Exp(1) effectively

            const gain = pl * shadow * fading;
            row.push(Math.sqrt(Math.max(gain, 1e-20)));
        }
        H.push(row);
    }
    return { tx, rx, H };
}


// ───────────────────────────────────────────────────────────────────────
// Helpers — WMMSE algorithm (single topology port of batch_WMMSE2)
// ───────────────────────────────────────────────────────────────────────

/* Mutates `state` in place. State shape:
 *   { K, H, alpha, b, f, w }
 * After init, `b` holds the precoder (sqrt of power); call stepWMMSE() to
 * advance one outer iteration (b update + filter/weight update).
 */
function initWMMSE(K, H) {
    const b = new Array(K).fill(Math.sqrt(PMAX));
    const alpha = new Array(K).fill(1.0);
    const f = new Array(K);
    const w = new Array(K);
    for (let i = 0; i < K; i++) {
        let interference = VAR_NOISE;
        for (let j = 0; j < K; j++) {
            const rxp = H[i][j] * b[j];
            interference += rxp * rxp;
        }
        const validRx = H[i][i] * b[i];
        f[i] = validRx / (interference + EPS);
        w[i] = 1 / (1 - f[i] * validRx + EPS);
    }
    return { K, H, alpha, b, f, w };
}

function stepWMMSE(s) {
    const { K, H, alpha, b, f, w } = s;

    // Update b (precoder)
    const bNew = new Array(K);
    const sqrtPmax = Math.sqrt(PMAX);
    for (let j = 0; j < K; j++) {
        const bup = alpha[j] * w[j] * H[j][j] * f[j];
        let bdown = 0;
        for (let i = 0; i < K; i++) {
            const rxd = H[i][j] * f[i];
            bdown += alpha[i] * w[i] * rxd * rxd;
        }
        let btmp = bup / (bdown + EPS);
        if (btmp < 0) btmp = 0;
        if (btmp > sqrtPmax) btmp = sqrtPmax;
        bNew[j] = btmp;
    }
    s.b = bNew;

    // Update f, w
    for (let i = 0; i < K; i++) {
        let interference = VAR_NOISE;
        for (let j = 0; j < K; j++) {
            const rxp = H[i][j] * s.b[j];
            interference += rxp * rxp;
        }
        const validRx = H[i][i] * s.b[i];
        s.f[i] = validRx / (interference + EPS);
        s.w[i] = 1 / (1 - s.f[i] * validRx + EPS);
    }
}

function powerVector(s) {
    const p = new Array(s.K);
    for (let i = 0; i < s.K; i++) p[i] = s.b[i] * s.b[i];
    return p;
}

function sumRate(K, H, p) {
    let sr = 0;
    for (let i = 0; i < K; i++) {
        const sig = p[i] * H[i][i] * H[i][i];
        let interf = VAR_NOISE;
        for (let j = 0; j < K; j++) {
            if (j !== i) interf += p[j] * H[i][j] * H[i][j];
        }
        sr += Math.log2(1 + sig / interf);
    }
    return sr;
}

// Run the full WMMSE trajectory and return per-iteration sum-rate + final p.
// Used to pre-compute the asymptote target for the convergence chart.
function runFullTrajectory(K, H) {
    const s = initWMMSE(K, H);
    const trajectory = new Array(MAX_ITER + 1);
    trajectory[0] = sumRate(K, H, powerVector(s));
    for (let it = 1; it <= MAX_ITER; it++) {
        stepWMMSE(s);
        trajectory[it] = sumRate(K, H, powerVector(s));
    }
    return trajectory;
}


// ───────────────────────────────────────────────────────────────────────
// Helpers — small SVG / formatting utilities
// ───────────────────────────────────────────────────────────────────────

function svgEl(name, attrs = {}) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [k, v] of Object.entries(attrs)) {
        if (v !== null && v !== undefined) el.setAttribute(k, String(v));
    }
    return el;
}

function fmtFixed(x, d = 2) {
    if (!Number.isFinite(x)) return '—';
    return x.toFixed(d);
}

function fmtMs(x) {
    if (!Number.isFinite(x)) return '—';
    if (x >= 100) return x.toFixed(0);
    if (x >= 10)  return x.toFixed(1);
    return x.toFixed(2);
}


// ───────────────────────────────────────────────────────────────────────
// Template
// ───────────────────────────────────────────────────────────────────────

const TEMPLATE = document.createElement('template');
TEMPLATE.innerHTML = /* html */ `
    <style>
        :host {
            display: block;
            background: var(--surface);
            border: 1px solid var(--rule);
            border-radius: var(--radius-card);
            padding: 20px 22px 18px;
            color: var(--text);
            font-family: var(--font-sans);
        }

        /* ── Header row ── */
        .head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 18px;
            margin-bottom: 14px;
            flex-wrap: wrap;
        }
        .eyebrow {
            font-family: var(--font-mono);
            font-size: 11px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--text-dim);
        }
        .eyebrow::before {
            content: '● ';
            color: var(--c-orange);
            margin-right: 2px;
        }
        .iter-meta {
            font-family: var(--font-mono);
            font-size: 12px;
            color: var(--text-dim);
            letter-spacing: 0.06em;
            font-variant-numeric: tabular-nums;
            font-feature-settings: "tnum" 1;
        }
        .iter-meta .num {
            color: var(--text);
            font-weight: 500;
        }

        /* ── Top: layout + curve side-by-side ── */
        .panels {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 14px;
        }
        .panel {
            background: rgba(0,0,0,0.18);
            border: 1px solid var(--rule);
            border-radius: 8px;
            padding: 10px 12px 8px;
            position: relative;
        }
        .panel-label {
            position: absolute;
            top: 10px;
            left: 12px;
            font-family: var(--font-mono);
            font-size: 10px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--text-mute);
            pointer-events: none;
        }
        svg { width: 100%; height: auto; display: block; overflow: visible; }

        /* ── Layout panel SVG styles ── */
        .field-edge {
            fill: none;
            stroke: rgba(255,255,255,0.08);
            stroke-dasharray: 2 4;
        }
        .interf-line {
            stroke: rgba(255,255,255,0.18);
            stroke-width: 0.8;
            stroke-dasharray: 1.5 3;
            fill: none;
            pointer-events: none;
        }
        .direct-line {
            stroke: var(--c-orange);
            stroke-linecap: round;
            fill: none;
            transition: opacity 80ms linear, stroke-width 120ms var(--ease);
        }
        .tx-mark {
            fill: var(--c-orange);
            transition: opacity 80ms linear;
        }
        .rx-mark {
            fill: var(--c-orange);
            transition: opacity 80ms linear;
        }
        .pair-hit {
            fill: transparent;
            stroke: transparent;
            cursor: pointer;
        }
        .pair-ring {
            fill: none;
            stroke: var(--text);
            stroke-width: 1;
            opacity: 0;
            transition: opacity 100ms var(--ease);
            pointer-events: none;
        }
        .pair-group.is-hot .pair-ring {
            opacity: 0.55;
        }
        .pair-group.is-hot .direct-line {
            stroke-width: ${DIRECT_LINK_W * 1.8};
        }

        /* ── Convergence chart styles ── */
        .grid-line { stroke: rgba(255,255,255,0.05); stroke-width: 1; }
        .axis-tick {
            font-family: var(--font-mono);
            font-size: 10px;
            fill: var(--text-mute);
            letter-spacing: 0.04em;
        }
        .axis-label {
            font-family: var(--font-mono);
            font-size: 10px;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            fill: var(--text-mute);
        }
        .target-line {
            stroke: rgba(255,255,255,0.32);
            stroke-width: 1;
            stroke-dasharray: 4 4;
        }
        .target-label {
            font-family: var(--font-mono);
            font-size: 10px;
            fill: var(--text-dim);
            letter-spacing: 0.04em;
        }
        .curve-line {
            fill: none;
            stroke: var(--c-orange);
            stroke-width: 2;
            stroke-linejoin: round;
            stroke-linecap: round;
        }
        .curve-halo {
            fill: none;
            stroke: var(--c-orange);
            stroke-width: 7;
            opacity: 0.18;
            filter: blur(2px);
        }
        .curve-head {
            fill: var(--c-orange);
        }
        .curve-baseline {
            stroke: rgba(255,255,255,0.22);
            stroke-width: 1;
            stroke-dasharray: 2 4;
        }

        /* ── Power-bar strip styles ── */
        .strip-wrap { margin-bottom: 14px; }
        .strip-track { stroke: var(--rule); stroke-width: 1; }
        .bar {
            fill: var(--c-orange);
            transition: opacity 80ms linear;
        }
        .bar-bg {
            fill: rgba(255,255,255,0.04);
        }
        .bar-hit {
            fill: transparent;
            cursor: pointer;
        }
        .bar-label {
            font-family: var(--font-mono);
            font-size: 9px;
            fill: var(--text-mute);
            text-anchor: middle;
            letter-spacing: 0.04em;
        }
        .bar-group.is-hot .bar { opacity: 1 !important; }
        .bar-group.is-hot .bar-label { fill: var(--text); }

        /* ── Controls row ── */
        .controls {
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
            padding: 10px 0 4px;
            border-top: 1px solid var(--rule);
        }
        .btn-group { display: inline-flex; gap: 6px; }
        button.tbtn {
            background: rgba(255,255,255,0.04);
            color: var(--text-dim);
            border: 1px solid var(--rule);
            border-radius: 6px;
            padding: 6px 12px;
            font-family: var(--font-mono);
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            cursor: pointer;
            transition: color var(--dur-fast) var(--ease),
                        background var(--dur-fast) var(--ease),
                        border-color var(--dur-fast) var(--ease);
        }
        button.tbtn:hover { color: var(--text); border-color: rgba(255,255,255,0.18); }
        button.tbtn.is-primary {
            background: rgba(255, 106, 61, 0.14);
            color: var(--text);
            border-color: rgba(255, 106, 61, 0.4);
        }
        button.tbtn.is-primary:hover { background: rgba(255, 106, 61, 0.22); }

        .control-group {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-family: var(--font-mono);
            font-size: 10px;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-mute);
        }
        .control-group select,
        .control-group input[type="range"] {
            font-family: inherit;
        }
        select.k-pick {
            background: rgba(0,0,0,0.3);
            color: var(--text);
            border: 1px solid var(--rule);
            border-radius: 5px;
            padding: 4px 8px;
            font-size: 11px;
            letter-spacing: 0.06em;
            cursor: pointer;
        }
        select.k-pick:hover { border-color: rgba(255,255,255,0.18); }

        input[type="range"].speed {
            accent-color: var(--c-orange);
            width: 110px;
        }

        .seed-display {
            color: var(--text-dim);
            font-variant-numeric: tabular-nums;
            min-width: 28px;
            text-align: right;
        }
        .spacer { flex: 1; }

        /* ── Readout panel ── */
        .readout {
            margin-top: 14px;
            padding: 12px 14px;
            background: rgba(0,0,0,0.22);
            border: 1px solid var(--rule);
            border-radius: 8px;
            display: grid;
            grid-template-columns: max-content 1fr max-content max-content;
            row-gap: 6px;
            column-gap: 16px;
            font-family: var(--font-mono);
            font-size: 12px;
            letter-spacing: 0.04em;
            font-variant-numeric: tabular-nums;
            font-feature-settings: "tnum" 1;
        }
        .readout .label { color: var(--text-mute); text-transform: uppercase; letter-spacing: 0.1em; font-size: 10px; }
        .readout .val   { color: var(--text); font-weight: 500; }
        .readout .arrow { color: var(--text-mute); }
        .readout .note  { color: var(--text-dim); }
        .readout .note .accent { color: var(--c-orange); }

        .progress-track {
            grid-column: 1 / -1;
            height: 3px;
            background: rgba(255,255,255,0.08);
            border-radius: 2px;
            margin-top: 4px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: var(--c-orange);
            box-shadow: 0 0 8px rgba(255, 106, 61, 0.4);
            width: 0;
            transition: width 60ms linear;
        }

        @media (max-width: 720px) {
            .panels { grid-template-columns: 1fr; }
            .readout { grid-template-columns: max-content 1fr; }
            .readout .arrow, .readout .note { grid-column: 1 / -1; padding-left: 0; }
        }
        @media (prefers-reduced-motion: reduce) {
            .direct-line, .tx-mark, .rx-mark, .bar, .progress-fill { transition: none; }
        }
    </style>

    <div class="head">
        <span class="eyebrow">interactive · iterate WMMSE</span>
        <span class="iter-meta">iter <span class="num" data-iter>0</span> / ${MAX_ITER} · K=<span class="num" data-k>${DEFAULT_K}</span></span>
    </div>

    <div class="panels">
        <div class="panel">
            <span class="panel-label">layout · brightness = power</span>
            <svg data-layout-svg viewBox="0 0 ${LAYOUT_W} ${LAYOUT_H}" preserveAspectRatio="xMidYMid meet"></svg>
        </div>
        <div class="panel">
            <span class="panel-label">sum-rate vs iteration</span>
            <svg data-curve-svg viewBox="0 0 ${CURVE_W} ${CURVE_H}" preserveAspectRatio="xMidYMid meet"></svg>
        </div>
    </div>

    <div class="strip-wrap">
        <svg data-strip-svg preserveAspectRatio="none"></svg>
    </div>

    <div class="controls">
        <div class="btn-group">
            <button class="tbtn is-primary" data-play aria-label="Play / pause WMMSE iteration">▶ Play</button>
            <button class="tbtn" data-step aria-label="Step one iteration">⏭ Step</button>
            <button class="tbtn" data-reset aria-label="Reset to iteration 0">↺ Reset</button>
        </div>
        <div class="control-group">
            <span>speed</span>
            <input type="range" class="speed" data-speed min="0" max="${SPEED_LEVELS.length - 1}" step="1" value="${DEFAULT_SPEED_INDEX}" aria-label="Playback speed" />
            <span class="seed-display" data-speed-label>1×</span>
        </div>
        <div class="spacer"></div>
        <div class="control-group">
            <span>K</span>
            <select class="k-pick" data-k-pick aria-label="Number of links">
                ${K_CHOICES.map(k => `<option value="${k}"${k === DEFAULT_K ? ' selected' : ''}>${k}</option>`).join('')}
            </select>
        </div>
        <div class="control-group">
            <span>seed</span>
            <span class="seed-display" data-seed>${DEFAULT_SEED}</span>
            <button class="tbtn" data-shuffle aria-label="Shuffle channel layout">↻</button>
        </div>
    </div>

    <div class="readout">
        <span class="label">iter</span>
        <span class="val"><span data-iter-text>0</span> / ${MAX_ITER}</span>
        <span class="arrow"></span>
        <span class="note"></span>

        <span class="label">sum-rate</span>
        <span class="val"><span data-sumrate>—</span> b/s/Hz</span>
        <span class="arrow">→</span>
        <span class="note">converges at <span class="num" data-target>—</span></span>

        <span class="label">wall clock</span>
        <span class="val"><span data-wallms>0.00</span> ms</span>
        <span class="arrow">→</span>
        <span class="note">GNN: <span class="accent">~<span data-gnnms>${GNN_MS_BY_K[DEFAULT_K]}</span> ms</span> (1 forward pass)</span>

        <div class="progress-track"><div class="progress-fill" data-progress></div></div>
    </div>
`;


// ───────────────────────────────────────────────────────────────────────
// Component
// ───────────────────────────────────────────────────────────────────────

class WmmseIter extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.appendChild(TEMPLATE.content.cloneNode(true));

        // Algorithm + topology state
        this.K = DEFAULT_K;
        this.seed = DEFAULT_SEED;
        this.tx = null;
        this.rx = null;
        this.H = null;
        this.state = null;             // WMMSE state (b/f/w + refs)
        this.iter = 0;
        this.targetTrajectory = null;  // pre-computed sum-rate per iter (length MAX_ITER+1)
        this.curveValues = [];         // sum-rate per iter so far (length iter+1)

        // Animation state
        this.playing = false;
        this.speedIndex = DEFAULT_SPEED_INDEX;
        this.iterAccum = 0;
        this.lastTs = 0;
        this.rafId = 0;

        // Interaction state
        this.hotLink = -1; // index of currently-highlighted link, -1 = none

        // Reduced-motion preference
        this.reduced = matchMedia('(prefers-reduced-motion: reduce)');

        // Element refs
        const $ = (sel) => this.shadowRoot.querySelector(sel);
        this.refs = {
            iterMeta:    $('[data-iter]'),
            kMeta:       $('[data-k]'),
            layoutSvg:   $('[data-layout-svg]'),
            curveSvg:    $('[data-curve-svg]'),
            stripSvg:    $('[data-strip-svg]'),
            playBtn:     $('[data-play]'),
            stepBtn:     $('[data-step]'),
            resetBtn:    $('[data-reset]'),
            speed:       $('[data-speed]'),
            speedLabel:  $('[data-speed-label]'),
            kPick:       $('[data-k-pick]'),
            seedDisplay: $('[data-seed]'),
            shuffleBtn:  $('[data-shuffle]'),
            iterText:    $('[data-iter-text]'),
            sumrate:     $('[data-sumrate]'),
            target:      $('[data-target]'),
            wallms:      $('[data-wallms]'),
            gnnms:       $('[data-gnnms]'),
            progress:    $('[data-progress]'),
        };
    }

    connectedCallback() {
        this.refs.playBtn.addEventListener('click', () => this._togglePlay());
        this.refs.stepBtn.addEventListener('click', () => this._doStep());
        this.refs.resetBtn.addEventListener('click', () => this._reset());
        this.refs.speed.addEventListener('input', (e) => {
            this.speedIndex = Number(e.target.value);
            this._updateSpeedLabel();
        });
        this.refs.kPick.addEventListener('change', (e) => {
            this.K = Number(e.target.value);
            this.refs.kMeta.textContent = String(this.K);
            this._resampleAndReset();
        });
        this.refs.shuffleBtn.addEventListener('click', () => {
            this.seed = (this.seed + 1) >>> 0;
            this.refs.seedDisplay.textContent = String(this.seed);
            this._resampleAndReset();
        });

        this.refs.layoutSvg.setAttribute('width', LAYOUT_W);
        this.refs.layoutSvg.setAttribute('height', LAYOUT_H);
        this.refs.curveSvg.setAttribute('width', CURVE_W);
        this.refs.curveSvg.setAttribute('height', CURVE_H);

        this._updateSpeedLabel();
        this._resampleAndReset();

        // Re-render the strip on host width changes so bar slots and labels
        // stay correct across viewport resizes / sidebar collapses.
        this.resizeObs = new ResizeObserver(() => this._renderStrip());
        this.resizeObs.observe(this.refs.stripSvg);
    }

    disconnectedCallback() {
        if (this.rafId) cancelAnimationFrame(this.rafId);
        if (this.resizeObs) this.resizeObs.disconnect();
    }

    // ──────────────── State transitions ────────────────

    _resampleAndReset() {
        const sampled = sampleChannels(this.K, this.seed);
        this.tx = sampled.tx;
        this.rx = sampled.rx;
        this.H  = sampled.H;
        this.targetTrajectory = runFullTrajectory(this.K, this.H);
        this._reset();
    }

    _reset() {
        this._stopPlayback();
        this.state = initWMMSE(this.K, this.H);
        this.iter = 0;
        this.iterAccum = 0;
        this.curveValues = [this.targetTrajectory[0]];
        this.hotLink = -1;
        this.refs.playBtn.textContent = '▶ Play';
        this._renderAll();
    }

    _togglePlay() {
        if (this.playing) {
            this._stopPlayback();
            this.refs.playBtn.textContent = '▶ Play';
            return;
        }
        if (this.iter >= MAX_ITER) {
            // Auto-reset before replaying so the curve animates from the start.
            this._reset();
        }
        this._startPlayback();
    }

    _startPlayback() {
        if (this.reduced.matches) {
            // No animation; jump to the converged state.
            while (this.iter < MAX_ITER) this._advanceOne();
            this._renderAll();
            return;
        }
        this.playing = true;
        this.refs.playBtn.textContent = '⏸ Pause';
        this.lastTs = 0;
        this.iterAccum = 0;
        const tick = (ts) => {
            if (!this.playing) return;
            if (this.lastTs === 0) this.lastTs = ts;
            const dt = ts - this.lastTs;
            this.lastTs = ts;

            const speed = SPEED_LEVELS[this.speedIndex];
            this.iterAccum += (dt / BASE_MS_PER_ITER) * speed;

            let stepped = false;
            while (this.iterAccum >= 1 && this.iter < MAX_ITER) {
                this._advanceOne();
                this.iterAccum -= 1;
                stepped = true;
            }
            if (stepped) this._renderAll();

            if (this.iter >= MAX_ITER) {
                this._stopPlayback();
                this.refs.playBtn.textContent = '▶ Play';
                return;
            }
            this.rafId = requestAnimationFrame(tick);
        };
        this.rafId = requestAnimationFrame(tick);
    }

    _stopPlayback() {
        this.playing = false;
        if (this.rafId) {
            cancelAnimationFrame(this.rafId);
            this.rafId = 0;
        }
    }

    _doStep() {
        this._stopPlayback();
        this.refs.playBtn.textContent = '▶ Play';
        if (this.iter >= MAX_ITER) return;
        this._advanceOne();
        this._renderAll();
    }

    _advanceOne() {
        stepWMMSE(this.state);
        this.iter += 1;
        this.curveValues.push(this.targetTrajectory[this.iter]);
        // Use the pre-computed trajectory value for the curve (numerically
        // identical, just avoids redundant per-frame sumRate work).
    }

    _updateSpeedLabel() {
        const s = SPEED_LEVELS[this.speedIndex];
        this.refs.speedLabel.textContent = (s < 1 ? s : Math.round(s)) + '×';
    }

    // ──────────────── Rendering ────────────────

    _renderAll() {
        this._renderMeta();
        this._renderLayout();
        this._renderCurve();
        this._renderStrip();
        this._renderReadout();
    }

    _renderMeta() {
        this.refs.iterMeta.textContent = String(this.iter);
    }

    _renderLayout() {
        const svg = this.refs.layoutSvg;
        svg.innerHTML = '';

        // Field boundary (faint dashed rect)
        svg.appendChild(svgEl('rect', {
            class: 'field-edge',
            x: LAYOUT_PAD * 0.5,
            y: LAYOUT_PAD * 0.5,
            width:  LAYOUT_W - LAYOUT_PAD,
            height: LAYOUT_H - LAYOUT_PAD,
            rx: 4,
        }));

        const innerW = LAYOUT_W - 2 * LAYOUT_PAD;
        const innerH = LAYOUT_H - 2 * LAYOUT_PAD;

        // Map from world coords [0..FIELD_SIZE] to SVG coords with padding.
        // World is square; if SVG aspect differs we letterbox to keep shapes correct.
        const scale = Math.min(innerW / FIELD_SIZE, innerH / FIELD_SIZE);
        const drawnW = FIELD_SIZE * scale;
        const drawnH = FIELD_SIZE * scale;
        const offX = LAYOUT_PAD + (innerW - drawnW) / 2;
        const offY = LAYOUT_PAD + (innerH - drawnH) / 2;
        const X = (x) => offX + x * scale;
        const Y = (y) => offY + y * scale;

        // Per-link normalized power for opacity scaling.
        const p = powerVector(this.state);

        // Interference layer: for each Rx, draw the strongest INTERFERER_PER_RX
        // incoming Tx-from-other-link lines. Static (does not depend on power).
        const K = this.K;
        const interfFrag = document.createDocumentFragment();
        for (let i = 0; i < K; i++) {
            const candidates = [];
            for (let j = 0; j < K; j++) {
                if (j === i) continue;
                candidates.push({ j, gain: this.H[i][j] * this.H[i][j] });
            }
            candidates.sort((a, b) => b.gain - a.gain);
            for (let m = 0; m < Math.min(INTERFERER_PER_RX, candidates.length); m++) {
                const j = candidates[m].j;
                interfFrag.appendChild(svgEl('line', {
                    class: 'interf-line',
                    x1: X(this.tx[j][0]), y1: Y(this.tx[j][1]),
                    x2: X(this.rx[i][0]), y2: Y(this.rx[i][1]),
                }));
            }
        }
        svg.appendChild(interfFrag);

        // Direct links + Tx + Rx, grouped per pair so hover highlights are easy.
        for (let i = 0; i < K; i++) {
            const power01 = Math.max(0, Math.min(1, p[i] / PMAX));
            const opacity = 0.15 + 0.85 * power01;

            const g = svgEl('g', { class: 'pair-group', 'data-pair-index': i });
            if (i === this.hotLink) g.classList.add('is-hot');

            // Direct line Tx → Rx
            g.appendChild(svgEl('line', {
                class: 'direct-line',
                x1: X(this.tx[i][0]), y1: Y(this.tx[i][1]),
                x2: X(this.rx[i][0]), y2: Y(this.rx[i][1]),
                'stroke-width': DIRECT_LINK_W,
                opacity: opacity,
            }));

            // Tx square
            g.appendChild(svgEl('rect', {
                class: 'tx-mark',
                x: X(this.tx[i][0]) - TX_SIZE / 2,
                y: Y(this.tx[i][1]) - TX_SIZE / 2,
                width: TX_SIZE, height: TX_SIZE,
                opacity: opacity,
            }));

            // Rx circle
            g.appendChild(svgEl('circle', {
                class: 'rx-mark',
                cx: X(this.rx[i][0]), cy: Y(this.rx[i][1]),
                r: RX_RADIUS,
                opacity: opacity,
            }));

            // Highlight ring around Rx (shows on hover)
            g.appendChild(svgEl('circle', {
                class: 'pair-ring',
                cx: X(this.rx[i][0]), cy: Y(this.rx[i][1]),
                r: RX_RADIUS + 4,
            }));

            // Invisible thick hit area along the link for hover
            g.appendChild(svgEl('line', {
                class: 'pair-hit',
                x1: X(this.tx[i][0]), y1: Y(this.tx[i][1]),
                x2: X(this.rx[i][0]), y2: Y(this.rx[i][1]),
                'stroke-width': 14,
                stroke: 'transparent',
            }));

            g.addEventListener('mouseenter', () => this._setHot(i));
            g.addEventListener('mouseleave', () => this._setHot(-1));
            svg.appendChild(g);
        }
    }

    _renderCurve() {
        const svg = this.refs.curveSvg;
        svg.innerHTML = '';

        const target = this.targetTrajectory[MAX_ITER];
        const initial = this.targetTrajectory[0];
        // Extend y range slightly beyond [initial, target] so the curve has
        // room to breathe and the dashed asymptote sits below the top edge.
        const yLo = Math.min(initial, target) - Math.max(0.5, 0.06 * Math.abs(target));
        const yHi = Math.max(initial, target) + Math.max(0.5, 0.06 * Math.abs(target));

        const px = (i) => CURVE_PAD_L + (i / MAX_ITER) * (CURVE_W - CURVE_PAD_L - CURVE_PAD_R);
        const py = (v) => CURVE_PAD_T + (1 - (v - yLo) / (yHi - yLo)) * (CURVE_H - CURVE_PAD_T - CURVE_PAD_B);

        // Y gridlines + ticks
        const nY = 4;
        for (let i = 0; i <= nY; i++) {
            const v = yLo + (i / nY) * (yHi - yLo);
            const y = py(v);
            svg.appendChild(svgEl('line', {
                class: 'grid-line',
                x1: CURVE_PAD_L, x2: CURVE_W - CURVE_PAD_R,
                y1: y, y2: y,
            }));
            const t = svgEl('text', {
                class: 'axis-tick',
                x: CURVE_PAD_L - 6, y: y + 3,
                'text-anchor': 'end',
            });
            t.textContent = fmtFixed(v, 1);
            svg.appendChild(t);
        }

        // X ticks
        const xTicks = [0, 25, 50, 75, 100];
        for (const xt of xTicks) {
            const x = px(xt);
            svg.appendChild(svgEl('line', {
                class: 'grid-line',
                x1: x, x2: x,
                y1: CURVE_PAD_T, y2: CURVE_H - CURVE_PAD_B,
            }));
            const t = svgEl('text', {
                class: 'axis-tick',
                x: x, y: CURVE_H - CURVE_PAD_B + 14,
                'text-anchor': 'middle',
            });
            t.textContent = String(xt);
            svg.appendChild(t);
        }

        // Axis labels
        const xLab = svgEl('text', {
            class: 'axis-label',
            x: (CURVE_PAD_L + CURVE_W - CURVE_PAD_R) / 2,
            y: CURVE_H - 4,
            'text-anchor': 'middle',
        });
        xLab.textContent = 'iteration';
        svg.appendChild(xLab);

        const yLab = svgEl('text', {
            class: 'axis-label',
            x: CURVE_PAD_L - 32,
            y: CURVE_PAD_T + 6,
            'text-anchor': 'start',
        });
        yLab.textContent = 'b/s/Hz';
        svg.appendChild(yLab);

        // Initial / baseline horizontal at iter 0 (faint)
        const yInit = py(initial);
        svg.appendChild(svgEl('line', {
            class: 'curve-baseline',
            x1: CURVE_PAD_L, x2: CURVE_W - CURVE_PAD_R,
            y1: yInit, y2: yInit,
        }));

        // Target asymptote (dashed)
        const yTar = py(target);
        svg.appendChild(svgEl('line', {
            class: 'target-line',
            x1: CURVE_PAD_L, x2: CURVE_W - CURVE_PAD_R,
            y1: yTar, y2: yTar,
        }));
        const tarLab = svgEl('text', {
            class: 'target-label',
            x: CURVE_W - CURVE_PAD_R - 4,
            y: yTar - 4,
            'text-anchor': 'end',
        });
        tarLab.textContent = 'target ' + fmtFixed(target, 2);
        svg.appendChild(tarLab);

        // Trajectory polyline up to current iter
        if (this.curveValues.length >= 2) {
            const pts = this.curveValues.map((v, i) => `${px(i)},${py(v)}`).join(' ');
            svg.appendChild(svgEl('polyline', { class: 'curve-halo', points: pts }));
            svg.appendChild(svgEl('polyline', { class: 'curve-line', points: pts }));
        }
        // Head dot at the latest iter
        const lastIdx = this.curveValues.length - 1;
        if (lastIdx >= 0) {
            svg.appendChild(svgEl('circle', {
                class: 'curve-head',
                cx: px(lastIdx),
                cy: py(this.curveValues[lastIdx]),
                r: 3.2,
            }));
        }
    }

    _renderStrip() {
        const svg = this.refs.stripSvg;
        svg.innerHTML = '';

        // The strip is full-bleed; use the live host width so bar widths
        // and labels read correctly across viewports. Re-rendered on resize
        // via the ResizeObserver wired up in connectedCallback.
        const w = Math.max(this.refs.stripSvg.clientWidth || 0, 320);
        svg.setAttribute('viewBox', `0 0 ${w} ${STRIP_H}`);
        svg.setAttribute('width',  w);
        svg.setAttribute('height', STRIP_H);

        const K = this.K;
        const trackY0 = STRIP_PAD_T;
        const trackY1 = STRIP_H - STRIP_PAD_B;
        const trackH = trackY1 - trackY0;

        // Baseline rule
        svg.appendChild(svgEl('line', {
            class: 'strip-track',
            x1: STRIP_PAD_X, x2: w - STRIP_PAD_X,
            y1: trackY1 + 0.5, y2: trackY1 + 0.5,
        }));

        const usableW = w - 2 * STRIP_PAD_X;
        const slotW = usableW / K;
        const barW = slotW * (1 - BAR_GAP_RATIO);
        const p = powerVector(this.state);

        for (let i = 0; i < K; i++) {
            const cx = STRIP_PAD_X + (i + 0.5) * slotW;
            const power01 = Math.max(0, Math.min(1, p[i] / PMAX));
            const h = power01 * trackH;
            const opacity = 0.25 + 0.75 * power01;

            const g = svgEl('g', { class: 'bar-group', 'data-bar-index': i });
            if (i === this.hotLink) g.classList.add('is-hot');

            // Background slot (faint full-height)
            g.appendChild(svgEl('rect', {
                class: 'bar-bg',
                x: cx - barW / 2, y: trackY0,
                width: barW, height: trackH,
                rx: 2,
            }));

            // Power bar
            g.appendChild(svgEl('rect', {
                class: 'bar',
                x: cx - barW / 2,
                y: trackY1 - h,
                width: barW, height: h,
                rx: 2,
                opacity,
            }));

            // Index label
            const lab = svgEl('text', {
                class: 'bar-label',
                x: cx,
                y: STRIP_H - 6,
            });
            lab.textContent = String(i + 1);
            g.appendChild(lab);

            // Hit overlay covering the slot
            g.appendChild(svgEl('rect', {
                class: 'bar-hit',
                x: cx - slotW / 2, y: 0,
                width: slotW, height: STRIP_H,
            }));

            g.addEventListener('mouseenter', () => this._setHot(i));
            g.addEventListener('mouseleave', () => this._setHot(-1));
            svg.appendChild(g);
        }
    }

    _renderReadout() {
        const target = this.targetTrajectory[MAX_ITER];
        const cur    = this.curveValues[this.curveValues.length - 1];
        const wallMs = this.iter * (MS_PER_ITER_BY_K[this.K] || 0.3);
        this.refs.iterText.textContent = String(this.iter);
        this.refs.sumrate.textContent  = fmtFixed(cur, 2);
        this.refs.target.textContent   = fmtFixed(target, 2);
        this.refs.wallms.textContent   = fmtMs(wallMs);
        this.refs.gnnms.textContent    = String(GNN_MS_BY_K[this.K] || 5);
        this.refs.progress.style.width = (100 * this.iter / MAX_ITER) + '%';
    }

    _setHot(i) {
        if (this.hotLink === i) return;
        this.hotLink = i;
        // Cheap targeted toggle without re-rendering everything.
        for (const g of this.shadowRoot.querySelectorAll('.pair-group')) {
            const idx = Number(g.getAttribute('data-pair-index'));
            g.classList.toggle('is-hot', idx === i);
        }
        for (const g of this.shadowRoot.querySelectorAll('.bar-group')) {
            const idx = Number(g.getAttribute('data-bar-index'));
            g.classList.toggle('is-hot', idx === i);
        }
    }
}

customElements.define('wmmse-iter', WmmseIter);
