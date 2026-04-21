/* <interference-sandbox> — drag-a-Tx, watch channels change.
 *
 * Pure physics. Path loss = (d0 / max(d, d0))^γ. No model inference.
 *
 * Renders two panels side-by-side:
 *   (1) Field with K transmitters and K co-located receivers; one Tx is draggable.
 *   (2) K × K channel-gain heatmap (log-scaled) that updates live.
 *
 * Parameters (documented inline): γ = 3, d0 = 10m, field = 200m, K = 5.
 *
 * Keyboard: when the active Tx is focused, arrow keys nudge ±1m, Shift+arrow ±10m.
 */

const FIELD = 200;  // m
const D0    = 10;   // m — reference distance
const GAMMA = 3;    // path-loss exponent

const TEMPLATE = document.createElement('template');
TEMPLATE.innerHTML = /* html */ `
    <style>
        :host {
            display: block;
            background: var(--surface);
            border: 1px solid var(--rule);
            border-radius: var(--radius-card);
            padding: 20px 22px;
            color: var(--text);
            font-family: var(--font-sans);
        }
        .head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
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
        }
        .readout {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-dim);
            letter-spacing: 0.04em;
        }
        .readout .val {
            color: var(--text);
            font-variant-numeric: tabular-nums;
            margin-left: 6px;
        }

        .stage {
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 20px;
            align-items: stretch;
        }
        @media (max-width: 720px) {
            .stage { grid-template-columns: 1fr; }
        }

        .panel-label {
            font-family: var(--font-mono);
            font-size: 10px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--text-mute);
            margin-bottom: 6px;
        }

        .field-wrap, .heatmap-wrap {
            position: relative;
            background: rgba(0,0,0,0.25);
            border: 1px solid var(--rule);
            border-radius: 6px;
            overflow: hidden;
            aspect-ratio: 1;
        }
        svg { width: 100%; height: 100%; display: block; }

        .tx-rect { cursor: grab; }
        .tx-rect.is-active { cursor: grabbing; }
        .tx-rect:focus-visible { outline: 2px solid var(--c-orange); outline-offset: 2px; }
        .drag-hint {
            position: absolute;
            bottom: 8px; left: 8px;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-mute);
            letter-spacing: 0.08em;
            pointer-events: none;
        }

        .controls {
            display: flex;
            gap: 12px;
            justify-content: space-between;
            align-items: center;
            margin-top: 12px;
            flex-wrap: wrap;
        }
        .ctrl {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-dim);
            letter-spacing: 0.04em;
        }
        button.reset {
            background: transparent;
            border: 1px solid var(--rule);
            color: var(--text-dim);
            font-family: var(--font-mono);
            font-size: 11px;
            padding: 5px 12px;
            border-radius: 6px;
            cursor: pointer;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        button.reset:hover { color: var(--text); border-color: var(--c-orange); }

        .caption {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-mute);
            letter-spacing: 0.04em;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px dashed var(--rule);
        }

        .colorbar {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 6px;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-mute);
            letter-spacing: 0.06em;
        }
        .colorbar .bar {
            height: 6px;
            flex: 1;
            border-radius: 3px;
            background: linear-gradient(90deg, #14161c, var(--c-orange));
        }
    </style>

    <div class="head">
        <span class="eyebrow">INTERACTIVE · drag Tx0 to see interference</span>
        <div class="readout">
            <span>γ = ${GAMMA}</span> · <span>d₀ = ${D0} m</span> · <span>field <span class="val">${FIELD}×${FIELD} m</span></span>
        </div>
    </div>

    <div class="stage">
        <div>
            <div class="panel-label">Field — 6 Tx / 6 Rx</div>
            <div class="field-wrap">
                <svg data-field viewBox="0 0 300 300" preserveAspectRatio="xMidYMid meet"></svg>
                <span class="drag-hint">drag the orange square · ← → ↑ ↓ nudge · shift = 10m</span>
            </div>
        </div>
        <div>
            <div class="panel-label">Channel matrix |h|² — log₁₀ scale</div>
            <div class="heatmap-wrap">
                <svg data-heatmap viewBox="0 0 300 300" preserveAspectRatio="xMidYMid meet"></svg>
            </div>
            <div class="colorbar">
                <span>min</span><span class="bar"></span><span>max</span>
            </div>
        </div>
    </div>

    <div class="controls">
        <span class="ctrl">ACTIVE · Tx<span class="val" data-active-idx>0</span> · <span data-active-pos></span></span>
        <button class="reset" type="button" data-reset>RESET</button>
    </div>

    <p class="caption">Path-loss physics only — no model inference. |h<sub>kj</sub>|² = (d₀ / max(d<sub>kj</sub>, d₀))^γ. Diagonal = direct link; off-diagonal = interference leaked to neighbours.</p>
`;

class InterferenceSandbox extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' }).appendChild(TEMPLATE.content.cloneNode(true));

        this._K = 6;
        this._active = 0;
        this._seeded = this._seedLayout(this._K);
        this._tx = this._seeded.tx.map((p) => ({ ...p }));
        this._rx = this._seeded.rx.map((p) => ({ ...p }));

        this.$field    = this.shadowRoot.querySelector('[data-field]');
        this.$heat     = this.shadowRoot.querySelector('[data-heatmap]');
        this.$activeIdx = this.shadowRoot.querySelector('[data-active-idx]');
        this.$activePos = this.shadowRoot.querySelector('[data-active-pos]');
        this.$reset    = this.shadowRoot.querySelector('[data-reset]');

        this.$reset.addEventListener('click', () => this._reset());
    }

    connectedCallback() {
        this._bindDrag();
        this._draw();
    }

    _seedLayout(K) {
        // Deterministic, visually pleasant placement around a ring.
        const tx = [], rx = [];
        for (let i = 0; i < K; i++) {
            const a = (i / K) * Math.PI * 2;
            const r = 60;
            const txx = FIELD / 2 + Math.cos(a) * r + (i % 2 ? -10 : 10);
            const txy = FIELD / 2 + Math.sin(a) * r + (i % 2 ? 8 : -6);
            const dx = Math.cos(a + 0.35) * 20;
            const dy = Math.sin(a + 0.35) * 20;
            tx.push({ x: txx, y: txy });
            rx.push({ x: txx + dx, y: txy + dy });
        }
        return { tx, rx };
    }

    _reset() {
        this._tx = this._seeded.tx.map((p) => ({ ...p }));
        this._rx = this._seeded.rx.map((p) => ({ ...p }));
        this._active = 0;
        this._draw();
    }

    _computeMatrix() {
        const K = this._K;
        const H = new Array(K);
        for (let k = 0; k < K; k++) {
            H[k] = new Array(K);
            for (let j = 0; j < K; j++) {
                const dx = this._tx[j].x - this._rx[k].x;
                const dy = this._tx[j].y - this._rx[k].y;
                const d  = Math.max(D0, Math.hypot(dx, dy));
                H[k][j] = Math.pow(D0 / d, GAMMA);
            }
        }
        return H;
    }

    _draw() {
        this._drawField();
        this._drawHeatmap();
        this.$activeIdx.textContent = this._active;
        const tx = this._tx[this._active];
        this.$activePos.textContent = `(${tx.x.toFixed(0)} m, ${tx.y.toFixed(0)} m)`;
    }

    _drawField() {
        const VB = 300;
        const scale = (v) => 12 + (v / FIELD) * (VB - 24);
        const svg = [];

        svg.push(`<rect x="8" y="8" width="${VB - 16}" height="${VB - 16}" rx="4" fill="none" stroke="rgba(255,255,255,0.05)" stroke-dasharray="2 4" />`);

        // Direct (Tx_k → Rx_k) links.
        for (let k = 0; k < this._K; k++) {
            const t = this._tx[k], r = this._rx[k];
            const isActive = k === this._active;
            svg.push(`<line x1="${scale(t.x)}" y1="${scale(t.y)}" x2="${scale(r.x)}" y2="${scale(r.y)}" stroke="${isActive ? 'var(--c-orange)' : 'var(--c-blue)'}" stroke-width="${isActive ? 1.6 : 1}" opacity="${isActive ? 0.9 : 0.55}" />`);
        }

        // Interference radii from active Tx — soft circles.
        const activeTx = this._tx[this._active];
        for (const d of [30, 60, 100]) {
            svg.push(`<circle cx="${scale(activeTx.x)}" cy="${scale(activeTx.y)}" r="${(d / FIELD) * (VB - 24)}" fill="none" stroke="var(--c-orange)" stroke-opacity="0.08" stroke-dasharray="2 6" />`);
        }

        // Rx rings.
        this._rx.forEach((r, i) => {
            const isActiveDirect = i === this._active;
            svg.push(`<circle cx="${scale(r.x)}" cy="${scale(r.y)}" r="5" fill="none" stroke="${isActiveDirect ? 'var(--c-orange)' : 'var(--c-blue)'}" stroke-width="1.5" opacity="0.8" />`);
            svg.push(`<text x="${scale(r.x) + 8}" y="${scale(r.y) + 3}" font-family="var(--font-mono)" font-size="9" fill="var(--text-mute)" letter-spacing="0.06em">R${i}</text>`);
        });

        // Tx squares.
        this._tx.forEach((t, i) => {
            const isActive = i === this._active;
            const color = isActive ? 'var(--c-orange)' : 'var(--c-blue)';
            const cls = isActive ? 'tx-rect is-active' : 'tx-rect';
            const glow = isActive ? `<rect x="${scale(t.x) - 10}" y="${scale(t.y) - 10}" width="20" height="20" fill="${color}" opacity="0.2" rx="3" />` : '';
            svg.push(glow);
            svg.push(`<rect class="${cls}" data-tx="${i}" x="${scale(t.x) - 5}" y="${scale(t.y) - 5}" width="10" height="10" fill="${color}" rx="1" tabindex="0" role="button" aria-label="Transmitter ${i} at (${t.x.toFixed(0)}, ${t.y.toFixed(0)})"><title>Tx${i} · drag to move</title></rect>`);
        });

        this.$field.innerHTML = svg.join('\n');
    }

    _drawHeatmap() {
        const H = this._computeMatrix();
        const K = this._K;

        // Log-scale clamp.
        let lo = Infinity, hi = -Infinity;
        for (let k = 0; k < K; k++) for (let j = 0; j < K; j++) {
            const v = H[k][j];
            if (v > 0) {
                const lv = Math.log10(v);
                if (lv < lo) lo = lv;
                if (lv > hi) hi = lv;
            }
        }
        if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi === lo) { lo = -3; hi = 0; }

        const VB = 300;
        const pad = 32;
        const cellW = (VB - pad - 12) / K;
        const cellH = (VB - pad - 12) / K;
        const svg = [];

        // Axis labels.
        for (let i = 0; i < K; i++) {
            svg.push(`<text x="${pad + cellW * (i + 0.5)}" y="${VB - 18}" font-family="var(--font-mono)" font-size="9" fill="var(--text-mute)" text-anchor="middle">Tx${i}</text>`);
            svg.push(`<text x="${pad - 6}" y="${pad + cellH * (i + 0.5) + 3}" font-family="var(--font-mono)" font-size="9" fill="var(--text-mute)" text-anchor="end">Rx${i}</text>`);
        }
        svg.push(`<text x="${VB / 2}" y="${VB - 4}" font-family="var(--font-mono)" font-size="10" fill="var(--text-dim)" text-anchor="middle" letter-spacing="1">TRANSMITTER →</text>`);
        svg.push(`<text x="${14}" y="${VB / 2}" font-family="var(--font-mono)" font-size="10" fill="var(--text-dim)" text-anchor="middle" letter-spacing="1" transform="rotate(-90 14 ${VB / 2})">RECEIVER →</text>`);

        // Heatmap cells.
        const color = (v) => {
            if (v <= 0) return '#0a0b0e';
            const t = (Math.log10(v) - lo) / (hi - lo);
            const clamped = Math.max(0, Math.min(1, t));
            // Map 0..1 → dark surface → orange.
            const r = Math.round(20 + (255 - 20) * clamped);
            const g = Math.round(22 + (106 - 22) * clamped);
            const b = Math.round(28 + (61  - 28) * clamped);
            return `rgb(${r},${g},${b})`;
        };

        for (let rx = 0; rx < K; rx++) {
            for (let tx = 0; tx < K; tx++) {
                const v = H[rx][tx];
                const x = pad + tx * cellW;
                const y = pad + rx * cellH;
                const fill = color(v);
                const isDiag = rx === tx;
                const stroke = isDiag ? 'var(--c-orange)' : 'rgba(255,255,255,0.04)';
                svg.push(`<rect x="${x}" y="${y}" width="${cellW - 1}" height="${cellH - 1}" fill="${fill}" stroke="${stroke}" stroke-width="${isDiag ? 1 : 0.5}"><title>|h_${rx},${tx}|² = ${v.toExponential(2)}</title></rect>`);
            }
        }

        this.$heat.innerHTML = svg.join('\n');
    }

    _bindDrag() {
        const svg = this.$field;

        const fieldCoord = (evt) => {
            const rect = svg.getBoundingClientRect();
            const VB = 300;
            const px = (evt.clientX - rect.left) / rect.width * VB;
            const py = (evt.clientY - rect.top)  / rect.height * VB;
            const x = (px - 12) / (VB - 24) * FIELD;
            const y = (py - 12) / (VB - 24) * FIELD;
            return {
                x: Math.max(0, Math.min(FIELD, x)),
                y: Math.max(0, Math.min(FIELD, y)),
            };
        };

        let draggingIdx = null;

        const onDown = (e) => {
            const target = e.target.closest('rect[data-tx]');
            if (!target) return;
            draggingIdx = Number(target.getAttribute('data-tx'));
            this._active = draggingIdx;
            target.setPointerCapture?.(e.pointerId);
            e.preventDefault();
            this._draw();
        };
        const onMove = (e) => {
            if (draggingIdx == null) return;
            const pos = fieldCoord(e);
            // Move the matching Tx/Rx pair so the direct link stays meaningful.
            const dx = pos.x - this._tx[draggingIdx].x;
            const dy = pos.y - this._tx[draggingIdx].y;
            this._tx[draggingIdx] = pos;
            this._rx[draggingIdx] = {
                x: Math.max(0, Math.min(FIELD, this._rx[draggingIdx].x + dx)),
                y: Math.max(0, Math.min(FIELD, this._rx[draggingIdx].y + dy)),
            };
            this._draw();
        };
        const onUp = () => { draggingIdx = null; };

        svg.addEventListener('pointerdown', onDown);
        svg.addEventListener('pointermove', onMove);
        svg.addEventListener('pointerup',   onUp);
        svg.addEventListener('pointercancel', onUp);

        // Keyboard nudge when a tx-rect is focused.
        svg.addEventListener('keydown', (e) => {
            const t = e.target.closest('rect[data-tx]');
            if (!t) return;
            const i = Number(t.getAttribute('data-tx'));
            this._active = i;
            const step = e.shiftKey ? 10 : 1;
            let moved = false;
            if (e.key === 'ArrowLeft')  { this._tx[i].x = Math.max(0, this._tx[i].x - step); this._rx[i].x = Math.max(0, this._rx[i].x - step); moved = true; }
            if (e.key === 'ArrowRight') { this._tx[i].x = Math.min(FIELD, this._tx[i].x + step); this._rx[i].x = Math.min(FIELD, this._rx[i].x + step); moved = true; }
            if (e.key === 'ArrowUp')    { this._tx[i].y = Math.max(0, this._tx[i].y - step); this._rx[i].y = Math.max(0, this._rx[i].y - step); moved = true; }
            if (e.key === 'ArrowDown')  { this._tx[i].y = Math.min(FIELD, this._tx[i].y + step); this._rx[i].y = Math.min(FIELD, this._rx[i].y + step); moved = true; }
            if (moved) { e.preventDefault(); this._draw(); }
        });

        // Click-to-select on tx squares (fallback when not dragging).
        svg.addEventListener('click', (e) => {
            const t = e.target.closest('rect[data-tx]');
            if (!t) return;
            this._active = Number(t.getAttribute('data-tx'));
            this._draw();
        });
    }
}

customElements.define('interference-sandbox', InterferenceSandbox);
