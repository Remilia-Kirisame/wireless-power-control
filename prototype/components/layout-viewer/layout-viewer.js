/* <layout-viewer> — single-layout SVG map.
 *
 * Renders one JSAC (or D2D) layout. Toggle between methods; Rx marker size + brightness
 * encode allocated power; edges colored by link type; click a Blue-car to isolate that cluster.
 *
 * Attributes:
 *   data-src       Path to layout JSON (WEB_PROMPT §7.D shape).
 *   data-initial   Initial method ("Naive" | "WMMSE" | "GNN"). Default "GNN".
 *   data-compact   "true" for smaller padding / font sizing inside galleries.
 */

const METHOD_COLORS = {
    Naive: 'var(--c-grey)',
    WMMSE: 'var(--c-blue)',
    GNN:   'var(--c-orange)',
};

// Rendering tunables — adjust to taste.
// Rx markers: solid filled discs at constant RX_RADIUS; power → dot opacity + glow opacity (no size scaling).
const RX_RADIUS            = 3.6;   // Filled disc radius
const RX_HIT_RADIUS        = 7;     // Invisible click target around the disc
const RX_GLOW_RADIUS_MULT  = 2.5;   // Glow halo size = RX_RADIUS × this
const RX_DOT_OPACITY_BASE  = 0.30;  // Disc opacity floor (low-power Rx)
const RX_DOT_OPACITY_RANGE = 0.70;  // Added at peak power (max = base+range = 1.0)
const RX_GLOW_OPACITY      = 0.28;  // Glow opacity at peak power
const RX_GLOW_THRESHOLD    = 0.10;  // Skip glow below this normalized power
// Edge opacities by isolation state (default = no isolation; active/inactive = isolation on).
const INTRA_OP_ACTIVE      = 0.35;  // Faint coloured intra-cluster lines
const INTRA_OP_INACTIVE    = 0.08;
const INTERF_OP_DEFAULT    = 0.25;  // Dashed grey lines, no isolation
const INTERF_OP_ACTIVE     = 0.55;  // Bumped from 0.25 -- relevant dashed lines when isolated
const INTERF_OP_INACTIVE   = 0.04;  // Irrelevant dashed lines when isolated
const INTERF_HIT_WIDTH     = 7;     // Invisible click target on interference lines

const TEMPLATE = document.createElement('template');
TEMPLATE.innerHTML = /* html */ `
    <style>
        :host {
            display: block;
            background: var(--surface);
            border: 1px solid var(--rule);
            border-radius: var(--radius-card);
            padding: 18px 20px;
            position: relative;
            color: var(--text);
            font-family: var(--font-sans);
        }
        :host([data-compact="true"]) { padding: 12px 14px; }

        .head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 14px;
            margin-bottom: 10px;
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
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-dim);
            letter-spacing: 0.04em;
        }
        .readout .key { color: var(--text-mute); }
        .readout .val { color: var(--text); font-variant-numeric: tabular-nums; margin-left: 6px; }

        .controls {
            display: flex;
            gap: 12px;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }

        .field-wrap {
            position: relative;
            width: 100%;
            aspect-ratio: 1;
            background: rgba(0,0,0,0.25);
            border: 1px solid var(--rule);
            border-radius: 6px;
            overflow: hidden;
        }
        svg.field {
            width: 100%;
            height: 100%;
            display: block;
        }
        .edge-interf     { stroke: rgba(255,255,255,0.08); stroke-dasharray: 1 4; pointer-events: none; }
        .edge-intra      { stroke: rgba(255,255,255,0.05); pointer-events: none; }
        .edge-interf-hit { cursor: pointer; }
        .rx              { transition: opacity var(--dur-mid, 260ms) var(--ease, ease); cursor: pointer; outline: none; }
        .rx:focus-visible { outline: 2px solid var(--c-orange); outline-offset: 2px; border-radius: 50%; }
        .tx              { transition: opacity var(--dur-mid, 260ms) var(--ease, ease); cursor: pointer; }
        .is-dim          { opacity: 0.18; }
        .is-half         { opacity: 0.5; }

        .legend {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-top: 10px;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-mute);
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .legend .item { display: inline-flex; align-items: center; gap: 6px; }
        .swatch-sq   { width: 8px; height: 8px; border-radius: 1px; }
        .swatch-dot  { width: 9px; height: 9px; border-radius: 50%; background: currentColor; }

        @media (prefers-reduced-motion: reduce) {
            .rx, .tx { transition: none; }
        }

        .empty {
            color: var(--text-mute);
            font-family: var(--font-mono);
            font-size: 12px;
            padding: 40px 10px;
            text-align: center;
            letter-spacing: 0.04em;
        }
    </style>

    <div class="head">
        <span class="eyebrow" data-hint>INTERACTIVE · click Tx · Rx · or interference line</span>
        <div class="readout" data-readout></div>
    </div>

    <div class="controls">
        <method-toggle data-methods="Naive,WMMSE,GNN" data-default="GNN" data-multi="false" data-colors="var(--c-grey),var(--c-blue),var(--c-orange)"></method-toggle>
        <span style="font-family: var(--font-mono); font-size: 10px; color: var(--text-mute); letter-spacing: 0.06em;">BRIGHTNESS · ALLOCATED POWER</span>
    </div>

    <div class="field-wrap">
        <svg class="field" data-field viewBox="0 0 300 300" preserveAspectRatio="xMidYMid meet"></svg>
    </div>

    <div class="legend">
        <span class="item"><span class="swatch-sq" style="background: var(--c-blue);"></span>Blue Tx</span>
        <span class="item" style="color: var(--c-yellow);"><span class="swatch-dot"></span>Yellow (sensing)</span>
        <span class="item" style="color: var(--c-green);"><span class="swatch-dot"></span>Green (comm)</span>
        <span class="item">─ ─ Interference (inter-cluster)</span>
    </div>

    <div class="empty" data-empty hidden>offline — open via <code>python -m http.server</code></div>
`;

class LayoutViewer extends HTMLElement {
    static get observedAttributes() { return ['data-src']; }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' }).appendChild(TEMPLATE.content.cloneNode(true));
        this._layout = null;
        this._method = this.getAttribute('data-initial') || 'GNN';
        this._isolate = null;

        this.$field   = this.shadowRoot.querySelector('[data-field]');
        this.$readout = this.shadowRoot.querySelector('[data-readout]');
        this.$empty   = this.shadowRoot.querySelector('[data-empty]');
        this.$toggle  = this.shadowRoot.querySelector('method-toggle');
    }

    connectedCallback() {
        this.$toggle.setAttribute('data-default', this._method);
        this.$toggle.addEventListener('method-change', (e) => {
            const last = e.detail.last || this._method;
            this._method = last;
            this._render();
        });
        this.$field.addEventListener('click', this._onFieldClick);
        this.$field.addEventListener('keydown', this._onFieldKey);
        this._load();
    }

    attributeChangedCallback(name) {
        if (!this.isConnected) return;
        if (name === 'data-src') this._load();
    }

    set data(layout) {
        this._layout = layout;
        this._render();
    }

    async _load() {
        const src = this.getAttribute('data-src');
        if (!src) return;
        try {
            const fetcher = window.fetchJSONCached || ((u) => fetch(u).then((r) => r.json()));
            const data = await fetcher(src);
            if (data?._stub) {
                this._showEmpty(true);
                return;
            }
            this._layout = data;
            this._isolate = null;
            this._showEmpty(false);
            this._render();
        } catch (err) {
            console.warn(`<layout-viewer> failed to load ${src}:`, err);
            this._showEmpty(true);
        }
    }

    _showEmpty(show) {
        this.$empty.hidden = !show;
    }

    _render() {
        if (!this._layout) return;
        const L = this._layout;
        const field = L.config?.field || 225;
        const blue  = L.blue || [];
        const rx    = L.rx   || [];
        const power = (L.power || {})[this._method] || [];
        const metrics = L.metrics || {};

        const pMax = Math.max(0.0001, ...power);
        const brightness = (i) => Math.min(1, (power[i] ?? 0) / pMax);

        const MARGIN = 20;
        const VB = 300;
        const scale = (v) => MARGIN + (v / field) * (VB - 2 * MARGIN);

        const view = this._isolationView();
        const isolated = !!view;

        const svg = [];

        svg.push(`<rect x="${MARGIN - 8}" y="${MARGIN - 8}" width="${VB - 2 * MARGIN + 16}" height="${VB - 2 * MARGIN + 16}" rx="4" fill="none" stroke="rgba(255,255,255,0.04)" stroke-dasharray="2 4" />`);

        rx.forEach((r) => {
            const bc = blue[r.blue];
            if (!bc) return;
            const active = !view || view.intraEdgeActive(r);
            const op = isolated ? (active ? INTRA_OP_ACTIVE : INTRA_OP_INACTIVE) : INTRA_OP_ACTIVE;
            const color = r.type === 'yellow' ? 'var(--c-yellow)' : 'var(--c-green)';
            svg.push(`<line class="edge-intra" x1="${scale(bc.x)}" y1="${scale(bc.y)}" x2="${scale(r.x)}" y2="${scale(r.y)}" stroke="${color}" opacity="${op}" stroke-width="1" />`);
        });

        const byChannel = new Map();
        rx.forEach((r) => {
            if (!byChannel.has(r.channel)) byChannel.set(r.channel, []);
            byChannel.get(r.channel).push(r);
        });
        const renderInterf = (txBlue, rxObj) => {
            const txPos = blue[txBlue];
            if (!txPos) return;
            const active = !view || view.interfEdgeActive(txBlue, rxObj);
            const op = isolated
                ? (active ? INTERF_OP_ACTIVE : INTERF_OP_INACTIVE)
                : INTERF_OP_DEFAULT;
            const x1 = scale(txPos.x), y1 = scale(txPos.y);
            const x2 = scale(rxObj.x), y2 = scale(rxObj.y);
            svg.push(`<line class="edge-interf" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--c-grey)" opacity="${op.toFixed(3)}" />`);
            svg.push(`<line class="edge-interf-hit" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="transparent" stroke-width="${INTERF_HIT_WIDTH}" data-tx-blue="${txBlue}" data-rx-id="${rxObj.id}" data-channel="${rxObj.channel}" />`);
        };
        byChannel.forEach((list) => {
            for (let i = 0; i < list.length; i++) {
                for (let j = i + 1; j < list.length; j++) {
                    const r1 = list[i], r2 = list[j];
                    if (r1.blue === r2.blue) continue;
                    renderInterf(r1.blue, r2);
                    renderInterf(r2.blue, r1);
                }
            }
        });

        rx.forEach((r, i) => {
            const cx = scale(r.x), cy = scale(r.y);
            const b = brightness(i);
            const color = r.type === 'yellow' ? 'var(--c-yellow)' : 'var(--c-green)';
            const dotOp = RX_DOT_OPACITY_BASE + RX_DOT_OPACITY_RANGE * b;
            const glowOp = RX_GLOW_OPACITY * b;
            const state = view ? view.rxState(r) : 'full';
            const cls = state === 'full' ? 'rx' : state === 'half' ? 'rx is-half' : 'rx is-dim';
            const parts = [];
            if (b > RX_GLOW_THRESHOLD && state !== 'dim') {
                parts.push(`<circle cx="${cx}" cy="${cy}" r="${(RX_RADIUS * RX_GLOW_RADIUS_MULT).toFixed(2)}" fill="${color}" opacity="${glowOp.toFixed(3)}" pointer-events="none" />`);
            }
            parts.push(`<circle cx="${cx}" cy="${cy}" r="${RX_RADIUS}" fill="${color}" opacity="${dotOp.toFixed(3)}" pointer-events="none" />`);
            parts.push(`<circle cx="${cx}" cy="${cy}" r="${RX_HIT_RADIUS}" fill="transparent" />`);
            const aria = `Rx ${r.id} (${r.type}, blue ${r.blue}, ch ${r.channel}), p=${(power[i]||0).toFixed(3)} — click to isolate channel`;
            svg.push(`<g class="${cls}" tabindex="0" role="button" data-rx="${r.id}" data-channel="${r.channel}" aria-label="${aria}"><title>${r.type} Rx · blue ${r.blue} · ch ${r.channel} · p=${(power[i]||0).toFixed(3)}</title>${parts.join('')}</g>`);
        });

        blue.forEach((bc) => {
            const cx = scale(bc.x), cy = scale(bc.y);
            const state = view ? view.txState(bc.id) : 'full';
            const cls = state === 'full' ? 'tx' : state === 'half' ? 'tx is-half' : 'tx is-dim';
            svg.push(`<g class="${cls}"><rect x="${cx - 4}" y="${cy - 4}" width="8" height="8" fill="var(--c-blue)" data-blue="${bc.id}" tabindex="0" role="button" aria-label="Blue-car ${bc.id} — click to isolate cluster"><title>Blue-car ${bc.id} — click to isolate cluster</title></rect></g>`);
        });

        this.$field.setAttribute('viewBox', `0 0 ${VB} ${VB}`);
        this.$field.innerHTML = svg.join('\n');

        const cfg = L.config || {};
        const chips = [];
        chips.push(`<span><span class="key">CONFIG</span><span class="val">B=${cfg.B} · M_y=${cfg.M_y} · M_g=${cfg.M_g} · K=${cfg.K}</span></span>`);
        chips.push(`<span><span class="key">METHOD</span><span class="val" style="color:${METHOD_COLORS[this._method]||'var(--text)'}">${this._method}</span></span>`);
        if (metrics?.green_sumrate?.[this._method] != null) {
            chips.push(`<span><span class="key">GREEN SUM-RATE</span><span class="val">${metrics.green_sumrate[this._method].toFixed(2)}</span></span>`);
        }
        if (metrics?.yellow_viol_rate?.[this._method] != null) {
            chips.push(`<span><span class="key">YELLOW VIOL</span><span class="val">${metrics.yellow_viol_rate[this._method].toFixed(1)}%</span></span>`);
        }
        if (this._isolate) {
            const iso = this._isolate;
            let label = '';
            if (iso.type === 'cluster') label = `cluster ${iso.blue}`;
            else if (iso.type === 'channel') label = `channel ${iso.channel} · Rx ${iso.rx_id}`;
            else if (iso.type === 'edge')    label = `Tx ${iso.tx_blue} → Rx ${iso.rx_id} (ch ${iso.channel})`;
            chips.push(`<span><span class="key">ISOLATED</span><span class="val">${label}</span></span>`);
        }
        this.$readout.innerHTML = chips.join('');
    }

    _isolationView() {
        const iso = this._isolate;
        if (!iso) return null;
        const rx = this._layout?.rx || [];
        const rxById = new Map(rx.map((r) => [r.id, r]));
        return {
            txState: (blueId) => {
                if (iso.type === 'cluster') return blueId === iso.blue ? 'full' : 'dim';
                if (iso.type === 'channel') {
                    const target = rxById.get(iso.rx_id);
                    if (target && blueId === target.blue) return 'full';
                    if (rx.some((r) => r.blue === blueId && r.channel === iso.channel)) return 'half';
                    return 'dim';
                }
                if (iso.type === 'edge') {
                    const target = rxById.get(iso.rx_id);
                    if (blueId === iso.tx_blue) return 'full';
                    if (target && blueId === target.blue) return 'full';
                    return 'dim';
                }
                return 'dim';
            },
            rxState: (r) => {
                if (iso.type === 'cluster') return r.blue === iso.blue ? 'full' : 'dim';
                if (iso.type === 'channel') {
                    if (r.id === iso.rx_id)        return 'full';
                    if (r.channel === iso.channel) return 'half';
                    return 'dim';
                }
                if (iso.type === 'edge')    return r.id === iso.rx_id ? 'full' : 'dim';
                return 'dim';
            },
            intraEdgeActive: (r) => {
                if (iso.type === 'cluster') return r.blue === iso.blue;
                if (iso.type === 'channel') return r.id === iso.rx_id;
                if (iso.type === 'edge')    return r.id === iso.rx_id;
                return false;
            },
            interfEdgeActive: (txBlue, rxObj) => {
                if (iso.type === 'cluster') return txBlue === iso.blue || rxObj.blue === iso.blue;
                if (iso.type === 'channel') return rxObj.id === iso.rx_id;
                if (iso.type === 'edge')    return txBlue === iso.tx_blue && rxObj.id === iso.rx_id;
                return false;
            },
        };
    }

    _onFieldClick = (e) => {
        const rect = e.target.closest('rect[data-blue]');
        if (rect) {
            e.stopPropagation();
            const id = Number(rect.getAttribute('data-blue'));
            const same = this._isolate?.type === 'cluster' && this._isolate.blue === id;
            this._isolate = same ? null : { type: 'cluster', blue: id };
            this._render();
            return;
        }
        const rxg = e.target.closest('g[data-rx]');
        if (rxg) {
            e.stopPropagation();
            const channel = Number(rxg.getAttribute('data-channel'));
            const rxId    = Number(rxg.getAttribute('data-rx'));
            const same = this._isolate?.type === 'channel' && this._isolate.rx_id === rxId;
            this._isolate = same ? null : { type: 'channel', channel, rx_id: rxId };
            this._render();
            return;
        }
        const line = e.target.closest('line.edge-interf-hit');
        if (line) {
            e.stopPropagation();
            const txBlue  = Number(line.getAttribute('data-tx-blue'));
            const rxId    = Number(line.getAttribute('data-rx-id'));
            const channel = Number(line.getAttribute('data-channel'));
            const iso = this._isolate;
            const same = iso?.type === 'edge' && iso.tx_blue === txBlue && iso.rx_id === rxId;
            this._isolate = same ? null : { type: 'edge', tx_blue: txBlue, rx_id: rxId, channel };
            this._render();
            return;
        }
        if (this._isolate) {
            this._isolate = null;
            this._render();
        }
    };

    _onFieldKey = (e) => {
        if (e.key !== 'Enter' && e.key !== ' ') return;
        if (!e.target.closest('rect[data-blue]') && !e.target.closest('g[data-rx]')) return;
        e.preventDefault();
        this._onFieldClick(e);
    };
}

customElements.define('layout-viewer', LayoutViewer);
