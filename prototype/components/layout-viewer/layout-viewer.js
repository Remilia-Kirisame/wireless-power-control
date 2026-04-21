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
        .edge-interf { stroke: rgba(255,255,255,0.08); stroke-dasharray: 1 4; }
        .edge-intra  { stroke: rgba(255,255,255,0.05); }
        .rx          { transition: transform var(--dur-mid, 260ms) var(--ease, ease), opacity var(--dur-mid, 260ms) var(--ease, ease); cursor: pointer; }
        .rx:focus-visible { outline: 2px solid var(--c-orange); outline-offset: 2px; }
        .tx          { transition: opacity var(--dur-mid, 260ms) var(--ease, ease); cursor: pointer; }
        .is-dim      { opacity: 0.18; }

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
        .swatch-sq { width: 8px; height: 8px; border-radius: 1px; }
        .swatch-ring { width: 10px; height: 10px; border-radius: 50%; border: 1.5px solid currentColor; }

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
        <span class="eyebrow" data-hint>INTERACTIVE · click a Blue-car to isolate</span>
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
        <span class="item" style="color: var(--c-yellow);"><span class="swatch-ring"></span>Yellow (sensing)</span>
        <span class="item" style="color: var(--c-green);"><span class="swatch-ring"></span>Green (comm)</span>
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

        // Normalize power → brightness in [0, 1].
        const pMax = Math.max(0.0001, ...power);
        const brightness = (i) => {
            const p = power[i] ?? 0;
            return Math.min(1, p / pMax);
        };

        const MARGIN = 20;
        const VB = 300;
        const scale = (v) => MARGIN + (v / field) * (VB - 2 * MARGIN);

        const svg = [];

        // Frame.
        svg.push(`<rect x="${MARGIN - 8}" y="${MARGIN - 8}" width="${VB - 2 * MARGIN + 16}" height="${VB - 2 * MARGIN + 16}" rx="4" fill="none" stroke="rgba(255,255,255,0.04)" stroke-dasharray="2 4" />`);

        // Edges: intra-cluster (faint) + same-channel inter-cluster (dashed).
        // Intra: every rx ↔ its blue-car.
        rx.forEach((r) => {
            const b = blue[r.blue];
            if (!b) return;
            const inactive = this._isolate != null && this._isolate !== r.blue;
            const op = inactive ? 0.08 : 0.35;
            const color = r.type === 'yellow' ? 'var(--c-yellow)' : 'var(--c-green)';
            svg.push(`<line class="edge-intra" x1="${scale(b.x)}" y1="${scale(b.y)}" x2="${scale(r.x)}" y2="${scale(r.y)}" stroke="${color}" opacity="${op}" stroke-width="1" />`);
        });

        // Interference edges — same channel across different blue cars, dashed grey.
        const byChannel = new Map();
        rx.forEach((r) => {
            const key = r.channel;
            if (!byChannel.has(key)) byChannel.set(key, []);
            byChannel.get(key).push(r);
        });
        byChannel.forEach((list) => {
            for (let i = 0; i < list.length; i++) {
                for (let j = i + 1; j < list.length; j++) {
                    const a = list[i], b = list[j];
                    if (a.blue === b.blue) continue;
                    const blueA = blue[a.blue], blueB = blue[b.blue];
                    if (!blueA || !blueB) continue;
                    const inactive = this._isolate != null && this._isolate !== a.blue && this._isolate !== b.blue;
                    const op = inactive ? 0.04 : 0.25;
                    // Draw Tx_A → Rx_B (the cross-cluster leak).
                    svg.push(`<line class="edge-interf" x1="${scale(blueA.x)}" y1="${scale(blueA.y)}" x2="${scale(b.x)}" y2="${scale(b.y)}" stroke="var(--c-grey)" opacity="${op}" />`);
                    svg.push(`<line class="edge-interf" x1="${scale(blueB.x)}" y1="${scale(blueB.y)}" x2="${scale(a.x)}" y2="${scale(a.y)}" stroke="var(--c-grey)" opacity="${op}" />`);
                }
            }
        });

        // Rx markers (open rings; size + brightness ∝ power).
        rx.forEach((r, i) => {
            const cx = scale(r.x), cy = scale(r.y);
            const b = brightness(i);
            const radius = 2.5 + 4.5 * b;
            const color = r.type === 'yellow' ? 'var(--c-yellow)' : 'var(--c-green)';
            const op = 0.35 + 0.65 * b;
            const glowOp = 0.12 * b;
            const inactive = this._isolate != null && this._isolate !== r.blue;
            const classAttr = inactive ? 'rx is-dim' : 'rx';
            if (b > 0.1 && !inactive) {
                svg.push(`<circle cx="${cx}" cy="${cy}" r="${radius * 2.5}" fill="${color}" opacity="${glowOp}" />`);
            }
            svg.push(`<circle class="${classAttr}" cx="${cx}" cy="${cy}" r="${radius.toFixed(1)}" fill="none" stroke="${color}" stroke-width="1.5" opacity="${op.toFixed(2)}" tabindex="0" role="button" aria-label="Rx ${r.id} (${r.type}, blue ${r.blue}), power ${(power[i]||0).toFixed(3)}"><title>${r.type} Rx · blue ${r.blue} · ch ${r.channel} · p=${(power[i]||0).toFixed(3)}</title></circle>`);
        });

        // Tx squares.
        blue.forEach((b) => {
            const cx = scale(b.x), cy = scale(b.y);
            const inactive = this._isolate != null && this._isolate !== b.id;
            const classAttr = inactive ? 'tx is-dim' : 'tx';
            svg.push(`<g class="${classAttr}"><rect x="${cx - 4}" y="${cy - 4}" width="8" height="8" fill="var(--c-blue)" data-blue="${b.id}" tabindex="0" role="button" aria-label="Blue-car ${b.id}"><title>Blue-car ${b.id} — click to isolate</title></rect></g>`);
        });

        this.$field.setAttribute('viewBox', `0 0 ${VB} ${VB}`);
        this.$field.innerHTML = svg.join('\n');

        // Wire up click/focus handlers for Tx squares.
        this.$field.querySelectorAll('rect[data-blue]').forEach((rect) => {
            const id = Number(rect.getAttribute('data-blue'));
            const handler = (e) => {
                e.stopPropagation();
                this._isolate = (this._isolate === id) ? null : id;
                this._render();
            };
            rect.addEventListener('click', handler);
            rect.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handler(e); }
            });
        });
        this.$field.addEventListener('click', (e) => {
            if (e.target === this.$field) { this._isolate = null; this._render(); }
        });

        // Readout panel.
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
        if (this._isolate != null) {
            chips.push(`<span><span class="key">ISOLATED</span><span class="val">blue ${this._isolate}</span></span>`);
        }
        this.$readout.innerHTML = chips.join('');
    }
}

customElements.define('layout-viewer', LayoutViewer);
