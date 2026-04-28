/* <sweep-slider> — slider + multi-line chart + numeric readout panel.
 *
 * Powers both the D2D K-scaling finding (C') and each of the JSAC sweep panels (C).
 *
 * Attributes:
 *   data-src       Path to the sweep JSON (PROMPT_WEB §8.1 shape).
 *   data-x-key     Field inside each point used as x-value ("K", "B", "M").
 *   data-x-label   Axis label for the slider.
 *   data-metric    Metric key to plot on the y-axis.
 *   data-metrics   (optional JSON array) list of switchable metrics.
 *                  [{key, label, unit, fmt}]
 *   data-series    JSON array of series definitions: [{key, color}].
 *   data-hint      Mono hint text ("drag K to explore" etc.).
 *   data-compact   "true" for the 2×2 JSAC grid (smaller chart).
 */

const TEMPLATE = document.createElement('template');
TEMPLATE.innerHTML = /* html */ `
    <style>
        :host {
            display: block;
            background: var(--surface);
            border: 1px solid var(--rule);
            border-radius: var(--radius-card);
            padding: 20px 22px 18px;
            position: relative;
            color: var(--text);
            font-family: var(--font-sans);
            /* Mirror the JS plot padding (see _draw) so the slider and tick row
             * align with the chart's data area. Update both sides if either changes. */
            --pad-l: 44px;
            --pad-r: 14px;
        }
        :host([data-compact="true"]) {
            padding: 16px 18px;
            --pad-l: 38px;
        }

        .head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 18px;
            margin-bottom: 8px;
            flex-wrap: wrap;
        }
        .hint-group {
            display: inline-flex;
            align-items: center;
            gap: 12px;
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
        .readout {
            display: flex;
            gap: 18px;
            flex-wrap: wrap;
            align-items: baseline;
        }
        .readout .metric {
            display: flex;
            align-items: baseline;
            gap: 8px;
            font-family: var(--font-mono);
            font-size: 12px;
            color: var(--text-dim);
            letter-spacing: 0.04em;
        }
        .readout .metric .val {
            font-size: 15px;
            color: var(--text);
            font-variant-numeric: tabular-nums;
            font-feature-settings: "tnum" 1;
            font-weight: 500;
        }
        .readout .dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 2px;
            translate: 0 -1px;
        }

        .metric-tabs {
            display: flex;
            gap: 2px;
            margin: 10px 0 12px;
            padding: 3px;
            border: 1px solid var(--rule);
            border-radius: 8px;
            background: rgba(0,0,0,0.25);
            width: fit-content;
        }
        .metric-tabs button {
            background: transparent;
            border: none;
            color: var(--text-dim);
            font-family: var(--font-mono);
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 6px 12px;
            border-radius: 5px;
            cursor: pointer;
            transition: color var(--dur-fast) var(--ease), background var(--dur-fast) var(--ease);
        }
        .metric-tabs button:hover { color: var(--text); }
        .metric-tabs button[aria-pressed="true"] {
            color: var(--text);
            background: rgba(255, 106, 61, 0.12);
        }

        .chart-wrap {
            position: relative;
            margin: 4px 0 12px;
        }
        svg {
            width: 100%;
            height: auto;
            overflow: visible;
            display: block;
        }
        .grid-line { stroke: rgba(255,255,255,0.05); stroke-width: 1; }
        .axis-tick {
            font-family: var(--font-mono);
            font-size: 10px;
            fill: var(--text-mute);
            letter-spacing: 0.05em;
        }
        .axis-label {
            font-family: var(--font-mono);
            font-size: 10px;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            fill: var(--text-mute);
        }
        .cursor-line {
            stroke: var(--c-orange);
            stroke-width: 1;
            stroke-dasharray: 3 4;
            opacity: 0.55;
        }
        .cursor-cap {
            fill: var(--c-orange);
        }
        .series-line {
            fill: none;
            stroke-width: 2;
            stroke-linecap: round;
            stroke-linejoin: round;
            transition: opacity var(--dur-fast) var(--ease);
        }
        .series-line.is-dim { opacity: 0.2; }
        .series-dot {
            transition: r var(--dur-fast) var(--ease), opacity var(--dur-fast) var(--ease);
        }
        .series-dot.is-active {
            r: 4.5;
        }
        .series-halo {
            fill: none;
            stroke-width: 7;
            opacity: 0.22;
            filter: blur(2.5px);
        }
        @media (prefers-reduced-motion: reduce) {
            .series-line, .series-dot { transition: none; }
        }

        .slider-row {
            margin-top: 4px;
            /* Align the native range track with the chart's data area. The −7px
             * compensates for the 14px-wide thumb: in Chrome/Safari the thumb
             * center is constrained to (thumb_radius .. width − thumb_radius)
             * within the input, so extending the input outward by the radius
             * lets the thumb center reach pad-l..(W−pad-r). */
            padding-left: calc(var(--pad-l) - 7px);
            padding-right: calc(var(--pad-r) - 7px);
        }
        .ticks {
            position: relative;
            height: 14px;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-mute);
            margin-top: 4px;
            letter-spacing: 0.04em;
        }
        .ticks > span {
            position: absolute;
            top: 0;
            transform: translateX(-50%);
            white-space: nowrap;
        }
        .kbdhint {
            display: inline-flex;
            gap: 4px;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-mute);
            letter-spacing: 0.06em;
        }
        .kbdhint kbd {
            font-family: var(--font-mono);
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid var(--rule);
            background: var(--surface-2, #1a1d26);
            color: var(--text-dim);
        }

        input[type="range"] {
            -webkit-appearance: none;
            appearance: none;
            width: 100%;
            height: 3px;
            margin: 0; /* override UA default 2px so slider-row padding measures cleanly */
            background: rgba(255,255,255,0.08);
            border-radius: 2px;
            cursor: pointer;
        }
        input[type="range"]:focus { outline: none; }
        input[type="range"]:focus-visible {
            box-shadow: 0 0 0 3px rgba(255, 106, 61, 0.25);
            border-radius: 4px;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 14px;
            height: 14px;
            background: var(--c-orange);
            border-radius: 50%;
            border: 2px solid var(--bg);
            box-shadow: 0 0 0 1px var(--c-orange), 0 0 12px rgba(255, 106, 61, 0.5);
            cursor: grab;
        }
        input[type="range"]::-moz-range-thumb {
            width: 14px;
            height: 14px;
            background: var(--c-orange);
            border-radius: 50%;
            border: 2px solid var(--bg);
            box-shadow: 0 0 0 1px var(--c-orange), 0 0 12px rgba(255, 106, 61, 0.5);
            cursor: grab;
        }

        .legend {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-dim);
            letter-spacing: 0.04em;
            margin-top: 12px;
        }
        .legend .item {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            padding: 3px 6px;
            border-radius: 4px;
            transition: background var(--dur-fast) var(--ease);
            user-select: none;
        }
        .legend .item:hover { background: rgba(255,255,255,0.03); }
        .legend .item[aria-pressed="false"] { opacity: 0.4; }
        .legend .item .swatch {
            width: 10px;
            height: 10px;
            border-radius: 2px;
        }

        .empty {
            color: var(--text-mute);
            font-family: var(--font-mono);
            font-size: 12px;
            padding: 24px 0;
            letter-spacing: 0.04em;
        }
    </style>

    <div class="head">
        <div class="hint-group">
            <span class="eyebrow" data-hint-label>INTERACTIVE · drag to explore</span>
            <span class="kbdhint" aria-hidden="true"><kbd>◀</kbd><kbd>▶</kbd></span>
        </div>
        <div class="readout" data-readout></div>
    </div>

    <div class="metric-tabs" data-metric-tabs role="tablist"></div>

    <div class="chart-wrap">
        <svg data-chart viewBox="0 0 600 260" preserveAspectRatio="none" role="img" aria-label="sweep chart"></svg>
        <div class="empty" data-empty hidden>offline — open via <code>python -m http.server</code></div>
    </div>

    <div class="slider-row">
        <input type="range" data-slider aria-label="x-axis value" />
    </div>
    <div class="ticks" data-ticks></div>

    <div class="legend" data-legend></div>
`;

class SweepSlider extends HTMLElement {
    static get observedAttributes() { return ['data-src', 'data-metric']; }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' }).appendChild(TEMPLATE.content.cloneNode(true));
        this._data = null;
        this._idx = 0;
        this._hiddenSeries = new Set();
        this._activeMetricKey = null;
        this._drawn = false;
        this._reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

        // Cached DOM handles.
        this.$head    = this.shadowRoot.querySelector('[data-readout]');
        this.$hint    = this.shadowRoot.querySelector('[data-hint-label]');
        this.$chart   = this.shadowRoot.querySelector('[data-chart]');
        this.$empty   = this.shadowRoot.querySelector('[data-empty]');
        this.$slider  = this.shadowRoot.querySelector('[data-slider]');
        this.$ticks   = this.shadowRoot.querySelector('[data-ticks]');
        this.$tabs    = this.shadowRoot.querySelector('[data-metric-tabs]');
        this.$legend  = this.shadowRoot.querySelector('[data-legend]');

        this.$slider.addEventListener('input', (e) => {
            this._setIndex(parseInt(e.target.value, 10));
        });
    }

    connectedCallback() {
        const hint = this.getAttribute('data-hint') || 'drag to explore';
        this.$hint.textContent = `INTERACTIVE · ${hint}`;

        this._seriesDefs = this._parseJSON('data-series', []);
        this._metricsDefs = this._parseJSON('data-metrics', []);
        this._activeMetricKey = this.getAttribute('data-metric');

        if (this._metricsDefs.length) this._renderMetricTabs();
        this._renderLegend();
        this._load();

        // Redraw on resize so chart geometry stays sharp.
        this._resizeObs = new ResizeObserver(() => this._draw());
        this._resizeObs.observe(this);

        // First-time draw-in via IntersectionObserver.
        if (!this._reducedMotion && 'IntersectionObserver' in window) {
            const io = new IntersectionObserver((entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) { this._playDrawIn(); io.disconnect(); }
                });
            }, { threshold: 0.15 });
            io.observe(this);
        }
    }

    disconnectedCallback() {
        this._resizeObs?.disconnect();
    }

    attributeChangedCallback(name) {
        if (!this.isConnected) return;
        if (name === 'data-src') this._load();
        if (name === 'data-metric') {
            this._activeMetricKey = this.getAttribute('data-metric');
            this._draw();
            this._updateReadout();
        }
    }

    _parseJSON(attr, fallback) {
        const raw = this.getAttribute(attr);
        if (!raw) return fallback;
        try { return JSON.parse(raw); } catch (e) {
            console.warn(`<sweep-slider> bad JSON in ${attr}:`, e);
            return fallback;
        }
    }

    async _load() {
        const src = this.getAttribute('data-src');
        if (!src) return;
        try {
            const fetcher = window.fetchJSONCached || ((u) => fetch(u).then((r) => r.json()));
            const data = await fetcher(src);
            if (data && data._stub) {
                console.warn(`<sweep-slider> stub data for ${src}:`, data._reason);
                this._showEmpty(true);
                return;
            }
            this._data = data;
            this._showEmpty(false);
            this._setup();
        } catch (err) {
            console.warn(`<sweep-slider> failed to load ${src}:`, err);
            this._showEmpty(true);
        }
    }

    _showEmpty(show) {
        this.$empty.hidden = !show;
        this.$chart.style.opacity = show ? 0.2 : 1;
    }

    _setup() {
        if (!this._data || !this._data.points) return;

        // Default-active metric: either the explicitly requested one, or the first.
        const metrics = this._metricsDefs.length
            ? this._metricsDefs
            : [{ key: this._activeMetricKey, label: this._activeMetricKey, unit: '', fmt: 'fixed' }];
        if (!this._activeMetricKey) this._activeMetricKey = metrics[0].key;

        // If the declared series don't match the file, fall back to whatever's there.
        if (!this._seriesDefs.length) {
            this._seriesDefs = (this._data.methods || []).map((m) => ({ key: m, color: 'var(--c-orange)' }));
            this._renderLegend();
        }

        // Slider bounds.
        const n = this._data.points.length;
        this.$slider.min = 0;
        this.$slider.max = Math.max(0, n - 1);
        this.$slider.step = 1;
        this._idx = Math.min(this._idx, n - 1);
        this.$slider.value = this._idx;

        this._renderTicks();
        this._draw();
        this._updateReadout();
    }

    _renderMetricTabs() {
        this.$tabs.innerHTML = '';
        this._metricsDefs.forEach((m) => {
            const b = document.createElement('button');
            b.type = 'button';
            b.role = 'tab';
            b.setAttribute('aria-pressed', m.key === this._activeMetricKey ? 'true' : 'false');
            b.textContent = m.label;
            b.addEventListener('click', () => {
                this._activeMetricKey = m.key;
                this.setAttribute('data-metric', m.key);
                this.shadowRoot.querySelectorAll('[data-metric-tabs] button').forEach((btn) => {
                    btn.setAttribute('aria-pressed', btn.textContent === m.label ? 'true' : 'false');
                });
                this._draw();
                this._updateReadout();
            });
            this.$tabs.appendChild(b);
        });
    }

    _renderLegend() {
        this.$legend.innerHTML = '';
        this._seriesDefs.forEach((s) => {
            const el = document.createElement('span');
            el.className = 'item';
            el.setAttribute('aria-pressed', 'true');
            el.setAttribute('role', 'button');
            el.tabIndex = 0;
            el.innerHTML = `<span class="swatch" style="background:${s.color}"></span>${s.key}`;
            const toggle = () => {
                if (this._hiddenSeries.has(s.key)) {
                    this._hiddenSeries.delete(s.key);
                    el.setAttribute('aria-pressed', 'true');
                } else {
                    this._hiddenSeries.add(s.key);
                    el.setAttribute('aria-pressed', 'false');
                }
                this._draw();
                this._updateReadout();
            };
            el.addEventListener('click', toggle);
            el.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
            });
            this.$legend.appendChild(el);
        });
    }

    _renderTicks() {
        this.$ticks.innerHTML = '';
        const n = this._data.points.length;
        this._data.points.forEach((p, i) => {
            const span = document.createElement('span');
            span.textContent = (p.label !== undefined) ? p.label : p.x;
            // Position each tick's center at its data point's x in chart-wrap
            // coords: pad-l + frac * (W − pad-l − pad-r). Mirrors xAt() in _draw.
            const frac = (n <= 1) ? 0.5 : i / (n - 1);
            span.style.left = `calc(var(--pad-l) * ${1 - frac} + (100% - var(--pad-r)) * ${frac})`;
            this.$ticks.appendChild(span);
        });
    }

    _setIndex(i) {
        const n = this._data?.points.length || 0;
        if (!n) return;
        this._idx = Math.max(0, Math.min(n - 1, i));
        this.$slider.value = this._idx;
        this._draw();
        this._updateReadout();
        this.dispatchEvent(new CustomEvent('sweep-change', {
            detail: { index: this._idx, point: this._data.points[this._idx] },
            bubbles: true,
        }));
    }

    _activeMetricDef() {
        return this._metricsDefs.find((m) => m.key === this._activeMetricKey)
            || { key: this._activeMetricKey, label: this._activeMetricKey, unit: '', fmt: 'fixed' };
    }

    _fmt(v, fmt, unit) {
        if (v == null || Number.isNaN(v)) return '—';
        let out;
        switch (fmt) {
            case 'ratio':  out = v.toFixed(3) + '×'; break;
            case 'log':    out = v >= 100 ? v.toFixed(0) : v.toFixed(1); out += unit ? ` ${unit}` : ''; break;
            case 'fixed':  out = v.toFixed(2); out += unit ? ` ${unit}` : ''; break;
            default:       out = String(v) + (unit ? ` ${unit}` : '');
        }
        return out;
    }

    _updateReadout() {
        const pt = this._data?.points[this._idx];
        if (!pt) { this.$head.innerHTML = ''; return; }
        const metric = this._activeMetricDef();

        const chips = [];
        const xkey = this.getAttribute('data-x-key') || 'x';
        chips.push(`<span class="metric"><span>${xkey.toUpperCase()}</span><span class="val">${pt.label ?? pt.x}</span></span>`);
        if (pt.K !== undefined) {
            chips.push(`<span class="metric"><span>K</span><span class="val">${pt.K}</span></span>`);
        }

        this._seriesDefs.forEach((s) => {
            if (this._hiddenSeries.has(s.key)) return;
            const m = pt.metrics?.[s.key];
            if (!m) return;
            const v = m[metric.key];
            if (v == null) return;
            chips.push(`<span class="metric"><span class="dot" style="background:${s.color}"></span><span>${s.key}</span><span class="val">${this._fmt(v, metric.fmt, metric.unit)}</span></span>`);
        });
        this.$head.innerHTML = chips.join('');
    }

    _playDrawIn() {
        if (!this._data) { return; } // will run on next _draw()
        const lines = this.$chart.querySelectorAll('.series-line');
        lines.forEach((line) => {
            const len = line.getTotalLength ? line.getTotalLength() : 600;
            line.style.strokeDasharray = `${len} ${len}`;
            line.style.strokeDashoffset = `${len}`;
            requestAnimationFrame(() => {
                line.style.transition = 'stroke-dashoffset 900ms cubic-bezier(0.2, 0.8, 0.2, 1)';
                line.style.strokeDashoffset = '0';
            });
        });
    }

    _draw() {
        if (!this._data || !this._data.points) return;
        const compact = this.getAttribute('data-compact') === 'true';
        const rect = this.$chart.getBoundingClientRect();
        const W = Math.max(360, rect.width || 600);
        const H = compact ? 180 : 240;
        this.$chart.setAttribute('viewBox', `0 0 ${W} ${H}`);
        this.$chart.setAttribute('preserveAspectRatio', 'none');
        this.$chart.style.height = `${H}px`;

        const metric = this._activeMetricDef();
        const points = this._data.points;
        const xs = points.map((p) => p.x);

        // Collect y-values across visible series.
        const allY = [];
        for (const s of this._seriesDefs) {
            if (this._hiddenSeries.has(s.key)) continue;
            for (const p of points) {
                const v = p.metrics?.[s.key]?.[metric.key];
                if (v != null && Number.isFinite(v)) allY.push(v);
            }
        }
        if (!allY.length) { this.$chart.innerHTML = ''; return; }

        const useLog = metric.fmt === 'log';
        let yMin = Math.min(...allY);
        let yMax = Math.max(...allY);

        if (useLog) {
            const logMin = Math.log10(Math.max(yMin, 0.01));
            const logMax = Math.log10(Math.max(yMax, logMin + 0.1));
            const padLog = (logMax - logMin) * 0.1 || 0.1;
            yMin = Math.pow(10, logMin - padLog);
            yMax = Math.pow(10, logMax + padLog);
        } else {
            if (metric.fmt === 'ratio') {
                yMin = Math.min(yMin, 0.9);
                yMax = Math.max(yMax, 1.05);
            }
            const pad = (yMax - yMin) * 0.1 || Math.abs(yMax) * 0.1 || 1;
            yMin -= pad;
            yMax += pad;
        }

        const padL = compact ? 38 : 44;
        const padR = 14;
        const padT = 16;
        const padB = 28;

        const plotW = W - padL - padR;
        const plotH = H - padT - padB;

        const xAt = (i) => padL + (xs.length <= 1 ? plotW / 2 : (i / (xs.length - 1)) * plotW);
        const yAt = (v) => {
            if (useLog) {
                const lo = Math.log10(yMin);
                const hi = Math.log10(yMax);
                return padT + plotH - ((Math.log10(Math.max(v, yMin)) - lo) / (hi - lo)) * plotH;
            }
            return padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH;
        };

        const svg = [];

        // Grid + y ticks.
        const NGRID = 4;
        for (let g = 0; g <= NGRID; g++) {
            const v = useLog
                ? Math.pow(10, Math.log10(yMin) + (Math.log10(yMax) - Math.log10(yMin)) * (g / NGRID))
                : yMin + (yMax - yMin) * (g / NGRID);
            const y = yAt(v);
            svg.push(`<line class="grid-line" x1="${padL}" x2="${W - padR}" y1="${y}" y2="${y}" />`);
            svg.push(`<text class="axis-tick" x="${padL - 6}" y="${y + 3}" text-anchor="end">${this._fmtTick(v, metric.fmt)}</text>`);
        }

        // Cursor line at active x.
        const cx = xAt(this._idx);
        svg.push(`<line class="cursor-line" x1="${cx}" x2="${cx}" y1="${padT}" y2="${padT + plotH}" />`);
        svg.push(`<rect class="cursor-cap" x="${cx - 4}" y="${padT - 5}" width="8" height="3" rx="1.5" />`);

        // Series lines.
        this._seriesDefs.forEach((s) => {
            if (this._hiddenSeries.has(s.key)) return;
            const pts = points.map((p, i) => {
                const v = p.metrics?.[s.key]?.[metric.key];
                if (v == null || !Number.isFinite(v)) return null;
                return [xAt(i), yAt(v)];
            });
            const d = this._smoothPath(pts);
            if (!d) return;
            // Halo for the "active" feel (very faint).
            svg.push(`<path class="series-halo" d="${d}" stroke="${s.color}" />`);
            svg.push(`<path class="series-line" d="${d}" stroke="${s.color}" />`);
            pts.forEach((pt, i) => {
                if (!pt) return;
                const isActive = i === this._idx;
                const r = isActive ? 4.5 : 2.8;
                svg.push(`<circle class="series-dot${isActive ? ' is-active' : ''}" cx="${pt[0]}" cy="${pt[1]}" r="${r}" fill="${s.color}" opacity="${isActive ? 1 : 0.6}" />`);
            });
        });

        // x-axis label.
        const xLabel = this.getAttribute('data-x-label') || '';
        if (xLabel) {
            svg.push(`<text class="axis-label" x="${padL + plotW / 2}" y="${H - 6}" text-anchor="middle">${xLabel}</text>`);
        }
        // y-axis metric label (rotated).
        svg.push(`<text class="axis-label" x="${padL}" y="${padT - 6}" text-anchor="start">${metric.label}${metric.unit ? ' (' + metric.unit + ')' : ''}</text>`);

        this.$chart.innerHTML = svg.join('\n');

        // Draw-in on first draw after entering viewport.
        if (!this._reducedMotion && !this._drawn) {
            this._drawn = true;
            this._playDrawIn();
        }
    }

    _fmtTick(v, fmt) {
        if (fmt === 'ratio')  return v.toFixed(2);
        if (fmt === 'log')    return v >= 100 ? v.toFixed(0) : v.toFixed(1);
        if (Math.abs(v) >= 100) return v.toFixed(0);
        if (Math.abs(v) >= 10)  return v.toFixed(1);
        return v.toFixed(2);
    }

    _smoothPath(points) {
        const filtered = points.filter(Boolean);
        if (!filtered.length) return null;
        if (filtered.length === 1) {
            return `M${filtered[0][0]},${filtered[0][1]}`;
        }
        // Simple cardinal-ish smoothing with low tension so the line still feels "honest".
        const t = 0.22;
        const pts = filtered;
        const d = [`M${pts[0][0]},${pts[0][1]}`];
        for (let i = 0; i < pts.length - 1; i++) {
            const p0 = pts[i === 0 ? i : i - 1];
            const p1 = pts[i];
            const p2 = pts[i + 1];
            const p3 = pts[i + 2 < pts.length ? i + 2 : i + 1];
            const cp1x = p1[0] + (p2[0] - p0[0]) * t;
            const cp1y = p1[1] + (p2[1] - p0[1]) * t;
            const cp2x = p2[0] - (p3[0] - p1[0]) * t;
            const cp2y = p2[1] - (p3[1] - p1[1]) * t;
            d.push(`C${cp1x.toFixed(2)},${cp1y.toFixed(2)} ${cp2x.toFixed(2)},${cp2y.toFixed(2)} ${p2[0]},${p2[1]}`);
        }
        return d.join(' ');
    }
}

customElements.define('sweep-slider', SweepSlider);
