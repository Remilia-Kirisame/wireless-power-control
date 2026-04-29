/* <layout-gallery> — carousel of pre-exported JSAC layouts.
 *
 * Uses <layout-viewer> internally to render the active snapshot; prev/next arrows
 * and keyboard (←/→) cycle through seeds. The method-toggle inside each viewer
 * handles Naive/WMMSE/GNN crossfade.
 *
 * Attributes:
 *   data-src   Path to layouts_index.json.
 */

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
        .kbdhint {
            display: inline-flex;
            gap: 4px;
            align-items: center;
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
            background: rgba(0,0,0,0.25);
            color: var(--text-dim);
        }

        .stage {
            display: grid;
            grid-template-columns: 1fr;
            gap: 12px;
            position: relative;
        }

        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
        }
        .nav {
            display: flex;
            gap: 8px;
        }
        .nav button {
            background: transparent;
            border: 1px solid var(--rule);
            color: var(--text-dim);
            font-family: var(--font-mono);
            font-size: 12px;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            transition: color var(--dur-fast, 160ms) var(--ease, ease), border-color var(--dur-fast, 160ms) var(--ease, ease);
        }
        .nav button:hover { color: var(--text); border-color: var(--c-orange); }
        .nav button:focus-visible { outline: none; box-shadow: 0 0 0 2px rgba(255, 106, 61, 0.4); }

        .thumbs {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }
        .thumb {
            font-family: var(--font-mono);
            font-size: 10px;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            padding: 5px 10px;
            border-radius: 999px;
            border: 1px solid var(--rule);
            color: var(--text-mute);
            background: transparent;
            cursor: pointer;
            transition: color var(--dur-fast, 160ms) var(--ease, ease), border-color var(--dur-fast, 160ms) var(--ease, ease), background var(--dur-fast, 160ms) var(--ease, ease);
        }
        .thumb[aria-pressed="true"] {
            color: var(--text);
            border-color: var(--c-orange);
            background: rgba(255, 106, 61, 0.08);
        }
        .thumb:hover { color: var(--text); }

        .viewer-slot { position: relative; min-height: 200px; }

        .empty {
            color: var(--text-mute);
            font-family: var(--font-mono);
            font-size: 12px;
            padding: 40px 10px;
            text-align: center;
        }
    </style>

    <div class="head">
        <span class="eyebrow">INTERACTIVE · ← → to navigate layouts</span>
        <span class="kbdhint"><kbd>◀</kbd><kbd>▶</kbd> layouts &nbsp; <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> methods</span>
    </div>

    <div class="stage">
        <div class="controls">
            <div class="nav">
                <button type="button" data-prev aria-label="Previous layout">◀ prev</button>
                <button type="button" data-next aria-label="Next layout">next ▶</button>
            </div>
            <div class="thumbs" data-thumbs role="tablist"></div>
        </div>

        <div class="viewer-slot" data-slot></div>
    </div>

    <div class="empty" data-empty hidden>offline — open via <code>python -m http.server</code></div>
`;

class LayoutGallery extends HTMLElement {
    static get observedAttributes() { return ['data-src']; }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' }).appendChild(TEMPLATE.content.cloneNode(true));
        this._index = null;
        this._i = 0;
        this._viewer = null;

        this.$slot   = this.shadowRoot.querySelector('[data-slot]');
        this.$prev   = this.shadowRoot.querySelector('[data-prev]');
        this.$next   = this.shadowRoot.querySelector('[data-next]');
        this.$thumbs = this.shadowRoot.querySelector('[data-thumbs]');
        this.$empty  = this.shadowRoot.querySelector('[data-empty]');

        this.$prev.addEventListener('click', () => this._step(-1));
        this.$next.addEventListener('click', () => this._step(+1));
    }

    connectedCallback() {
        this.tabIndex = 0;
        this.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft')  { e.preventDefault(); this._step(-1); }
            if (e.key === 'ArrowRight') { e.preventDefault(); this._step(+1); }
            if (['1','2','3'].includes(e.key) && this._viewer) {
                e.preventDefault();
                const methods = ['Naive', 'WMMSE', 'GNN'];
                const toggle = this._viewer.shadowRoot?.querySelector('method-toggle');
                if (toggle) toggle.set(methods[Number(e.key) - 1]);
            }
        });
        this._load();
    }

    attributeChangedCallback(name) {
        if (this.isConnected && name === 'data-src') this._load();
    }

    async _load() {
        const src = this.getAttribute('data-src');
        if (!src) return;
        try {
            const fetcher = window.fetchJSONCached || ((u) => fetch(u).then((r) => r.json()));
            const data = await fetcher(src);
            if (data?._stub) { this._showEmpty(true); return; }
            this._index = data?.layouts || [];
            if (!this._index.length) { this._showEmpty(true); return; }
            this._showEmpty(false);
            this._renderThumbs();
            this._go(0);
        } catch (err) {
            console.warn(`<layout-gallery> failed to load ${src}:`, err);
            this._showEmpty(true);
        }
    }

    _showEmpty(show) {
        this.$empty.hidden = !show;
    }

    _renderThumbs() {
        this.$thumbs.innerHTML = '';
        this._index.forEach((entry, i) => {
            const b = document.createElement('button');
            b.className = 'thumb';
            b.type = 'button';
            b.setAttribute('aria-pressed', 'false');
            b.textContent = `seed ${entry.id}`;
            b.addEventListener('click', () => this._go(i));
            this.$thumbs.appendChild(b);
        });
    }

    _step(delta) {
        if (!this._index?.length) return;
        const n = this._index.length;
        this._go((this._i + delta + n) % n);
    }

    _go(i) {
        this._i = i;
        this.$thumbs.querySelectorAll('.thumb').forEach((b, idx) => {
            b.setAttribute('aria-pressed', idx === i ? 'true' : 'false');
        });

        const entry = this._index[i];
        if (!entry) return;
        const base = this.getAttribute('data-src') || '';
        const baseDir = base.includes('/') ? base.slice(0, base.lastIndexOf('/') + 1) : '';
        const path = entry.path.startsWith('layouts/') ? (baseDir + entry.path) : entry.path;

        // Reuse the same viewer for crossfade smoothness.
        if (!this._viewer) {
            this._viewer = document.createElement('layout-viewer');
            this._viewer.setAttribute('data-compact', 'true');
            this._viewer.setAttribute('data-initial', 'GNN');
            this.$slot.appendChild(this._viewer);
        }
        // Trigger a quick fade on the viewer slot.
        this.$slot.animate?.(
            [{ opacity: 0.4 }, { opacity: 1 }],
            { duration: 180, easing: 'ease-out' }
        );
        this._viewer.setAttribute('data-src', path);
    }
}

customElements.define('layout-gallery', LayoutGallery);
