/* <method-toggle> — segmented control that emits `method-change` events.
 *
 * Attributes:
 *   data-methods   Comma-separated list: "Naive,WMMSE,GNN".
 *   data-default   "all" (default) or a single method name.
 *   data-multi     "true" allows multi-select; otherwise single-select radio behaviour.
 *   data-colors    Optional comma-separated color CSS values aligned with methods.
 *
 * Emits:
 *   CustomEvent("method-change", { detail: { active: Set<string>, last: string } })
 */

const TEMPLATE = document.createElement('template');
TEMPLATE.innerHTML = /* html */ `
    <style>
        :host {
            display: inline-flex;
            gap: 2px;
            padding: 3px;
            border: 1px solid var(--rule);
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.25);
            font-family: var(--font-mono);
        }
        button {
            background: transparent;
            border: none;
            color: var(--text-dim);
            font-family: inherit;
            font-size: 11px;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            padding: 6px 14px;
            border-radius: 5px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: color var(--dur-fast, 160ms) var(--ease, ease), background var(--dur-fast, 160ms) var(--ease, ease);
        }
        button:hover { color: var(--text); }
        button:focus-visible {
            outline: none;
            box-shadow: 0 0 0 2px rgba(255, 106, 61, 0.4);
        }
        button[aria-pressed="true"] {
            color: var(--text);
            background: rgba(255, 255, 255, 0.05);
        }
        button .swatch {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-mute);
        }
        button[aria-pressed="true"] .swatch { box-shadow: 0 0 6px currentColor; }
    </style>
`;

class MethodToggle extends HTMLElement {
    static get observedAttributes() { return ['data-methods', 'data-default']; }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' }).appendChild(TEMPLATE.content.cloneNode(true));
        this._active = new Set();
    }

    connectedCallback() {
        this._render();
    }

    attributeChangedCallback() {
        if (this.isConnected) this._render();
    }

    get active() { return new Set(this._active); }

    set(method, on = true) {
        if (this._multi) {
            if (on) this._active.add(method); else this._active.delete(method);
        } else {
            this._active.clear();
            if (on) this._active.add(method);
        }
        this._sync();
        this._emit(method);
    }

    _render() {
        const methods = (this.getAttribute('data-methods') || '').split(',').map((s) => s.trim()).filter(Boolean);
        const colors  = (this.getAttribute('data-colors')  || '').split(',').map((s) => s.trim());
        const defaultAttr = (this.getAttribute('data-default') || 'all').toLowerCase();
        this._multi = this.getAttribute('data-multi') === 'true' || defaultAttr === 'all';
        this._methods = methods;

        // seed _active
        this._active.clear();
        if (defaultAttr === 'all') {
            methods.forEach((m) => this._active.add(m));
        } else if (methods.includes(this.getAttribute('data-default'))) {
            this._active.add(this.getAttribute('data-default'));
        } else if (methods.length) {
            this._active.add(methods[0]);
        }

        // Clear previously rendered buttons.
        this.shadowRoot.querySelectorAll('button').forEach((b) => b.remove());

        methods.forEach((m, i) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.setAttribute('aria-pressed', this._active.has(m) ? 'true' : 'false');
            btn.setAttribute('aria-label', m);
            const color = colors[i] || 'var(--text-mute)';
            btn.innerHTML = `<span class="swatch" style="background:${color};color:${color};"></span>${m}`;
            btn.addEventListener('click', () => this._toggle(m));
            this.shadowRoot.appendChild(btn);
        });
    }

    _toggle(method) {
        if (this._multi) {
            if (this._active.has(method)) {
                if (this._active.size > 1) this._active.delete(method);
            } else {
                this._active.add(method);
            }
        } else {
            this._active.clear();
            this._active.add(method);
        }
        this._sync();
        this._emit(method);
    }

    _sync() {
        this.shadowRoot.querySelectorAll('button').forEach((b) => {
            const m = b.getAttribute('aria-label');
            b.setAttribute('aria-pressed', this._active.has(m) ? 'true' : 'false');
        });
    }

    _emit(last) {
        this.dispatchEvent(new CustomEvent('method-change', {
            detail: { active: new Set(this._active), last },
            bubbles: true,
        }));
    }
}

customElements.define('method-toggle', MethodToggle);
