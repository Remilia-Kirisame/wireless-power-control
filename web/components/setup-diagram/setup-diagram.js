/* <setup-diagram> — static setup schematics for the D2D and JSAC chapters. */

/* Tune: alpha to change the line opacity */
const D2D_INTERFERENCE = [
    { alpha: 0.16, x1: 70,  y1: 80,  x2: 205, y2: 200 },
    { alpha: 0.24, x1: 300, y1: 90,  x2: 205, y2: 200 },
    { alpha: 0.34, x1: 320, y1: 220, x2: 205, y2: 200 },
    { alpha: 0.30, x1: 80,  y1: 240, x2: 205, y2: 200 },
    { alpha: 0.52, x1: 240, y1: 250, x2: 205, y2: 200 },
];

const JSAC_INTERFERENCE = [
    { alpha: 0.16, x1: 90,  y1: 90,  x2: 250, y2: 280 },
    { alpha: 0.27, x1: 300, y1: 110, x2: 250, y2: 280 },
    { alpha: 0.40, x1: 115, y1: 245, x2: 250, y2: 280 },
];

const SHARED_STYLES = `
    :host {
        display: block;
    }
    svg {
        display: block;
        width: 100%;
        height: auto;
    }
    .setup-interference-link {
        stroke: var(--c-grey);
        stroke-width: var(--setup-interference-width);
        stroke-dasharray: var(--setup-interference-dash);
        opacity: var(--line-alpha, 0.26);
    }
    .setup-legend {
        font-family: var(--font-mono);
        font-size: 9px;
        fill: var(--text-mute);
        letter-spacing: 1.4px;
    }
`;

const interferenceLines = (links) => links.map(({ alpha, x1, y1, x2, y2 }) => (
    `<line class="setup-interference-link" style="--line-alpha: ${alpha}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`
)).join('');

const d2dTemplate = () => `
    <style>${SHARED_STYLES}</style>
    <svg viewBox="0 0 400 340" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <marker id="d2d-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0,0 L10,5 L0,10 z" fill="var(--c-blue)" opacity="0.6"/>
            </marker>
        </defs>
        <rect x="20" y="20" width="360" height="280" rx="8" fill="none" stroke="rgba(255,255,255,0.06)" stroke-dasharray="2 4"/>
        <g>${interferenceLines(D2D_INTERFERENCE)}</g>
        <g stroke="var(--c-blue)" stroke-width="1.4" opacity="0.9" marker-end="url(#d2d-arrow)">
            <line x1="70" y1="80" x2="130" y2="115"/>
            <line x1="300" y1="90" x2="260" y2="135"/>
            <line x1="150" y1="180" x2="205" y2="200"/>
            <line x1="320" y1="220" x2="275" y2="195"/>
            <line x1="80" y1="240" x2="140" y2="235"/>
            <line x1="240" y1="250" x2="200" y2="265"/>
        </g>
        <g fill="var(--c-blue)">
            <rect x="65" y="75" width="10" height="10"/>
            <rect x="295" y="85" width="10" height="10"/>
            <rect x="145" y="175" width="10" height="10"/>
            <rect x="315" y="215" width="10" height="10"/>
            <rect x="75" y="235" width="10" height="10"/>
            <rect x="235" y="245" width="10" height="10"/>
        </g>
        <g fill="none" stroke="var(--c-blue)" stroke-width="1.6">
            <circle cx="130" cy="115" r="6"/>
            <circle cx="260" cy="135" r="6"/>
            <circle cx="205" cy="200" r="6"/>
            <circle cx="275" cy="195" r="6"/>
            <circle cx="140" cy="235" r="6"/>
            <circle cx="200" cy="265" r="6"/>
        </g>
        <text class="setup-legend" x="36" y="325" xml:space="preserve">DIRECT LINK ───    INTERFERENCE - - -</text>
    </svg>
`;

const jsacTemplate = () => `
    <style>${SHARED_STYLES}</style>
    <svg viewBox="0 0 400 340" xmlns="http://www.w3.org/2000/svg">
        <rect x="20" y="20" width="360" height="280" rx="8" fill="none" stroke="rgba(255,255,255,0.06)" stroke-dasharray="2 4"/>
        <g>${interferenceLines(JSAC_INTERFERENCE)}</g>
        <g stroke="var(--c-yellow)" stroke-width="1" opacity="0.8">
            <line x1="90" y1="90" x2="55" y2="65"/>
            <line x1="90" y1="90" x2="130" y2="50"/>
        </g>
        <g stroke="var(--c-green)" stroke-width="1" opacity="0.8">
            <line x1="90" y1="90" x2="55" y2="130"/>
            <line x1="90" y1="90" x2="140" y2="115"/>
        </g>
        <g stroke="var(--c-yellow)" stroke-width="1" opacity="0.8">
            <line x1="300" y1="110" x2="265" y2="65"/>
            <line x1="300" y1="110" x2="345" y2="60"/>
        </g>
        <g stroke="var(--c-green)" stroke-width="1" opacity="0.8">
            <line x1="300" y1="110" x2="265" y2="155"/>
            <line x1="300" y1="110" x2="350" y2="140"/>
        </g>
        <g stroke="var(--c-yellow)" stroke-width="1" opacity="0.8">
            <line x1="115" y1="245" x2="80" y2="205"/>
            <line x1="115" y1="245" x2="70" y2="280"/>
        </g>
        <g stroke="var(--c-green)" stroke-width="1" opacity="0.8">
            <line x1="115" y1="245" x2="150" y2="210"/>
            <line x1="115" y1="245" x2="160" y2="280"/>
        </g>
        <g stroke="var(--c-yellow)" stroke-width="1" opacity="0.8">
            <line x1="290" y1="245" x2="250" y2="205"/>
            <line x1="290" y1="245" x2="330" y2="205"/>
        </g>
        <g stroke="var(--c-green)" stroke-width="1" opacity="0.8">
            <line x1="290" y1="245" x2="250" y2="280"/>
            <line x1="290" y1="245" x2="340" y2="280"/>
        </g>
        <g fill="var(--c-blue)">
            <rect x="85" y="85" width="10" height="10"/>
            <rect x="295" y="105" width="10" height="10"/>
            <rect x="110" y="240" width="10" height="10"/>
            <rect x="285" y="240" width="10" height="10"/>
        </g>
        <g fill="none" stroke="var(--c-yellow)" stroke-width="1.6">
            <circle cx="55" cy="65" r="5"/>
            <circle cx="130" cy="50" r="5"/>
            <circle cx="265" cy="65" r="5"/>
            <circle cx="345" cy="60" r="5"/>
            <circle cx="80" cy="205" r="5"/>
            <circle cx="70" cy="280" r="5"/>
            <circle cx="250" cy="205" r="5"/>
            <circle cx="330" cy="205" r="5"/>
        </g>
        <g fill="none" stroke="var(--c-green)" stroke-width="1.6">
            <circle cx="55" cy="130" r="5"/>
            <circle cx="140" cy="115" r="5"/>
            <circle cx="265" cy="155" r="5"/>
            <circle cx="350" cy="140" r="5"/>
            <circle cx="150" cy="210" r="5"/>
            <circle cx="160" cy="280" r="5"/>
            <circle cx="250" cy="280" r="5"/>
            <circle cx="340" cy="280" r="5"/>
        </g>
        <text class="setup-legend" x="36" y="325" xml:space="preserve">TX ▪   YELLOW ◯   GREEN ◯   INTER-CLUSTER - - -</text>
    </svg>
`;

class SetupDiagram extends HTMLElement {
    static get observedAttributes() {
        return ['variant'];
    }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
    }

    connectedCallback() {
        this.render();
    }

    attributeChangedCallback() {
        if (this.isConnected) this.render();
    }

    render() {
        const variant = this.getAttribute('variant') === 'jsac' ? 'jsac' : 'd2d';
        this.shadowRoot.innerHTML = variant === 'jsac' ? jsacTemplate() : d2dTemplate();
    }
}

customElements.define('setup-diagram', SetupDiagram);
