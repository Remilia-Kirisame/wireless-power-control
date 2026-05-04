const SVG_NS = 'http://www.w3.org/2000/svg';
const EPS = 1e-12;

export const D2D_SIGNAL_ANIMATION_STYLES = /* css */ `
        .d2d-direct-base,
        .d2d-direct-glow,
        .d2d-direct-packet,
        .d2d-emission-pulse,
        .d2d-arrival-pulse,
        .d2d-tx-power-glow,
        .d2d-rx-power-glow,
        .message-edge,
        .heat-focus-line {
            pointer-events: none;
        }
        .d2d-direct-base,
        .d2d-direct-glow,
        .d2d-tx-power-glow,
        .d2d-rx-power-glow {
            transition: opacity var(--dur-fast) var(--ease), stroke-width var(--dur-fast) var(--ease), r var(--dur-fast) var(--ease);
        }
        .d2d-direct-packet {
            animation: d2dDirectPacket var(--packet-duration, 2.2s) linear infinite;
            animation-delay: var(--packet-delay, 0s);
            opacity: 0;
            stroke-linecap: round;
            filter: drop-shadow(0 0 7px rgba(255, 106, 61, 0.5));
        }
        @keyframes d2dDirectPacket {
            0% { stroke-dashoffset: 104; opacity: 0; }
            11% { opacity: var(--packet-opacity, 0.62); }
            78% { opacity: var(--packet-opacity, 0.62); }
            100% { stroke-dashoffset: 0; opacity: 0; }
        }
        .d2d-emission-pulse {
            animation: d2dEmissionPulse var(--pulse-duration, 2.2s) ease-out infinite;
            animation-delay: var(--pulse-delay, 0s);
            opacity: 0;
            transform-box: fill-box;
            transform-origin: center;
            filter: drop-shadow(0 0 8px rgba(77, 163, 255, 0.42));
        }
        .d2d-arrival-pulse {
            animation: d2dArrivalPulse var(--pulse-duration, 2.2s) ease-out infinite;
            animation-delay: var(--pulse-delay, 0s);
            opacity: 0;
            transform-box: fill-box;
            transform-origin: center;
            filter: drop-shadow(0 0 8px rgba(255, 106, 61, 0.44));
        }
        @keyframes d2dEmissionPulse {
            0% { opacity: 0; transform: scale(0.42); }
            8% { opacity: var(--pulse-opacity, 0.42); transform: scale(0.68); }
            30% { opacity: 0; transform: scale(1.35); }
            100% { opacity: 0; transform: scale(1.35); }
        }
        @keyframes d2dArrivalPulse {
            0%, 68% { opacity: 0; transform: scale(0.5); }
            79% { opacity: var(--pulse-opacity, 0.44); transform: scale(0.72); }
            100% { opacity: 0; transform: scale(1.55); }
        }
        .message-edge {
            animation: messagePulse 1.6s linear infinite;
            animation-delay: var(--edge-delay, 0s);
        }
        @keyframes messagePulse {
            0% { stroke-dashoffset: 0; opacity: var(--edge-opacity-low, 0.16); }
            45% { opacity: var(--edge-opacity-high, 0.62); }
            100% { stroke-dashoffset: -26; opacity: var(--edge-opacity-low, 0.16); }
        }
        .heat-focus-line {
            animation: focusPulse 1.1s ease-in-out infinite;
        }
        @keyframes focusPulse {
            0%, 100% { opacity: 0.48; }
            50% { opacity: 0.95; }
        }
`;

function svgEl(name, attrs = {}) {
    const el = document.createElementNS(SVG_NS, name);
    for (const [k, v] of Object.entries(attrs)) {
        if (v !== undefined && v !== null) el.setAttribute(k, String(v));
    }
    return el;
}

function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

function fmt(v, digits = 2) {
    return Number.isFinite(v) ? Number(v).toFixed(digits) : '--';
}

function phaseDelay(now, duration, offset = 0) {
    const seconds = (Number.isFinite(now) ? now : performance.now()) / 1000;
    return -((seconds + offset) % duration);
}

function selectedIndex(selected) {
    if (!selected || !Number.isFinite(selected.index)) return -1;
    return selected.index;
}

function focusMultiplier(selected, index) {
    const focus = selectedIndex(selected);
    if (focus < 0) return 1;
    return focus === index ? 1.18 : 0.34;
}

function activeRates(rates, k) {
    if (!Array.isArray(rates)) return new Array(k).fill(0);
    return rates.slice(0, k).map((v) => (Number.isFinite(v) ? Math.max(0, v) : 0));
}

function appendDirectBase(svg, state) {
    const { tx, rx, k, power, selected, isSettling } = state;
    for (let i = 0; i < k; i++) {
        if (!tx[i] || !rx[i]) continue;
        const p = clamp(power[i] || 0, 0, 1);
        const root = Math.sqrt(p);
        const focus = focusMultiplier(selected, i);
        const settling = isSettling ? 0.52 : 1;
        const opacity = clamp((0.15 + 0.58 * root) * focus * settling, 0.045, 0.9);
        const width = 1.15 + 2.35 * root + (selectedIndex(selected) === i ? 0.7 : 0);
        if (p > 0.025) {
            svg.appendChild(svgEl('line', {
                class: 'd2d-direct-glow',
                x1: tx[i].x,
                y1: tx[i].y,
                x2: rx[i].x,
                y2: rx[i].y,
                stroke: `rgba(255,106,61,${fmt(opacity * 0.22, 3)})`,
                'stroke-width': fmt(width + 6.5, 2),
                'stroke-linecap': 'round',
            }));
        }
        svg.appendChild(svgEl('line', {
            class: 'd2d-direct-base',
            x1: tx[i].x,
            y1: tx[i].y,
            x2: rx[i].x,
            y2: rx[i].y,
            stroke: `rgba(255,106,61,${fmt(opacity, 3)})`,
            'stroke-width': fmt(width, 2),
        }));
    }
}

function interferenceCandidates({ tx, rx, k, power, losses, selected }) {
    if (!losses) return [];
    const focus = selectedIndex(selected);
    const edges = [];
    for (let target = 0; target < k; target++) {
        for (let source = 0; source < k; source++) {
            if (target === source || !tx[source] || !rx[target]) continue;
            if (focus >= 0 && selected?.kind === 'rx' && target !== focus) continue;
            if (focus >= 0 && selected?.kind === 'tx' && source !== focus) continue;
            edges.push({
                target,
                source,
                score: (losses[target]?.[source] || 0) * (power[source] || 0),
            });
        }
    }
    edges.sort((a, b) => b.score - a.score);
    return edges;
}

function appendInterference(svg, state) {
    const { tx, rx, losses, selected, isSettling, reduceMotion, now } = state;
    const edges = interferenceCandidates(state);
    if (!edges.length || !losses) return;

    const focus = selectedIndex(selected);
    const maxEdges = focus >= 0 ? 12 : 70;
    const maxScore = edges[0]?.score || 1;
    edges.slice(0, Math.min(maxEdges, edges.length)).forEach((edge, n) => {
        const normalized = Math.sqrt(edge.score / (maxScore + EPS));
        const baseHigh = focus >= 0 ? 0.72 : 0.46;
        const high = clamp(0.1 + baseHigh * normalized, 0.1, focus >= 0 ? 0.82 : 0.48) * (isSettling ? 0.5 : 1);
        const low = clamp(high * 0.38, 0.04, 0.34);
        const delay = reduceMotion ? 0 : phaseDelay(now, 1.6, n * 0.08);
        svg.appendChild(svgEl('line', {
            class: 'message-edge',
            x1: tx[edge.source].x,
            y1: tx[edge.source].y,
            x2: rx[edge.target].x,
            y2: rx[edge.target].y,
            stroke: `rgba(77,163,255,${fmt(high, 3)})`,
            'stroke-width': focus >= 0 ? 1.7 : 1.4,
            'stroke-dasharray': '5 8',
            style: `--edge-delay:${fmt(delay, 3)}s;--edge-opacity-low:${fmt(low, 3)};--edge-opacity-high:${fmt(high, 3)};`,
        }));
    });
}

function appendHeatFocus(svg, { tx, rx, hoverEdge }) {
    if (!hoverEdge || !tx[hoverEdge.source] || !rx[hoverEdge.target]) return;
    const source = tx[hoverEdge.source];
    const target = rx[hoverEdge.target];
    svg.appendChild(svgEl('line', {
        class: 'heat-focus-line',
        x1: source.x,
        y1: source.y,
        x2: target.x,
        y2: target.y,
        stroke: 'rgba(255,255,255,0.92)',
        'stroke-width': 2.6,
        'stroke-dasharray': '7 5',
    }));
    svg.appendChild(svgEl('circle', {
        cx: target.x,
        cy: target.y,
        r: 14,
        fill: 'none',
        stroke: 'rgba(255,255,255,0.75)',
        'stroke-width': 2,
    }));
}

function appendDirectPackets(svg, state) {
    const { tx, rx, k, power, selected, isSettling, reduceMotion, now } = state;
    if (reduceMotion || isSettling) return;
    const rates = activeRates(state.rates, k);
    const maxRate = Math.max(...rates, 0.1);

    for (let i = 0; i < k; i++) {
        if (!tx[i] || !rx[i]) continue;
        const p = clamp(power[i] || 0, 0, 1);
        const focus = focusMultiplier(selected, i);
        const threshold = selectedIndex(selected) === i ? 0.025 : 0.04;
        if (p < threshold || focus < 0.18) continue;

        const root = Math.sqrt(p);
        const distance = Math.hypot(rx[i].x - tx[i].x, rx[i].y - tx[i].y);
        const duration = clamp(2.45 - 0.82 * root + distance / 2400, 1.28, 2.7);
        const packetLen = clamp(6.2 + 10.2 * root, 6.2, 16.4);
        const opacity = clamp((0.34 + 0.58 * root) * focus, 0.16, 0.96);
        const width = 3.0 + 4.6 * root + (selectedIndex(selected) === i ? 0.9 : 0);
        const count = p >= 0.45 && distance > 30 ? 2 : 1;

        for (let n = 0; n < count; n++) {
            const delay = phaseDelay(now, duration, i * 0.19 + n * duration / count);
            svg.appendChild(svgEl('line', {
                class: 'd2d-direct-packet',
                x1: tx[i].x,
                y1: tx[i].y,
                x2: rx[i].x,
                y2: rx[i].y,
                pathLength: 100,
                stroke: `rgba(255,142,83,${fmt(opacity, 3)})`,
                'stroke-width': fmt(width, 2),
                'stroke-dasharray': `${fmt(packetLen, 2)} ${fmt(Math.max(45, 100 - packetLen), 2)}`,
                style: `--packet-duration:${fmt(duration, 3)}s;--packet-delay:${fmt(delay, 3)}s;--packet-opacity:${fmt(opacity, 3)};`,
            }));
        }

        const rateNorm = clamp(rates[i] / (maxRate + EPS), 0, 1);
        const pulseOpacity = clamp((0.30 + 0.36 * root + 0.18 * rateNorm) * focus, 0.18, 0.82);
        for (let n = 0; n < count; n++) {
            const delay = phaseDelay(now, duration, i * 0.19 + n * duration / count);
            svg.appendChild(svgEl('circle', {
                class: 'd2d-emission-pulse',
                cx: tx[i].x,
                cy: tx[i].y,
                r: fmt(8.5 + 9.2 * root, 2),
                fill: `rgba(77,163,255,${fmt(pulseOpacity, 3)})`,
                stroke: 'none',
                style: `--pulse-duration:${fmt(duration, 3)}s;--pulse-delay:${fmt(delay, 3)}s;--pulse-opacity:${fmt(pulseOpacity, 3)};`,
            }));
            svg.appendChild(svgEl('circle', {
                class: 'd2d-arrival-pulse',
                cx: rx[i].x,
                cy: rx[i].y,
                r: fmt(8.0 + 7.2 * root + 4.2 * rateNorm, 2),
                fill: `rgba(255,106,61,${fmt(pulseOpacity, 3)})`,
                stroke: 'none',
                style: `--pulse-duration:${fmt(duration, 3)}s;--pulse-delay:${fmt(delay, 3)}s;--pulse-opacity:${fmt(pulseOpacity, 3)};`,
            }));
        }
    }
}

export function appendD2DSignalLayers(svg, options) {
    const state = {
        ...options,
        k: options.k || Math.min(options.tx?.length || 0, options.rx?.length || 0),
        power: options.power || [],
        rates: options.rates || [],
        losses: options.losses || null,
        reduceMotion: Boolean(options.reduceMotion),
        isSettling: Boolean(options.isSettling),
        now: Number.isFinite(options.now) ? options.now : performance.now(),
    };
    appendDirectBase(svg, state);
    appendInterference(svg, state);
    appendHeatFocus(svg, state);
    appendDirectPackets(svg, state);
}

export function appendD2DNodePowerGlows(svg, options) {
    const tx = options.tx || [];
    const rx = options.rx || [];
    const k = options.k || Math.min(tx.length, rx.length);
    const power = options.power || [];
    const selected = options.selected;
    const isSettling = Boolean(options.isSettling);
    for (let i = 0; i < k; i++) {
        const p = clamp(power[i] || 0, 0, 1);
        if (p < 0.025 || !tx[i] || !rx[i]) continue;
        const focus = selected && selected.index !== i ? 0.4 : 1;
        const stale = isSettling ? 0.52 : 1;
        const root = Math.sqrt(p);
        const txOpacity = clamp((0.09 + 0.22 * root) * focus * stale, 0.04, 0.31);
        const rxOpacity = clamp((0.10 + 0.25 * root) * focus * stale, 0.045, 0.35);
        svg.appendChild(svgEl('circle', {
            class: 'd2d-tx-power-glow',
            cx: tx[i].x,
            cy: tx[i].y,
            r: fmt(10.5 + 13.5 * root, 2),
            fill: `rgba(77,163,255,${fmt(txOpacity, 3)})`,
            stroke: 'none',
        }));
        svg.appendChild(svgEl('circle', {
            class: 'd2d-rx-power-glow',
            cx: rx[i].x,
            cy: rx[i].y,
            r: fmt(9.5 + 12.5 * root, 2),
            fill: `rgba(255,106,61,${fmt(rxOpacity, 3)})`,
            stroke: 'none',
        }));
    }
}

export function describeD2DSelection({ selected, tx, rx, power = [], rates = [], losses = [] }) {
    const index = selectedIndex(selected);
    if (index < 0 || !tx?.[index] || !rx?.[index]) {
        return 'Click a node or drag a link endpoint.';
    }

    const kind = selected.kind === 'tx' ? 'TX' : 'RX';
    const directDistance = Math.hypot(tx[index].x - rx[index].x, tx[index].y - rx[index].y);
    const p = clamp(power[index] || 0, 0, 1);
    const rate = Number.isFinite(rates[index]) ? rates[index] : NaN;
    const suffix = strongestLeak({ selected, index, losses, power });
    return `${kind}${index} - link ${index} / p ${fmt(p, 2)} / rate ${fmt(rate, 2)} / d ${fmt(directDistance, 1)} m${suffix}`;
}

function strongestLeak({ selected, index, losses, power }) {
    if (!Array.isArray(losses) || !losses.length) return '';
    const edges = [];
    if (selected.kind === 'rx') {
        for (let source = 0; source < losses.length; source++) {
            if (source === index) continue;
            edges.push({
                label: `top leak T${source}`,
                score: (losses[index]?.[source] || 0) * (power[source] || 0),
            });
        }
    } else {
        for (let target = 0; target < losses.length; target++) {
            if (target === index) continue;
            edges.push({
                label: `leaks to R${target}`,
                score: (losses[target]?.[index] || 0) * (power[index] || 0),
            });
        }
    }
    edges.sort((a, b) => b.score - a.score);
    if (!edges[0] || edges[0].score <= 0) return '';
    return ` / ${edges[0].label}`;
}
