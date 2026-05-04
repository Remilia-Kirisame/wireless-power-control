const SVG_NS = 'http://www.w3.org/2000/svg';
const EPS = 1e-12;

export const JSAC_SIGNAL_ANIMATION_STYLES = /* css */ `
        .jsac-direct-base,
        .jsac-direct-glow,
        .jsac-comm-packet,
        .jsac-comm-arrival,
        .jsac-radar-perimeter,
        .jsac-radar-inner-ring,
        .jsac-radar-sweep,
        .jsac-radar-sector,
        .jsac-radar-echo,
        .jsac-radar-hit,
        .jsac-blue-power-glow,
        .jsac-rx-power-glow,
        .jsac-scan-node-refresh,
        .jsac-interference-edge,
        .jsac-heat-focus-line,
        .jsac-heat-focus-ring {
            pointer-events: none;
        }
        .jsac-direct-base,
        .jsac-direct-glow,
        .jsac-blue-power-glow,
        .jsac-rx-power-glow {
            transition: opacity var(--dur-fast) var(--ease), stroke-width var(--dur-fast) var(--ease), r var(--dur-fast) var(--ease);
        }
        .jsac-comm-packet {
            filter: drop-shadow(0 0 4px rgba(76, 175, 80, 0.34));
            stroke-linecap: round;
        }
        .jsac-comm-arrival {
            filter: drop-shadow(0 0 7px rgba(76, 175, 80, 0.42));
        }
        .jsac-radar-perimeter {
            opacity: var(--radar-opacity, 0.28);
        }
        .jsac-radar-inner-ring {
            opacity: var(--radar-inner-opacity, 0.14);
        }
        .jsac-radar-sweep {
            filter: drop-shadow(0 0 5px rgba(173, 255, 64, 0.25));
        }
        .jsac-radar-sector {
            opacity: var(--sector-opacity, 0.18);
        }
        .jsac-radar-hit,
        .jsac-radar-echo {
            filter: drop-shadow(0 0 5px rgba(246, 196, 69, 0.22));
        }
        .jsac-scan-node-refresh {
            filter: drop-shadow(0 0 5px rgba(246, 196, 69, 0.36));
        }
        .jsac-interference-edge {
            animation: jsacInterferenceFlow 1.65s linear infinite;
            animation-delay: var(--edge-delay, 0s);
            stroke-linecap: round;
        }
        @keyframes jsacInterferenceFlow {
            0% { stroke-dashoffset: 0; opacity: var(--edge-opacity-low, 0.11); }
            42% { opacity: var(--edge-opacity-high, 0.45); }
            100% { stroke-dashoffset: -28; opacity: var(--edge-opacity-low, 0.11); }
        }
        .jsac-heat-focus-line {
            animation: jsacFocusPulse 1.1s ease-in-out infinite;
        }
        .jsac-heat-focus-ring {
            animation: jsacFocusRing 1.1s ease-in-out infinite;
            transform-box: fill-box;
            transform-origin: center;
        }
        @keyframes jsacFocusPulse {
            0%, 100% { opacity: 0.48; }
            50% { opacity: 0.95; }
        }
        @keyframes jsacFocusRing {
            0%, 100% { opacity: 0.45; transform: scale(0.92); }
            50% { opacity: 0.82; transform: scale(1.08); }
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

function phaseValue(now, duration, offset = 0) {
    const seconds = (Number.isFinite(now) ? now : performance.now()) / 1000;
    return ((seconds + offset) / duration) % 1;
}

function polarPoint(cx, cy, radius, degrees) {
    const radians = degrees * Math.PI / 180;
    return {
        x: cx + Math.cos(radians) * radius,
        y: cy + Math.sin(radians) * radius,
    };
}

function sectorPath(cx, cy, radius, startDeg, endDeg) {
    const start = polarPoint(cx, cy, radius, startDeg);
    const end = polarPoint(cx, cy, radius, endDeg);
    const largeArc = Math.abs(endDeg - startDeg) > 180 ? 1 : 0;
    return `M ${fmt(cx, 2)} ${fmt(cy, 2)} L ${fmt(start.x, 2)} ${fmt(start.y, 2)} A ${fmt(radius, 2)} ${fmt(radius, 2)} 0 ${largeArc} 1 ${fmt(end.x, 2)} ${fmt(end.y, 2)} Z`;
}

function arcPath(cx, cy, radius, startDeg, endDeg) {
    const start = polarPoint(cx, cy, radius, startDeg);
    const end = polarPoint(cx, cy, radius, endDeg);
    const largeArc = Math.abs(endDeg - startDeg) > 180 ? 1 : 0;
    return `M ${fmt(start.x, 2)} ${fmt(start.y, 2)} A ${fmt(radius, 2)} ${fmt(radius, 2)} 0 ${largeArc} 1 ${fmt(end.x, 2)} ${fmt(end.y, 2)}`;
}

function radarPhase(now, duration, group) {
    const seconds = (Number.isFinite(now) ? now : performance.now()) / 1000;
    return ((seconds / duration) + group * 0.17) % 1;
}

function radarAge(now, duration, group, angleNorm) {
    const phase = radarPhase(now, duration, group);
    return ((phase - angleNorm + 1) % 1) * duration;
}

function smoothPulse(progress, center, width) {
    const d = Math.abs(progress - center);
    if (d >= width) return 0;
    const x = 1 - d / width;
    return x * x * (3 - 2 * x);
}

function radarDurationFromUtil(util) {
    const root = Math.sqrt(clamp(util, 0, 1));
    return clamp(8.65 - 1.35 * root, 6.8, 9.2);
}

function groupFocus(focusGroup, group, dim = 0.24) {
    if (!Number.isFinite(focusGroup)) return 1;
    return focusGroup === group ? 1 : dim;
}

function linkPoint(blue, rx) {
    if (!rx || !blue?.[rx.blue]) return null;
    return blue[rx.blue];
}

function groupUtilTotal(options, group) {
    const total = options.method?.groupUtil?.[group]?.total;
    if (Number.isFinite(total)) return clamp(total, 0, 1);
    let sum = 0;
    for (let i = 0; i < options.meta.k; i++) {
        if (options.meta.groupIds[i] === group) sum += options.power[i] || 0;
    }
    return clamp(sum, 0, 1);
}

function maxGreenRate(method, meta) {
    if (!Array.isArray(method?.rates)) return 0.1;
    let max = 0.1;
    for (let i = 0; i < meta.k; i++) {
        if (meta.greenMask[i]) max = Math.max(max, method.rates[i] || 0);
    }
    return max;
}

function greenPowerStats(meta, rx, power, group) {
    let max = 0;
    let total = 0;
    let count = 0;
    for (let i = 0; i < meta.k; i++) {
        if (!meta.greenMask[i] || rx[i]?.blue !== group) continue;
        const p = clamp(power[i] || 0, 0, 1);
        max = Math.max(max, p);
        total += p;
        count++;
    }
    return { max, total, count };
}

function greenActivity(state, index) {
    const p = clamp(state.power[index] || 0, 0, 1);
    if (p <= 0.002) return 0;
    const target = state.rx[index];
    const stats = greenPowerStats(state.meta, state.rx, state.power, target.blue);
    const local = stats.max > EPS ? p / stats.max : 0;
    const share = stats.total > EPS && stats.count ? p * stats.count / stats.total : local;
    return clamp(0.36 + 0.42 * Math.sqrt(local) + 0.22 * Math.sqrt(clamp(share, 0, 1.4) / 1.4), 0.36, 1);
}

function appendDirectBases(svg, state) {
    const { blue, rx, meta, power, focusGroup, isSettling } = state;
    for (let i = 0; i < meta.k; i++) {
        const target = rx[i];
        const source = linkPoint(blue, target);
        if (!source || !target) continue;

        const isYellow = meta.yellowMask[i];
        const p = clamp(power[i] || 0, 0, 1);
        const root = Math.sqrt(p);
        const focus = groupFocus(focusGroup, target.blue);
        const stale = isSettling ? 0.58 : 1;
        const rgb = isYellow ? '246,196,69' : '76,175,80';
        const baseOpacity = isYellow ? 0.12 + 0.42 * root : 0.16 + 0.58 * root;
        const opacity = clamp(baseOpacity * focus * stale, 0.04, isYellow ? 0.58 : 0.82);
        const width = (isYellow ? 0.44 : 0.54) + (isYellow ? 1.0 : 1.35) * root;

        if (p > 0.035 && !isYellow) {
            svg.appendChild(svgEl('line', {
                class: 'jsac-direct-glow',
                x1: source.x,
                y1: source.y,
                x2: target.x,
                y2: target.y,
                stroke: `rgba(${rgb},${fmt(opacity * 0.2, 3)})`,
                'stroke-width': fmt(width + 3.4, 2),
                'stroke-linecap': 'round',
            }));
        }

        svg.appendChild(svgEl('line', {
            class: 'jsac-direct-base',
            x1: source.x,
            y1: source.y,
            x2: target.x,
            y2: target.y,
            stroke: `rgba(${rgb},${fmt(opacity, 3)})`,
            'stroke-width': fmt(width, 2),
            'stroke-linecap': 'round',
            'stroke-dasharray': isYellow ? '1.2 2.2' : null,
        }));
    }
}

function appendCommunicationPackets(svg, state) {
    const { blue, rx, meta, power, method, focusGroup, isSettling, reduceMotion, now } = state;
    if (reduceMotion || isSettling) return;

    const maxRate = maxGreenRate(method, meta);
    for (let i = 0; i < meta.k; i++) {
        if (!meta.greenMask[i]) continue;
        const target = rx[i];
        const source = linkPoint(blue, target);
        if (!source || !target) continue;

        const focus = groupFocus(focusGroup, target.blue, 0.18);
        const activity = greenActivity(state, i);
        if (activity <= 0 || focus < 0.2) continue;

        const rateNorm = clamp((method?.rates?.[i] || 0) / (maxRate + EPS), 0, 1);
        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const distance = Math.hypot(dx, dy);
        if (distance < EPS) continue;
        const ux = dx / distance;
        const uy = dy / distance;
        const duration = clamp(2.86 - 0.34 * activity - 0.12 * rateNorm + distance / 1450, 1.78, 3.35);
        const packetLen = clamp(1.05 + 1.7 * activity + 0.45 * rateNorm, 1.05, 3.2);
        const opacity = clamp((0.2 + 0.22 * activity + 0.06 * rateNorm) * focus, 0.1, 0.46);
        const width = 0.42 + 0.45 * activity;
        const count = activity >= 0.82 && distance > 24 ? 2 : 1;

        for (let n = 0; n < count; n++) {
            const progress = phaseValue(now, duration, i * 0.17 + n * duration / count);
            const travelFade = clamp(Math.min(progress / 0.14, (1 - progress) / 0.1), 0, 1);
            if (travelFade <= 0) continue;
            const cx = source.x + dx * progress;
            const cy = source.y + dy * progress;
            const half = packetLen * 0.5;
            svg.appendChild(svgEl('line', {
                class: 'jsac-comm-packet',
                x1: fmt(cx - ux * half, 2),
                y1: fmt(cy - uy * half, 2),
                x2: fmt(cx + ux * half, 2),
                y2: fmt(cy + uy * half, 2),
                stroke: `rgba(112,222,124,${fmt(opacity * travelFade, 3)})`,
                'stroke-width': fmt(width, 2),
            }));
            const arrival = smoothPulse(progress, 0.94, 0.22);
            if (arrival <= 0) continue;
            const pulseOpacity = clamp((0.05 + 0.105 * activity + 0.03 * rateNorm) * focus * arrival, 0, 0.2);
            svg.appendChild(svgEl('circle', {
                class: 'jsac-comm-arrival',
                cx: target.x,
                cy: target.y,
                r: fmt(2.25 + 1.35 * arrival + 0.45 * activity, 2),
                fill: `rgba(112,222,124,${fmt(pulseOpacity * 0.45, 3)})`,
                stroke: 'none',
            }));
            svg.appendChild(svgEl('circle', {
                class: 'jsac-comm-arrival',
                cx: target.x,
                cy: target.y,
                r: fmt(1.55 + 0.84 * arrival + 0.28 * activity, 2),
                fill: `rgba(112,222,124,${fmt(pulseOpacity, 3)})`,
                stroke: 'none',
            }));
        }
    }
}

function yellowLinksForGroup(meta, rx, group) {
    const links = [];
    for (let i = 0; i < meta.k; i++) {
        if (meta.yellowMask[i] && rx[i]?.blue === group) links.push(i);
    }
    return links;
}

function radarRadiusForGroup(blue, rx, links, field) {
    let radius = 18;
    for (const i of links) {
        const target = rx[i];
        if (!target) continue;
        radius = Math.max(radius, Math.hypot(target.x - blue.x, target.y - blue.y) + 7);
    }
    return clamp(radius, 16, Math.max(24, field * 0.22));
}

function appendRadarSweeps(svg, state) {
    const { blue, rx, meta, field, focusGroup, isSettling, reduceMotion, now } = state;
    if (!Number.isFinite(focusGroup)) return;
    for (let b = 0; b < blue.length; b++) {
        if (b !== focusGroup) continue;
        const source = blue[b];
        const links = yellowLinksForGroup(meta, rx, b);
        if (!source || !links.length) continue;

        const util = groupUtilTotal(state, b);
        const root = Math.sqrt(util);
        const focus = groupFocus(focusGroup, b, 0.18);
        const radius = radarRadiusForGroup(source, rx, links, field);
        const duration = radarDurationFromUtil(util);
        const opacity = clamp((0.2 + 0.4 * root) * focus * (isSettling ? 0.55 : 1), 0.045, 0.68);
        const phase = radarPhase(now, duration, b);
        const start = phase * 360;

        svg.appendChild(svgEl('circle', {
            class: 'jsac-radar-perimeter',
            cx: source.x,
            cy: source.y,
            r: fmt(radius, 2),
            fill: 'none',
            stroke: `rgba(246,196,69,${fmt(opacity * 0.72, 3)})`,
            'stroke-width': 0.58,
            'stroke-dasharray': '2.4 3.6',
            'stroke-dashoffset': fmt(-phase * 18, 2),
            style: `--radar-opacity:${fmt(opacity, 3)};--radar-opacity-low:${fmt(opacity * 0.42, 3)};--radar-duration:${fmt(duration, 3)}s;`,
        }));
        svg.appendChild(svgEl('circle', {
            class: 'jsac-radar-inner-ring',
            cx: source.x,
            cy: source.y,
            r: fmt(radius * 0.56, 2),
            fill: 'none',
            stroke: `rgba(246,196,69,${fmt(opacity * 0.36, 3)})`,
            'stroke-width': 0.42,
            'stroke-dasharray': '1.4 4.8',
            'stroke-dashoffset': fmt(phase * 16, 2),
            style: `--radar-opacity:${fmt(opacity * 0.5, 3)};--radar-opacity-low:${fmt(opacity * 0.2, 3)};--radar-inner-opacity:${fmt(opacity * 0.34, 3)};--radar-duration:${fmt(duration, 3)}s;`,
        }));

        if (reduceMotion || isSettling || focus < 0.2 || util < 0.025) continue;

        const sweep = svgEl('g', {
            class: 'jsac-radar-sweep',
            style: `--radar-opacity:${fmt(opacity, 3)};--radar-duration:${fmt(duration, 3)}s;`,
        });

        [
            [-118, -78, 0.046],
            [-96, -52, 0.09],
            [-74, -24, 0.158],
            [-50, 10, 0.25],
        ].forEach(([a0, a1, alpha]) => {
            sweep.appendChild(svgEl('path', {
                class: 'jsac-radar-sector',
                d: sectorPath(source.x, source.y, radius, start + a0, start + a1),
                fill: `rgba(196,255,86,${fmt(opacity * alpha * 2.35, 3)})`,
                stroke: 'none',
                style: `--sector-opacity:${fmt(opacity * alpha, 3)};`,
            }));
        });
        svg.appendChild(sweep);
    }
}

function appendRadarEchoes(svg, state) {
    const { blue, rx, meta, power, method, focusGroup, isSettling, reduceMotion, now, sinrMin } = state;
    if (reduceMotion || isSettling) return;
    if (!Number.isFinite(focusGroup)) return;

    for (let i = 0; i < meta.k; i++) {
        if (!meta.yellowMask[i]) continue;
        const target = rx[i];
        if (target?.blue !== focusGroup) continue;
        const source = linkPoint(blue, target);
        if (!source || !target) continue;

        const p = clamp(power[i] || 0, 0, 1);
        const focus = groupFocus(focusGroup, target.blue, 0.18);
        if (focus < 0.2 || p < 0.018) continue;

        const util = groupUtilTotal(state, target.blue);
        const duration = radarDurationFromUtil(util);
        const angle = Math.atan2(target.y - source.y, target.x - source.x);
        const angleNorm = (angle + Math.PI * 2) % (Math.PI * 2) / (Math.PI * 2);
        const age = radarAge(now, duration, target.blue, angleNorm);
        const sinr = method?.sinr?.[i];
        const alert = Number.isFinite(sinr) && Number.isFinite(sinrMin) && sinr < sinrMin;
        const freshness = Math.exp(-age / 0.45);
        const echoGrowth = clamp(age / 0.85, 0, 1);
        const blink = smoothPulse(age / duration, 0, 0.064);
        const opacity = clamp((0.3 + 0.32 * Math.sqrt(p)) * focus * (0.22 * freshness + 0.78 * blink), 0, alert ? 0.62 : 0.54);
        const hitOpacity = clamp((0.2 + 0.28 * Math.sqrt(p)) * focus * Math.max(freshness, blink), 0, alert ? 0.54 : 0.44);
        if (hitOpacity <= 0.01 && opacity <= 0.01) continue;

        svg.appendChild(svgEl('circle', {
            class: 'jsac-radar-hit',
            cx: target.x,
            cy: target.y,
            r: fmt(2.7 + 2.25 * Math.sqrt(p) + 1.35 * blink, 2),
            fill: alert ? `rgba(255,99,99,${fmt(hitOpacity, 3)})` : `rgba(246,196,69,${fmt(hitOpacity, 3)})`,
            stroke: 'none',
        }));

        svg.appendChild(svgEl('circle', {
            class: `jsac-radar-echo${alert ? ' is-alert' : ''}`,
            cx: target.x,
            cy: target.y,
            r: fmt(3.35 + 6.0 * echoGrowth + 2.45 * Math.sqrt(p), 2),
            fill: 'none',
            stroke: alert ? `rgba(255,99,99,${fmt(opacity, 3)})` : `rgba(246,196,69,${fmt(opacity, 3)})`,
            'stroke-width': alert ? 1.15 : 0.92,
        }));
    }
}

export function appendJSACScanNodeOverlays(svg, options) {
    const state = {
        ...options,
        blue: options.blue || [],
        rx: options.rx || [],
        meta: options.meta || { k: 0, yellowMask: [] },
        power: options.power || [],
        method: options.method || null,
        focusGroup: Number.isFinite(options.focusGroup) ? options.focusGroup : null,
        reduceMotion: Boolean(options.reduceMotion),
        isSettling: Boolean(options.isSettling),
        now: Number.isFinite(options.now) ? options.now : performance.now(),
    };
    if (state.reduceMotion || state.isSettling) return;
    if (!Number.isFinite(state.focusGroup)) return;
    for (let i = 0; i < state.meta.k; i++) {
        if (!state.meta.yellowMask[i]) continue;
        const target = state.rx[i];
        if (target?.blue !== state.focusGroup) continue;
        const source = linkPoint(state.blue, target);
        if (!source || !target) continue;
        const focus = groupFocus(state.focusGroup, target.blue, 0.18);
        if (focus < 0.2) continue;
        const p = clamp(state.power[i] || 0, 0, 1);
        const util = groupUtilTotal(state, target.blue);
        const duration = radarDurationFromUtil(util);
        const angle = Math.atan2(target.y - source.y, target.x - source.x);
        const angleNorm = (angle + Math.PI * 2) % (Math.PI * 2) / (Math.PI * 2);
        const age = radarAge(state.now, duration, target.blue, angleNorm);
        const freshness = Math.exp(-age / 0.52);
        const blink = smoothPulse(age / duration, 0, 0.064);
        const alpha = clamp((0.48 * freshness + 0.42 * blink) * focus, 0, 0.74);
        if (alpha <= 0.02) continue;
        svg.appendChild(svgEl('circle', {
            class: 'jsac-scan-node-refresh',
            cx: target.x,
            cy: target.y,
            r: fmt(2.55 + 1.15 * Math.sqrt(p) + 0.75 * blink, 2),
            fill: `rgba(255,232,120,${fmt(alpha, 3)})`,
            stroke: 'none',
        }));
    }
}

export function appendJSACSignalLayers(svg, options) {
    const state = {
        ...options,
        blue: options.blue || [],
        rx: options.rx || [],
        meta: options.meta || { k: 0, greenMask: [], yellowMask: [], groupIds: [] },
        power: options.power || [],
        method: options.method || null,
        focusGroup: Number.isFinite(options.focusGroup) ? options.focusGroup : null,
        field: Number.isFinite(options.field) ? options.field : 225,
        reduceMotion: Boolean(options.reduceMotion),
        isSettling: Boolean(options.isSettling),
        now: Number.isFinite(options.now) ? options.now : performance.now(),
    };
    appendDirectBases(svg, state);
    appendCommunicationPackets(svg, state);
    appendRadarSweeps(svg, state);
    appendRadarEchoes(svg, state);
}

export function appendJSACNodeEnergyGlows(svg, options) {
    const state = {
        ...options,
        blue: options.blue || [],
        rx: options.rx || [],
        meta: options.meta || { k: 0, greenMask: [], yellowMask: [], groupIds: [] },
        power: options.power || [],
        method: options.method || null,
        focusGroup: Number.isFinite(options.focusGroup) ? options.focusGroup : null,
        isSettling: Boolean(options.isSettling),
    };

    for (let b = 0; b < state.blue.length; b++) {
        const source = state.blue[b];
        if (!source) continue;
        const util = groupUtilTotal(state, b);
        const focus = groupFocus(state.focusGroup, b);
        const root = Math.sqrt(util);
        svg.appendChild(svgEl('circle', {
            class: 'jsac-blue-power-glow',
            cx: source.x,
            cy: source.y,
            r: fmt(3.5 + 3.6 * root, 2),
            fill: `rgba(77,163,255,${fmt((0.055 + util * 0.12) * focus * (state.isSettling ? 0.55 : 1), 3)})`,
            stroke: 'none',
        }));
    }

    for (let i = 0; i < state.meta.k; i++) {
        const target = state.rx[i];
        if (!target) continue;
        const p = clamp(state.power[i] || 0, 0, 1);
        if (p < 0.055) continue;
        const focus = groupFocus(state.focusGroup, target.blue);
        const isYellow = state.meta.yellowMask[i];
        const root = Math.sqrt(p);
        const rgb = isYellow ? '246,196,69' : '76,175,80';
        svg.appendChild(svgEl('circle', {
            class: 'jsac-rx-power-glow',
            cx: target.x,
            cy: target.y,
            r: fmt(3.05 + 3.85 * root, 2),
            fill: `rgba(${rgb},${fmt((0.085 + 0.15 * root) * focus * (state.isSettling ? 0.58 : 1), 3)})`,
            stroke: 'none',
        }));
    }
}

export function appendJSACInterferenceLayers(svg, options) {
    const state = {
        ...options,
        blue: options.blue || [],
        rx: options.rx || [],
        meta: options.meta || { k: 0, groupIds: [], interfMask: [] },
        power: options.power || [],
        losses: options.losses || null,
        focusGroup: Number.isFinite(options.focusGroup) ? options.focusGroup : null,
        reduceMotion: Boolean(options.reduceMotion),
        isSettling: Boolean(options.isSettling),
        now: Number.isFinite(options.now) ? options.now : performance.now(),
    };
    const { blue, rx, meta, power, losses, focusGroup } = state;
    if (!losses) return;

    const edges = [];
    for (let target = 0; target < meta.k; target++) {
        for (let source = 0; source < meta.k; source++) {
            if (!meta.interfMask?.[target]?.[source]) continue;
            edges.push({ target, source, score: (losses[target]?.[source] || 0) * (power[source] || 0) });
        }
    }
    edges.sort((a, b) => b.score - a.score);
    const maxScore = edges[0]?.score || 1;
    const maxEdges = Number.isFinite(focusGroup) ? 22 : 76;
    edges.slice(0, Math.min(maxEdges, edges.length)).forEach((edge, n) => {
        const sourceGroup = meta.groupIds[edge.source];
        const targetGroup = rx[edge.target]?.blue;
        const source = blue[sourceGroup];
        const target = rx[edge.target];
        if (!source || !target) return;
        const related = !Number.isFinite(focusGroup) || sourceGroup === focusGroup || targetGroup === focusGroup;
        const normalized = Math.sqrt(edge.score / (maxScore + EPS));
        const high = clamp(0.08 + (Number.isFinite(focusGroup) ? 0.56 : 0.36) * normalized, 0.08, Number.isFinite(focusGroup) ? 0.72 : 0.45) * (related ? 1 : 0.2) * (state.isSettling ? 0.48 : 1);
        const low = clamp(high * 0.32, 0.025, 0.24);
        const delay = state.reduceMotion ? 0 : phaseDelay(state.now, 1.65, n * 0.075);
        svg.appendChild(svgEl('line', {
            class: 'jsac-interference-edge',
            x1: source.x,
            y1: source.y,
            x2: target.x,
            y2: target.y,
            stroke: `rgba(126,170,210,${fmt(high, 3)})`,
            'stroke-width': fmt(0.5 + 1.05 * normalized, 2),
            'stroke-dasharray': '3.4 6.4',
            style: `--edge-delay:${fmt(delay, 3)}s;--edge-opacity-low:${fmt(low, 3)};--edge-opacity-high:${fmt(high, 3)};`,
        }));
    });
}

export function appendJSACHeatFocus(svg, options) {
    const blue = options.blue || [];
    const rx = options.rx || [];
    const meta = options.meta || { groupIds: [] };
    const hoverEdge = options.hoverEdge;
    if (!hoverEdge || meta.groupIds[hoverEdge.source] === undefined) return;
    const source = blue[meta.groupIds[hoverEdge.source]];
    const target = rx[hoverEdge.target];
    if (!source || !target) return;
    svg.appendChild(svgEl('line', {
        class: 'jsac-heat-focus-line',
        x1: source.x,
        y1: source.y,
        x2: target.x,
        y2: target.y,
        stroke: 'rgba(255,255,255,0.92)',
        'stroke-width': 1.35,
        'stroke-dasharray': '3 2',
    }));
    svg.appendChild(svgEl('circle', {
        class: 'jsac-heat-focus-ring',
        cx: target.x,
        cy: target.y,
        r: 6.8,
        fill: 'none',
        stroke: 'rgba(255,255,255,0.75)',
        'stroke-width': 1.1,
    }));
}
