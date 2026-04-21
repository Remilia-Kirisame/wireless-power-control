// sandbox/script.js
// Two tiny SVG demos for the "clean academic poster" sandbox:
//   1. A schematic JSAC layout (3 Blue clusters + interference edges).
//   2. A bar chart comparing Equal / WMMSE / GNN Green sum-rate across B.
//
// Everything is hard-coded — the real site will read JSON dumps of
// results_sweep_B.pkl produced by Scenario_JSAC/main.py.

"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";

const COLORS = {
    blue:   "#2196f3",
    orange: "#ff5722",
    yellow: "#f6c445",
    green:  "#4caf50",
    grey:   "#888888",
    ink:    "#1c1c1c",
    rule:   "#d9d4c8",
    faint:  "#8a8a8a",
};

/* ---------- small helpers ---------- */

function el(tag, attrs = {}, parent = null) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
        node.setAttribute(k, v);
    }
    if (parent) parent.appendChild(node);
    return node;
}

function text(parent, x, y, str, opts = {}) {
    const t = el("text", {
        x, y,
        "text-anchor": opts.anchor || "start",
        "dominant-baseline": opts.baseline || "alphabetic",
        "font-family": opts.family || "Charter, Georgia, serif",
        "font-size": opts.size || 12,
        "font-style": opts.italic ? "italic" : "normal",
        fill: opts.fill || COLORS.ink,
    }, parent);
    t.textContent = str;
    return t;
}

/* ---------- Fig 1 — JSAC layout schematic ---------- */

function drawLayout() {
    const svg = document.getElementById("layout-svg");
    if (!svg) return;

    // Three Blue clusters. Hand-placed so the figure reads left-to-right.
    const clusters = [
        { cx:  95, cy: 170, yellows: [[-26, -30], [ 18, -38]], greens: [[-30,  28], [ 22,  24], [ -4,  48]] },
        { cx: 240, cy: 110, yellows: [[-22, -26], [ 28, -14]], greens: [[-30,  20], [  8,  38], [ 32,  28]] },
        { cx: 370, cy: 200, yellows: [[-24,  28], [ 26, -28]], greens: [[-32, -18], [ 14,  34], [ 30,   2]] },
    ];

    // Pale backdrop.
    el("rect", { x: 0, y: 0, width: 480, height: 320, fill: "transparent" }, svg);

    // Inter-cluster same-channel interference (dashed).
    const pairs = [[0, 1], [1, 2], [0, 2]];
    for (const [a, b] of pairs) {
        el("line", {
            x1: clusters[a].cx, y1: clusters[a].cy,
            x2: clusters[b].cx, y2: clusters[b].cy,
            stroke: COLORS.faint,
            "stroke-width": 1,
            "stroke-dasharray": "4 4",
            opacity: 0.7,
        }, svg);
    }

    // Each cluster: draw intra-group link lines first (so dots sit on top).
    for (const c of clusters) {
        for (const [dx, dy] of [...c.yellows, ...c.greens]) {
            el("line", {
                x1: c.cx, y1: c.cy,
                x2: c.cx + dx, y2: c.cy + dy,
                stroke: COLORS.ink, opacity: 0.18, "stroke-width": 1,
            }, svg);
        }
    }

    // Rx dots (yellow rings for sensing, green rings for comm).
    for (const c of clusters) {
        for (const [dx, dy] of c.yellows) {
            el("circle", {
                cx: c.cx + dx, cy: c.cy + dy, r: 7,
                fill: "#fff", stroke: COLORS.yellow, "stroke-width": 2.5,
            }, svg);
        }
        for (const [dx, dy] of c.greens) {
            el("circle", {
                cx: c.cx + dx, cy: c.cy + dy, r: 7,
                fill: "#fff", stroke: COLORS.green, "stroke-width": 2.5,
            }, svg);
        }
    }

    // Tx squares on top.
    for (const c of clusters) {
        el("rect", {
            x: c.cx - 8, y: c.cy - 8, width: 16, height: 16,
            fill: COLORS.blue, stroke: COLORS.ink, "stroke-width": 1,
        }, svg);
    }

    // Labels.
    clusters.forEach((c, i) => {
        text(svg, c.cx, c.cy - 18, `Blue ${i + 1}`, {
            anchor: "middle", size: 11, family: "JetBrains Mono, monospace", fill: COLORS.faint,
        });
    });

    // Tiny inline legend (bottom-left).
    const lx = 16, ly = 298;
    el("rect", { x: lx,        y: ly - 8, width: 10, height: 10, fill: COLORS.blue }, svg);
    el("circle", { cx: lx + 68, cy: ly - 3, r: 5, fill: "#fff", stroke: COLORS.yellow, "stroke-width": 2 }, svg);
    el("circle", { cx: lx + 148, cy: ly - 3, r: 5, fill: "#fff", stroke: COLORS.green, "stroke-width": 2 }, svg);
    text(svg, lx + 14, ly,  "Blue (Tx)",   { size: 11, fill: COLORS.ink });
    text(svg, lx + 78, ly,  "Yellow (sense)", { size: 11, fill: COLORS.ink });
    text(svg, lx + 158, ly, "Green (comm)",   { size: 11, fill: COLORS.ink });

    text(svg, 464, 298, "— — same-channel interference", {
        anchor: "end", size: 11, italic: true, fill: COLORS.faint,
    });
}

/* ---------- Fig 2 — grouped bar chart ---------- */

// Illustrative values only. Shape matches results_sweep_B.pkl output.
const BAR_DATA = {
    B_vals: [3, 5, 7, 10],
    series: [
        { name: "Equal Power", color: COLORS.grey,   values: [ 42,  68,  91, 128] },
        { name: "WMMSE",       color: COLORS.blue,   values: [ 55,  88, 118, 165] },
        { name: "GNN",         color: COLORS.orange, values: [ 52,  84, 113, 158] },
    ],
};

function drawBars() {
    const svg = document.getElementById("bars-svg");
    if (!svg) return;

    const W = 480, H = 260;
    const pad = { top: 22, right: 16, bottom: 44, left: 44 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const maxY = 180;
    const yTicks = [0, 45, 90, 135, 180];

    // Y-axis gridlines + labels.
    for (const t of yTicks) {
        const y = pad.top + plotH * (1 - t / maxY);
        el("line", {
            x1: pad.left, x2: pad.left + plotW, y1: y, y2: y,
            stroke: COLORS.rule, "stroke-width": 1,
        }, svg);
        text(svg, pad.left - 8, y + 4, String(t), {
            anchor: "end", size: 11, family: "JetBrains Mono, monospace", fill: COLORS.faint,
        });
    }

    // Y-axis title.
    const yTitleX = 14, yTitleY = pad.top + plotH / 2;
    const yTitle = text(svg, yTitleX, yTitleY, "Green sum-rate (bits/s/Hz)", {
        anchor: "middle", size: 12, italic: true, fill: COLORS.ink,
    });
    yTitle.setAttribute("transform", `rotate(-90 ${yTitleX} ${yTitleY})`);

    // Grouped bars.
    const nGroups = BAR_DATA.B_vals.length;
    const nSeries = BAR_DATA.series.length;
    const groupW = plotW / nGroups;
    const barW = Math.min(22, (groupW - 16) / nSeries);
    const innerPad = (groupW - barW * nSeries) / 2;

    BAR_DATA.B_vals.forEach((bVal, gi) => {
        const gx = pad.left + gi * groupW;

        BAR_DATA.series.forEach((s, si) => {
            const v = s.values[gi];
            const h = plotH * (v / maxY);
            const x = gx + innerPad + si * barW;
            const y = pad.top + plotH - h;
            el("rect", {
                x, y, width: barW, height: h,
                fill: s.color, opacity: 0.92,
            }, svg);

            text(svg, x + barW / 2, y - 4, String(v), {
                anchor: "middle", size: 10, family: "JetBrains Mono, monospace", fill: COLORS.faint,
            });
        });

        // X tick label.
        text(svg, gx + groupW / 2, pad.top + plotH + 18, `B = ${bVal}`, {
            anchor: "middle", size: 12, family: "JetBrains Mono, monospace", fill: COLORS.ink,
        });
    });

    // X-axis line.
    el("line", {
        x1: pad.left, x2: pad.left + plotW,
        y1: pad.top + plotH, y2: pad.top + plotH,
        stroke: COLORS.ink, "stroke-width": 1,
    }, svg);

    // X-axis title.
    text(svg, pad.left + plotW / 2, H - 8,
         "Number of Blue cars", {
        anchor: "middle", size: 12, italic: true, fill: COLORS.ink,
    });
}

/* ---------- boot ---------- */

document.addEventListener("DOMContentLoaded", () => {
    drawLayout();
    drawBars();
});
