# Post-build feature specs

Add-on features applied after the initial website build. Each level-2 entry below is a self-contained component spec — append a new entry per feature request.

---

## WMMSE iteration visualizer — `<wmmse-iter>`

A Shadow-DOM Web Component that animates iterative WMMSE on a small interference channel — three coordinated views of the same algorithm state plus transport controls — so the reader feels the cost of "iterative" before reading any pitch about a faster method.

File: `components/wmmse-iter/wmmse-iter.js`. Same conventions as the other components: ES-module class extending `HTMLElement`, Shadow DOM, design tokens (`--bg`, `--surface`, `--text`, `--text-dim`, `--text-mute`, `--rule`, `--c-orange`, `--font-sans`, `--font-mono`, `--radius-card`, `--ease`, `--dur-fast`) pierce in via the host stylesheet.

### Behavior — three coordinated views

All three update on the same heartbeat. State = the current `(K, seed)` topology + iteration index.

- **Layout panel** (≈ 380 × 260 SVG, `preserveAspectRatio="xMidYMid meet"`). K Tx/Rx pairs on a 2D field. Tx are filled orange squares (~6px); Rx are filled orange circles (~4.5px). Direct link Tx<sub>i</sub> → Rx<sub>i</sub> as a thin orange line; **opacity of the line, Tx, and Rx all encode the link's current normalized power** as `0.15 + 0.85·(p/Pmax)`. Interference: for each Rx, draw the **top 2** strongest incoming Tx-from-other-link as faint dashed grey lines (static — independent of current power). Field boundary as a faint dashed rect. An invisible thick stroke along each Tx→Rx line acts as the hover hit area.

- **Convergence chart** (≈ 380 × 260 SVG). Linear x: `0..MAX_ITER`. Linear y: `[min(initial,target)−ε, max(initial,target)+ε]` with ε ≈ 6% of |target|. **Dashed faint horizontal at the converged sum-rate** with a small `target X.XX` label anchored to the right edge. Faint baseline horizontal at the iter-0 sum-rate. Trajectory rendered as an orange polyline up to the current iter, with a soft blurred halo polyline behind it for a quiet glow. Orange dot marks the head. 4 y-gridlines + ticks; x-ticks at `[0, 25, 50, 75, 100]`. Mono axis labels.

- **Power-bar strip** (full-width SVG, ~92px tall). K vertical bars on a baseline rule. Bar height = `(p/Pmax)·trackHeight`; bar opacity = `0.25 + 0.75·(p/Pmax)`. Faint full-height slot behind each bar. Mono `1..K` index labels below. **`ResizeObserver` re-renders on host width changes** so labels stay legible across viewports — `preserveAspectRatio="none"` would stretch the text. The strip's `viewBox` is set to live `clientWidth` on each render.

Below the views: a transport row and a four-row mono readout panel.

### Algorithm — single-topology WMMSE port

Direct JS port of `batch_WMMSE2` (no batch dimension). Per outer iteration:

1. **Update precoder** $b \in \mathbb{R}^K$:
$$
b_{up}[j] = \alpha[j]\,w[j]\,H[j,j]\,f[j], \quad
b_{down}[j] = \sum_i \alpha[i]\,w[i]\,(H[i,j]\,f[i])^2, \quad
b[j] = \mathrm{clip}\!\left(\frac{b_{up}[j]}{b_{down}[j]},\ 0,\ \sqrt{P_{max}}\right)
$$

2. **Update receiver filter and MMSE weight** $f, w \in \mathbb{R}^K$:
$$
\mathrm{interference}[i] = \sum_j (H[i,j]\,b[j])^2 + \sigma^2 \quad\text{(includes self-power: total received + noise)}
$$
$$
f[i] = \frac{H[i,i]\,b[i]}{\mathrm{interference}[i]}, \qquad
w[i] = \frac{1}{1 - f[i]\,H[i,i]\,b[i]}
$$

Initial state: `b[i] = √Pmax` (every link transmits at full power), then run the f/w update once. Per-iteration power vector: `p[i] = b[i]²`. Use `α[i] = 1`, `Pmax = 1`, `var_noise = 1` (normalized; tuned with the channel sampling so mid-K SINRs land in a visually interesting range). Use a small `EPS = 1e-12` in every denominator.

Sum-rate, given `p` and `H`:
$$
\mathrm{SR} = \sum_{i} \log_2\!\left(1 + \frac{p[i]\,H[i,i]^2}{\sum_{j\ne i} p[j]\,H[i,j]^2 + \sigma^2}\right)
$$

When `(K, seed)` changes: **silently run the full `MAX_ITER` trajectory once** to compute (a) the converged target value for the dashed asymptote, and (b) the per-iter sum-rate sequence the curve will replay. Then reset to iter 0. The animation reuses these pre-computed values for the curve so per-frame cost is constant.

### Channel sampling — seeded JS, no JSON

Channels are generated client-side so the K dropdown and seed shuffle work without any preloaded data.

- **PRNG**: `mulberry32` seeded by `seed * 0x9E3779B9 + K * 0x85EBCA6B` so neighboring `(K, seed)` pairs give visibly different layouts.
- **Layout**: K Tx placed uniformly in `[50, FIELD_SIZE − 50]²` with `FIELD_SIZE = 1000` m. Each Rx<sub>i</sub> placed at radius `30..80` m around its Tx<sub>i</sub> at uniform random angle (Tx-Rx pairs stay close; interferers come from other Tx far away).
- **Channel magnitude $H[i,j]$** (Tx j → Rx i):
  - Distance $d = \max(1,\ \|Tx_j - Rx_i\|)$.
  - Path loss gain $(d_0/d)^\gamma$ with $d_0 = 200$ m, $\gamma = 3$.
  - Optional log-normal shadowing $10^{N(0,\sigma^2)/10}$ with `SHADOW_DB = 0` (disabled by default for clarity; component-level `const`).
  - Rayleigh fading $|h|^2 = 0.5\,(X^2 + Y^2)$ with $X, Y \sim N(0,1)$ (Box-Muller).
  - $H[i,j] = \sqrt{\,pl \cdot \text{shadow} \cdot \text{fading}\,}$.

### Controls

- **▶ Play / ⏸ Pause** — toggles autoplay. If `iter == MAX_ITER`, Play first auto-resets so the curve animates from the start instead of being stuck at the asymptote.
- **⏭ Step** — advances exactly one iteration; pauses playback if running.
- **↺ Reset** — back to iter 0 with the current `(K, seed)` topology unchanged.
- **Speed slider** — log-spaced multipliers `[0.25, 0.5, 1, 2, 4, 8]×`. Default `1×`. Speed label updates live.
- **K dropdown** — `{5, 8, 10, 15, 20}`. Default `8`. Changing K resamples channels at the current seed and resets.
- **Seed `↻` button** — increments seed by 1 and resamples (same K, fresh layout).
- All buttons keyboard-operable; `aria-label` on icon-only ones.

### Pacing

At speed = 1×, one WMMSE iteration plays per **`BASE_MS_PER_ITER = 90 ms`** of wall time (chosen so early iterations are dramatic and a full 100-iter run takes ~9 s). RAF loop with an accumulator: each frame advances `dt · speed / BASE_MS_PER_ITER` "iterations of debt"; while debt ≥ 1 and `iter < MAX_ITER`, take one WMMSE step and decrement.

### Wall-clock readout — calibrated, decoupled from playback

The "wall clock" line shows **iters · per-iter cost**, *not* real elapsed playback time. Per-iter cost is per-K, calibrated to known WMMSE runtimes (e.g. K=20 ≈ 52 ms / 100 iters):

```js
MS_PER_ITER_BY_K = { 5: 0.09, 8: 0.20, 10: 0.32, 15: 0.55, 20: 0.78 };  // ms
```

Alongside, a **GNN single-shot comparison** (also K-aware so the contrast widens honestly with K, since the GNN scales with K too — just much more slowly):

```js
GNN_MS_BY_K = { 5: 3, 8: 4, 10: 5, 15: 7, 20: 9 };  // ms
```

Rendered as e.g. `wall clock  55.0 ms  →  GNN: ~7 ms (1 forward pass)` at K=15, iter=100. The GNN value updates whenever K changes; rendered in `--c-orange`.

### Hover coupling

Hovering a layout pair (via the invisible thick stroke along the Tx→Rx line + the Rx hit area) toggles `is-hot` on both that `.pair-group` and the matching `.bar-group` in the strip. Vice versa for hovering a bar slot. `is-hot` styling: a faint white ring around Rx + thicker direct-line stroke; bar text label brightens to `--text`. Cleared on `mouseleave` and on Reset / K change / Seed shuffle.

### Readout panel

Four rows, mono, in a CSS grid (`max-content 1fr max-content max-content`), `font-variant-numeric: tabular-nums` everywhere:

- `iter        N / MAX_ITER`
- `sum-rate    X.XX b/s/Hz       →  converges at  Y.YY`
- `wall clock  W.W ms             →  GNN: ~G ms (1 forward pass)` *(GNN value K-dependent, in `--c-orange`)*
- A thin orange progress bar (3px tall, soft orange `box-shadow` glow) spanning all columns, width = `100·iter/MAX_ITER %`.

### Reduced-motion

Under `(prefers-reduced-motion: reduce)`, **Play immediately advances `iter` to `MAX_ITER` and renders once** instead of starting the RAF loop. Step/Reset still work normally. CSS transitions on `.direct-line`, `.tx-mark`, `.rx-mark`, `.bar`, `.progress-fill` are zeroed via the same media query.

### Constants block (top of `wmmse-iter.js`)

Group all tunables into a single `const` block at the top of the file so a reader can skim what's adjustable without grep:

- Iteration / speed: `MAX_ITER`, `BASE_MS_PER_ITER`, `SPEED_LEVELS`, `DEFAULT_SPEED_INDEX`.
- Topology choices: `K_CHOICES`, `DEFAULT_K`, `DEFAULT_SEED`.
- Channel: `FIELD_SIZE`, `D0`, `PATHLOSS_G`, `RX_RADIUS_LO`, `RX_RADIUS_HI`, `SHADOW_DB`, `VAR_NOISE`, `PMAX`.
- Calibration tables: `MS_PER_ITER_BY_K`, `GNN_MS_BY_K`.
- Layout SVG: `LAYOUT_W`, `LAYOUT_H`, `LAYOUT_PAD`, `TX_SIZE`, `RX_RADIUS`, `DIRECT_LINK_W`, `INTERFERER_PER_RX`, `INTERFERER_LINE_W`.
- Curve SVG: `CURVE_W`, `CURVE_H`, `CURVE_PAD_L/R/T/B`.
- Strip SVG: `STRIP_H`, `STRIP_PAD_X/T/B`, `BAR_GAP_RATIO`.

### Suggested lead-in copy (one paragraph above the widget)

> Initial state: every link transmits at full power. WMMSE alternates between updating receiver filters and re-allocating transmit power; weak links get silenced so strong ones stop drowning each other out. The dashed line is the converged sum-rate — the algorithm chases it, but the last few percent take most of the iterations.

Heading style: `Watch the iteration` with a faint mono lead-in `— hit play, raise K, feel the wall clock grow.`
