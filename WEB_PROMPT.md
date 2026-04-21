# WEB_PROMPT.md — build brief for the capstone showcase site

> **Purpose.** This file is a **self-contained prompt for an AI coding assistant** (Claude Code, Cursor, etc.) to build v1 of the capstone showcase website. It consolidates decisions from `BRAINSTORM.md` and removes the "we're still deciding" framing. Read `CLAUDE.md` for repository context; read `WEBSITE.md` only if you need to explain the dev loop to a beginner.
>
> **Deliverable.** A polished, one-page static site that runs with `python -m http.server 8000` and tells the capstone's story with three core interactive widgets (a D2D scaling slider, a JSAC topology slider, and a JSAC layout gallery) plus two richer widgets (a power-allocation viewer and an interference physics sandbox).
>
> **Target directory.** The user specifies the output directory **at invocation time** (e.g., *"build into `prototype/`"*, or *"build into `web/`"*). Throughout this file, the placeholder **`{{SITE_DIR}}/`** stands for that directory. Substitute the name the user gave you. Create the directory fresh — **do not build into the existing `site/`**; `site/` is a protected reference scaffold the user keeps as-is. (There may or may not be a `sandbox/` folder — ignore it if present; it is not an input.)

---

## 1. Objective

Build a single-page static website that communicates the result of an ML-based wireless-power-allocation capstone to a technically literate reader (capstone committee + general ML/wireless audience). The site should (a) render the core figures with web-native polish, (b) let the reader *feel* the key findings through five interactive widgets, and (c) look like it was built by engineers — **"modern cool-tech"**, not "academic poster."

---

## 2. Project context (what the research is)

Two scenarios, both first-class:

- **Scenario_D2D — Foundation.** Device-to-Device / interference-channel power allocation, K transceiver pairs. We trained **both a DNN (MLP, supervised against iterative WMMSE) and a GNN (IGCNet, unsupervised against sum-rate)**. **The headline finding: the DNN's approximation quality degrades as K grows; the GNN tracks WMMSE across sizes.** Reason is architectural — the MLP's input/output dimensions are tied to K (no weight sharing, parameter count blows up), while the GNN's message-passing shares one small module across however many nodes you hand it. A variant (`test_QoS.py`) adds a minimum-rate QoS constraint per user.
- **Scenario_JSAC — Application.** Joint Sensing-and-Communication on vehicular links. **Blue cars (Tx)** each serve several **Yellow (sensing Rx)** and **Green (communication Rx)** on orthogonal channels inside a cluster; same-channel links across different Blue clusters interfere. GNN-only. Adds a **hard per-Blue-car power budget** (enforced by construction via per-group softmax) and a **soft Yellow-SINR constraint** (squared-hinge penalty in the loss). Compared against WMMSE and a naive equal-power baseline.

The method = IGCNet (4 stacked message-passing layers with `aggr='max'`, K-independent). The motivating finding = D2D scaling. The constrained-application demonstration = JSAC.

Relevant files in the repo:

- `Scenario_D2D/` — has `model_dnn.py`, `model_gnn.py`, `main.py`, `test_QoS.py`; outputs live in `saves/` (no-QoS) and `saves_QoS/`.
- `Scenario_JSAC/` — has `model_gnn.py`, `main.py`, `test_JSAC.py`; outputs live in `save_main/` (sweeps) and `save_test/` (single-topology deep dive).
- Color constants (shared between Python plots and the website): `Scenario_JSAC/main.py:393` — `{'Naive': '#888888', 'WMMSE': '#2196F3', 'GNN': '#FF5722'}`. Add `Yellow #F6C445` (sensing), `Green #4CAF50` (comm), and an extra `DNN` color for D2D (suggest `#9C27B0` violet — visually distinct from Blue/Orange).

---

## 3. Scope, constraints, non-negotiables

- **Plain HTML / CSS / JavaScript only.** No React, no Vue, no Svelte, no Next.js, no Astro. No bundler. No build step. No `package.json`. No `node_modules/`.
- **Runs from `{{SITE_DIR}}/` via `python -m http.server 8000`.** Opening `index.html` directly via `file://` should degrade gracefully (images render; `fetch()` calls log a clear error message rather than crashing the page).
- **Offline-friendly.** No runtime dependency on external CDNs. Fonts may be self-hosted (`{{SITE_DIR}}/assets/fonts/`) or loaded from Google Fonts with a `system-ui` fallback stack that still looks acceptable. Any third-party JS (there shouldn't be any) must be vendored into `{{SITE_DIR}}/assets/vendor/`.
- **GitHub-Pages-clean.** No absolute paths that assume a server root; all asset URLs relative to `index.html`. The same files that work locally must work when dropped on Pages.
- **Accessibility basics.** Semantic HTML (`<main>`, `<section>`, `<figure>`, `<figcaption>`, `<nav>`). Keyboard-navigable interactive widgets. `aria-label` on icon-only buttons. Respect `prefers-reduced-motion: reduce`.
- **No model artifacts shipped.** `.pth` and `.pkl` files never enter `{{SITE_DIR}}/assets/`. Everything the site reads is pre-exported JSON or SVG/PNG.

---

## 4. Audience & tone

- Primary: capstone defense committee (wireless + ML PhDs).
- Secondary: general technical reader stumbling on the link.
- Not the audience: casual consumer / marketing visitor.

Copy should be **concise, precise, understated**. No marketing exclamations. No "Revolutionary!" / "Amazing!" / "🚀". Captions explain figures, not the other way around.

---

## 5. Vibe spec — "Modern, cool-tech"

Reference tone: Linear, Vercel, Anthropic docs, research-lab microsites. Looks *built*, not *typeset*.

**Color (dark by default).**
- Background: `#0a0b0e` (page), `#14161c` (panel/card surface).
- Body text: `#e8e8ea`. Secondary: `#9a9aa0`. Tertiary / rule lines: `rgba(255,255,255,0.06–0.10)`.
- Scenario accents (source of truth = `Scenario_JSAC/main.py:393`, saturation-nudged for dark):
  - `Blue #4DA3FF` — WMMSE baseline, Blue-car Tx.
  - `Orange #FF6A3D` — GNN (primary accent).
  - `Yellow #F6C445` — Yellow / sensing links.
  - `Green #4CAF50` — Green / communication links.
  - `Grey #888888` — Naive / equal power.
  - `Violet #B265D9` — DNN (new; D2D only).
- A single near-neutral steel tone (`#2a2e3a`) for UI chrome (tabs, inactive buttons).

**Typography.**
- Display / headings: **Inter** or **Geist** (600–700 weight, tight tracking `-0.01em`). Fallback stack includes `system-ui`, `-apple-system`, `Segoe UI`.
- Body: same sans family, 400–500 weight, 16–17px, line-height 1.6–1.7.
- Monospace accents: **JetBrains Mono** or **IBM Plex Mono**, 12–13px, used for eyebrows (`SECTION · 02`), axis ticks, numeric readouts, keyboard pills, inline code, DOIs. Track out `0.08–0.14em`.
- Numeric readouts use `font-variant-numeric: tabular-nums`.
- Scale: hero ~`clamp(40px, 6vw, 84px)`; H2 28–32px; H3 20–22px.

**Surfaces & motion.**
- One or two elevation levels via 1px borders (`rgba(255,255,255,0.06)`) and very quiet inner glows — *no* drop shadows. Rounded corners 8–12px.
- Sparing accent glow: soft radial tint behind the hero; faint colored halo around the *active* chart line. Never neon-party — "data center at 2am."
- Grid-based layout (not single column). Hero full-bleed; content in a ~1200px max container; figures can break out to ~1100px.
- Motion as signal: fade + 4–8px rise on scroll entry (≤300ms, `cubic-bezier(0.2, 0.8, 0.2, 1)`). Chart lines draw-in once on enter. Method toggles crossfade ~200ms. Nothing bounces or spins. **Disable all of it under `prefers-reduced-motion: reduce`.**

**Data-native chrome.**
- Monospace axis ticks and metric labels.
- Small live-ish touches: `● LIVE · GNN inference ~8ms` style readouts (use real numbers from the JSON dumps, not fake ones).
- Keyboard hint pills (`◀ ▶` or `[ / ]`) next to navigable widgets.
- Every interactive widget wears a mono eyebrow like `INTERACTIVE · drag K to explore` so the reader knows to engage.

**What to avoid.** Gradient-washed backgrounds, pastel blobs, playful illustrations, drop shadows, logo clouds, "we launched on Product Hunt" energy, emoji in body copy, any hero-sized stock photo.

---

## 6. Page structure (linear narrative)

One scrollable page, roughly these sections, in this order:

### 6.1 Hero
Full-bleed, ~90vh.
- Eyebrow mono: `WIRELESS POWER CONTROL · CAPSTONE`.
- Title (serif-free; Inter/Geist display): e.g. *"Approximating WMMSE with graph neural networks — for the classical interference channel, and for joint sensing-and-communication."* (Treat this as a placeholder the user will refine.)
- Sub-hero paragraph (~40 words) summarizing the arc: *DNN stopped scaling, GNN didn't, so we took the GNN to a harder problem.*
- Author line: render the literal tokens **`{{AUTHOR_NAME}}`, `{{AFFILIATION}}`, `{{DATE}}`** in the HTML. Do not invent values; leave them as `{{…}}` so the user can find-and-replace afterward. Style them as if the real values were present (same typography, same layout) so the hero still looks finished.
- Anchor row: monospace chips linking to each major section (`01 · PROBLEM`, `02 · D2D`, `03 · METHOD`, `04 · JSAC`, `05 · DEEP DIVE`, `06 · REFS`).
- Soft radial orange glow bottom-right; subtle grid or scanline overlay is OK if it stays quiet.

### 6.2 Problem (01)
Why this problem matters; why WMMSE's speed cost is real.
- Short text (≤120 words).
- One diagram: a single link's SINR formula rendered cleanly (MathML or SVG), paired with a tiny animated hint that iterating WMMSE is the bottleneck.
- Optional inline **widget F candidate** (see §7 — not required in v1): an animated WMMSE iteration counter.

### 6.3 Scenario 1 · D2D (Foundation) (02)
The scenario that earned the method.
- **6.3.1 Setup.** SVG diagram of K transceiver pairs in a square field. Minimal annotation.
- **6.3.2 Two models we built.** Side-by-side architecture cards: DNN (MLP 3×hidden, flattened \|h\|² input, K-dim power output, supervised MSE vs WMMSE) and GNN (IGCNet, 4 × message-passing, unsupervised sum-rate loss). Each card has: input shape, output shape, parameter count, "scales with K?" row.
- **6.3.3 Headline figure — the scaling finding.** This is the page's money shot. See **widget C′** in §7.
- **6.3.4 Why DNN scales poorly.** Three-sentence architectural explainer pinned next to the figure; monospace mini-table listing: *MLP: input `K²` floats, output `K` floats, params `O(K²)`. GNN: input per-node, shared params `~5k` regardless of K.*
- **6.3.5 Optional sub-strip — QoS variant.** A compact chart showing GNN+QoS respects `r_min` across a range of thresholds where unconstrained WMMSE violates. Use data from `Scenario_D2D/saves_QoS/`.

### 6.4 Method bridge · Why GNN (03)
~⅔ viewport height. A single clean diagram of IGCNet: `[node features]` → 4 × `IGConv (aggr='max')` → per-node output head → `p` (power vector). Callouts: weight sharing across nodes, `O(|E|)` inference, K-independent. Keep prose under 80 words; the diagram does the work.

### 6.5 Scenario 2 · JSAC (Application) (04)
The GNN handed a harder problem.
- **6.5.1 Setup.** SVG with three Blue clusters, their Yellow/Green Rx, dashed lines for same-channel interference across Blue clusters, solid light lines for intra-cluster links. Visual vocabulary: Tx as filled squares in `--c-blue`, Yellow Rx as open rings in `--c-yellow`, Green Rx as open rings in `--c-green`, inter-cluster interference as dashed grey lines, intra-cluster links as thin faint solid lines.
- **6.5.2 What's new vs D2D.** Three bullets: two edge types, per-group softmax enforces `∑_g p = P_max` by construction, squared-hinge on Yellow SINR.
- **6.5.3 Results.** 2×2 grid rendered in SVG from `{{SITE_DIR}}/assets/data/sweep_B.json` + `sweep_M.json`:
  - (top-left) Green sum-rate vs B — three lines (Naive/WMMSE/GNN).
  - (top-right) Yellow violation % vs B.
  - (bottom-left) Green sum-rate vs (M_y, M_g).
  - (bottom-right) Yellow violation % vs (M_y, M_g).
- **6.5.4 Runtime table.** Compact monospaced table: per-method mean inference time (ms) on the largest evaluated topology; highlight the GNN row.

### 6.6 Deep dive (05)
Lower-density. Two sub-panels:
- **JSAC** — three charts from `jsac_deep_dive.png` (per-link power bars with Yellow/Green colored, per-group budget utilization violin, Green-vs-Yellow power-share box) + a strip of layout snapshots using **widget A** (see §7).
- **D2D (QoS)** — a smaller strip: for the unconstrained and the constrained cases, plot CDF of per-user rate with the `r_min` threshold overlaid. Optional for v1 if data export is painful.

### 6.7 References (06)
Monospaced bibliography block. The three papers from `README.md` (W1P1, W2P2, W4P2) with clickable DOIs rendered in accent color. Footer-scale type, `letter-spacing: 0.04em`.

### 6.8 Footer
Author, year, repo link, commit SHA (optional — can be hand-edited).

---

## 7. Interactive widgets (A–E in scope for v1)

Build all five. Each is a **Web Component (custom element)** with its own folder in `site/components/<name>/`, using **Shadow DOM for scoped styles**. Each reads its data from `{{SITE_DIR}}/assets/data/`; none of them run models at runtime.

**Shared rules for all widgets.**
- A mono eyebrow inside the component: `INTERACTIVE · <hint>`.
- Every widget must be keyboard-operable (Tab to focus, arrow keys to step).
- Must respect `prefers-reduced-motion`: swap animated transitions for instantaneous state changes when the media query matches.
- Loading state: render a quiet skeleton until JSON resolves; if `fetch` fails, render the eyebrow text + a mono `offline — open via \`python -m http.server\`` message in the tertiary text color. Do **not** throw.

### A. `<layout-gallery>` — JSAC layout snapshot browser *(easy)*
Carousel of pre-rendered layout snapshots, cycling through a handful of seeds. For each snapshot, three method views (Naive / WMMSE / GNN) crossfade when the method toggle changes. Rx dot *brightness* encodes allocated power.
- Inputs: `{{SITE_DIR}}/assets/images/layouts/jsac_seed{01..N}_{naive|wmmse|gnn}.svg` (or PNG).
- Controls: prev/next arrows, method toggle (segmented control), thumbnail strip.
- Data file (optional metadata): `{{SITE_DIR}}/assets/data/layouts_index.json` with `[{seed, config, metrics_per_method}]`.
- Keyboard: `←/→` to navigate, `1/2/3` to switch method.

### B. `<method-toggle>` — in-chart method filter *(easy)*
A segmented-control component that, when placed above or inside an SVG chart, toggles visibility of individual series with a 200ms crossfade. Emits a `change` custom event. Used in both Section 6.3.3 (D2D scaling) and Section 6.5.3 (JSAC sweep charts).
- Attributes: `data-methods="Naive,WMMSE,GNN"`, `data-default="all"`.
- Target chart is discovered via `data-target="<chart-id>"` or by being nested inside it.

### C + C′. `<sweep-slider>` — one component, two instances *(medium)*
A slider + multi-line chart + numeric readout panel. The same component powers both:
- **C · JSAC topology slider.** Swept over `B ∈ {3,5,7,10,13}`, series = `[Naive, WMMSE, GNN]`, metrics = `[green_sum_rate, yellow_violation_pct, inference_ms]`. Source: `{{SITE_DIR}}/assets/data/sweep_B.json`.
- **C′ · D2D scaling slider.** Swept over `K ∈ {10, 20, 30, 50, 100, ...}`, series = `[WMMSE, GNN, DNN]`, metrics = `[sum_rate, sum_rate_ratio_vs_wmmse, inference_ms]`. Source: `{{SITE_DIR}}/assets/data/d2d_sweep_K.json`.

Behavior:
- Slider snaps to the x-values present in the JSON.
- On change, the vertical "cursor" line in the chart slides to the selected x; the readout panel updates with tabular-nums.
- Draw-in animation first time the component enters the viewport (use `IntersectionObserver`).
- Attributes: `data-src`, `data-x-key`, `data-series`, `data-metrics`, `data-x-label`, `data-y-label`.

C′ is the highest-leverage widget on the page — treat it as a first-class deliverable, not a variant.

### D. `<layout-viewer>` — SVG power-allocation map *(medium)*
Render one layout as an SVG map. Toggle between methods; Rx marker size *and* brightness encode allocated power; edges colored by link type; clicking a Blue-car highlights only its cluster and dims the rest.
- Input: `{{SITE_DIR}}/assets/data/layouts/jsac_layout_{id}.json`, shape:
  ```jsonc
  {
    "scenario": "jsac",
    "config": { "B": 10, "M_y": 2, "M_g": 3, "field": 225 },
    "blue":   [ { "id": 0, "x": 34.2, "y": 110.5 }, … ],
    "rx":     [
      { "id": 0, "blue": 0, "channel": 0, "type": "yellow", "x": …, "y": … },
      { "id": 1, "blue": 0, "channel": 1, "type": "green",  "x": …, "y": … },
      …
    ],
    "power": { "Naive": [ … ], "WMMSE": [ … ], "GNN": [ … ] },
    "metrics": { "green_sumrate": { "Naive": 132.2, "WMMSE": 154.8, "GNN": 151.1 }, … }
  }
  ```
- Includes a D2D variant (`"scenario": "d2d"`) — same component; the renderer branches on `scenario`.

### E. `<interference-sandbox>` — drag-a-Tx, watch channels change *(medium-hard)*
Reader drags one Tx around a 2D field. The component recomputes path loss in JavaScript (use the same formula family as the codebase: path loss ~ `(d0/d)^γ` with log-normal shadowing frozen per-Tx — shadowing is optional and can be toggled off for clarity). Render the `K × K` channel magnitude matrix as a heatmap that updates live.
- No model inference. Physics only.
- Parameters fixed: γ = 3, reference distance d0 = 200 m (or pick a pedagogically clean value — document in a one-line note inside the component). Start with 4–6 Tx; let one of them be draggable.
- Keyboard: when the active Tx is focused, arrow keys nudge its position by 1m; Shift+arrow by 10m.
- Caption explicitly tells the reader: *this is channel physics, not a model prediction.*

---

## 8. Data — what to export and where

Write two small Python helper scripts (one per scenario) that load the existing pickles and dump small JSONs into `{{SITE_DIR}}/assets/data/`. **These scripts belong at `Scenario_*/export_for_site.py` and should be self-contained (load the repo's pickles, dump JSON, done — no retraining).**

### 8.1 JSAC exports (`Scenario_JSAC/export_for_site.py`)
From `save_main/results_sweep_B.pkl` and `save_main/results_sweep_M.pkl`:
- `{{SITE_DIR}}/assets/data/sweep_B.json`
- `{{SITE_DIR}}/assets/data/sweep_M.json`

Shape:
```jsonc
{
  "sweep": "B",                         // or "M"
  "x_key": "B",                         // or "M" (M_y+M_g)
  "x_label": "Number of Blue cars",
  "methods": ["Naive", "WMMSE", "GNN"],
  "points": [
    {
      "x": 3,
      "K": 15,
      "metrics": {
        "Naive": { "green_sumrate": …, "yellow_violation_pct": …, "inference_ms": … },
        "WMMSE": { … },
        "GNN":   { … }
      }
    },
    …
  ]
}
```

From `save_test/test_jsac_results.pkl` + layout snapshots: export **3–6 representative layouts** to `{{SITE_DIR}}/assets/data/layouts/jsac_layout_{id}.json` in the shape shown in §7.D. If `save_test/` has pre-rendered snapshot PNGs, also copy them to `{{SITE_DIR}}/assets/images/layouts/` with the naming scheme `jsac_seed{id}_{method}.png`.

### 8.2 D2D exports (`Scenario_D2D/export_for_site.py`)
- `{{SITE_DIR}}/assets/data/d2d_sweep_K.json` — same JSON shape as JSAC sweeps but with `methods: ["WMMSE", "GNN", "DNN"]` and metrics `sum_rate`, `sum_rate_ratio_vs_wmmse`, `inference_ms`.
- `{{SITE_DIR}}/assets/data/d2d_qos.json` (optional) — CDFs of per-user rate for unconstrained vs QoS-constrained, one object per method.
- If the D2D `saves/` pickles are absent on a fresh clone (they're `.gitignore`d), the script should print a clear `"(re-run Scenario_D2D/main.py to regenerate)"` message and exit cleanly rather than crash.

### 8.3 Figures as SVG
Prefer exporting the polished matplotlib figures as **SVG** (`plt.savefig('x.svg')`) into `{{SITE_DIR}}/assets/images/`. Raster PNGs are acceptable only for the layout snapshots (where scatter density makes SVG bloated).

### 8.4 When an asset doesn't exist yet — use the placeholder
An image placeholder already lives in the existing reference scaffold at `site/assets/images/placeholder.svg` (a 120×120 generic "picture" glyph). **Copy that file into `{{SITE_DIR}}/assets/images/placeholder.svg` as the first step of your build**, and use it whenever the image this spec references hasn't been produced yet. Concretely:

- In HTML, reference it: `<img src="assets/images/placeholder.svg" alt="(TODO: jsac_deep_dive figure)" />`.
- Leave an HTML comment next to it: `<!-- TODO: export jsac_deep_dive.svg from Scenario_JSAC/save_test/ -->` — so a later pass knows what to replace.
- **Do not delete the `<img>` element.** Do not fabricate image content as inline SVG or data URIs unless the figure is genuinely meant to be drawn in the browser (widgets A–E are the exception — they *build* SVG at runtime and that's the point).

Same idea for missing JSON data: if an `export_for_site.py` run can't produce a file because the upstream pickle is absent, write a **stub JSON** with the correct top-level shape and a `"_stub": true` marker, so components render an empty state (the mono `offline — open via \`python -m http.server\`` message pattern from §7) instead of throwing. Print a clear warning from the export script naming exactly which `python Scenario_*/main.py` run would regenerate the missing pickle.

---

## 9. File layout

```
{{SITE_DIR}}/
├── index.html
├── styles.css                    # tokens (CSS variables) + page-level layout
├── script.js                     # page orchestration; imports components as ES modules
├── components/
│   ├── sweep-slider/
│   │   ├── sweep-slider.js       # defines <sweep-slider>
│   │   └── sweep-slider.css      # adopted via CSSStyleSheet into Shadow DOM
│   ├── layout-gallery/
│   ├── method-toggle/
│   ├── layout-viewer/
│   ├── interference-sandbox/
│   └── _frozen/                  # copies of known-good component versions (created as we go)
├── assets/
│   ├── images/
│   │   ├── placeholder.svg       # copied from site/assets/images/placeholder.svg (see §8.4)
│   │   ├── layouts/
│   │   └── figures/              # exported SVGs from the matplotlib pipeline
│   ├── icons/                    # SVG glyphs (arrow, info, external-link, etc.)
│   ├── fonts/                    # self-hosted Inter + JetBrains Mono .woff2 (optional)
│   ├── data/
│   │   ├── sweep_B.json
│   │   ├── sweep_M.json
│   │   ├── d2d_sweep_K.json
│   │   ├── d2d_qos.json
│   │   ├── layouts_index.json
│   │   └── layouts/
│   │       └── jsac_layout_01.json …
│   └── vendor/                   # any vendored third-party JS (ideally empty)
```

The existing `site/` directory at the repo root is a **minimal reference scaffold the user keeps for their own tracking** — do not build into it, do not edit files inside it, do not delete it. The only file you read from it is the placeholder SVG (§8.4), which you *copy* into `{{SITE_DIR}}/assets/images/`.

---

## 10. Tech conventions (how to build it)

- Components are **ES-module Web Components** with **Shadow DOM**. Load via `<script type="module" src="components/sweep-slider/sweep-slider.js">` in `index.html`.
- Use native `<svg>` — no D3, no Chart.js, no Plotly. The math for lines, bars, axes, and gridlines is small and readable, and custom code keeps the "modern cool-tech" detailing (mono ticks, tabular-nums readouts, accent halos on active lines) under our control.
- State inside components is plain class fields. No state-management library.
- Data loading: one top-level `DataStore` in `script.js` (or per-component `fetch`) — pick one and be consistent. Cache fetched JSON on `window.__dataCache` to avoid refetch when the user scrolls back.
- Animations via CSS transitions first, `requestAnimationFrame` only when CSS can't do it (e.g. draw-in of an SVG path using `stroke-dashoffset`). All motion gated by `prefers-reduced-motion`.
- `styles.css` defines CSS custom properties (design tokens):
  ```css
  :root {
    --bg:        #0a0b0e;
    --surface:   #14161c;
    --text:      #e8e8ea;
    --text-dim:  #9a9aa0;
    --rule:      rgba(255,255,255,0.08);
  
    --c-blue:    #4da3ff;
    --c-orange:  #ff6a3d;
    --c-yellow:  #f6c445;
    --c-green:   #4caf50;
    --c-grey:    #888888;
    --c-violet:  #b265d9;
  
    --font-sans: "Inter", system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    --font-mono: "JetBrains Mono", ui-monospace, Menlo, monospace;
  
    --radius-card: 10px;
    --pad-card: 20px 24px;
    --ease:   cubic-bezier(0.2, 0.8, 0.2, 1);
  }
  ```
- Every component imports these tokens via `:host { color: var(--text); … }` inside its Shadow DOM stylesheet (custom properties pierce Shadow DOM by design — this is the intended pattern).

---

## 11. Non-goals (explicit)

- **No live Python.** No Pyodide, no ONNX-in-browser for v1 — pre-computed JSON only. (ONNX is a possible v2.)
- **No backend.** No Flask, no FastAPI, no Node server.
- **No SSR, no hydration, no routing.** One HTML file, client-side interactivity only.
- **No dependency on a CMS, markdown pipeline, or templating engine.** Content is written directly into `index.html`.
- **No analytics, no cookie banner, no telemetry.**
- **No dark-mode toggle in v1.** Dark is canonical. (Light mode is a v2 option.)
- **Don't retrain or refactor the research code.** The only allowed research-side change is the two `export_for_site.py` helpers in §8.
- **Don't modify the existing `site/` directory.** It's a minimal reference scaffold the user keeps for their own tracking. Read-only — except for copying `site/assets/images/placeholder.svg` into your build as instructed in §8.4.

---

## 12. Build order (suggested)

1. **Tokens and shell.** Write `styles.css` tokens + a skeletal `index.html` with all section `<section id="...">` landmarks and the monospace anchor nav.
2. **Data export scripts.** Write `Scenario_JSAC/export_for_site.py` and `Scenario_D2D/export_for_site.py`; run them; confirm JSON lives under `{{SITE_DIR}}/assets/data/`.
3. **`<sweep-slider>` first** (C + C′). Ship it rendering both D2D scaling and JSAC B-sweep. This is the highest-risk component; de-risk early.
4. **`<method-toggle>`** (B) — wire into the 2×2 JSAC grid.
5. **`<layout-gallery>`** (A) — export representative layout PNG/SVGs first, then the component is a thin viewer.
6. **`<layout-viewer>`** (D) — pure SVG from layout JSONs.
7. **`<interference-sandbox>`** (E) — physics + heatmap, last because it's furthest from the core narrative.
8. **Hero + copy pass.** Author text lives in `index.html`; the user will refine wording.
9. **Motion + polish.** Draw-in animations, accent glow, keyboard hints, the bibliography footer.
10. **Accessibility + reduced-motion sweep.** Tab through the page; verify every widget responds to keyboard; verify `prefers-reduced-motion` kills animations.

---

## 13. Acceptance checklist

A reviewer should be able to answer *yes* to each:

- [ ] `cd {{SITE_DIR}} && python -m http.server 8000` → page loads at `localhost:8000` with no console errors.
- [ ] The narrative reads D2D-first, JSAC-second; both are visually equivalent in weight.
- [ ] The D2D scaling slider (C′) is the visual centerpiece of the D2D section.
- [ ] All five widgets (A, B, C, C′, D, E) render real data from `{{SITE_DIR}}/assets/data/`, not placeholders.
- [ ] Tab-navigating from the top of the page reaches every interactive widget in a sensible order.
- [ ] Enabling `prefers-reduced-motion` in the OS or DevTools disables all animations; content still reads correctly.
- [ ] No `.pth` / `.pkl` files under `{{SITE_DIR}}/`.
- [ ] No `package.json`, no `node_modules/`, no build scripts required.
- [ ] Opening the site on a 1280px-wide laptop, a 1920px display, and a 390px phone all produce a readable layout.
- [ ] Color palette matches §5 and the scenario plots in `Scenario_JSAC/main.py:393` (saturation-nudged for dark).
- [ ] The bibliography renders the three papers with working DOI links.

---

## 14. One-liner restatement (the prompt to hand the AI)

> Build `{{SITE_DIR}}/` (directory name supplied by the user at invocation — do **not** build into the existing `site/`, which is a read-only reference scaffold) — a polished, offline, one-page static showcase for a wireless-power-allocation capstone. Plain HTML/CSS/JS, no framework, no build step. Dark modern-tech aesthetic (Linear / Vercel / Anthropic-docs reference). Tell the two-scenario story: **D2D (DNN-vs-GNN scaling finding) → method bridge → JSAC (constrained application)** → deep dive → refs. Ship five Web Components (Shadow DOM): `<sweep-slider>` used twice (D2D K-scaling + JSAC B-sweep), `<method-toggle>` for in-chart series filtering, `<layout-gallery>` for JSAC snapshots, `<layout-viewer>` for single-layout SVG maps, `<interference-sandbox>` for drag-a-Tx channel physics. Read pre-exported JSONs from `{{SITE_DIR}}/assets/data/` (write `Scenario_{D2D,JSAC}/export_for_site.py` helpers to produce them). Respect `prefers-reduced-motion`; keyboard-navigate every widget; `COLORS` palette from `Scenario_JSAC/main.py:393` nudged for dark. See `WEB_PROMPT.md` for the full spec.
