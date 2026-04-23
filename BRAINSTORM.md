In this branch `feat-visual`, we are going to develop a running website to show-off the result of this capstone project, and maybe some interactive features.

First, I'll brainstorm with AI and see what we can do with a website.

After I got a full precise prompt of the website (the vibe, the features, etc.), ask AI to scaffold the site version 1. Get it running locally.

Then decide whether Figma would help you refine specific screens. Repeat the workflow (Figma+Claude Code) to refine the website until a final deliverable.

---

## Brainstorm — what the website could be

The capstone has two scenarios and **both are first-class results**, not one main + one archived. The narrative arc is:

1. **D2D — Foundation.** We built *both* a DNN (MLP, supervised) and a GNN (IGCNet, also approximating WMMSE). The headline finding: **DNN approximation quality degrades as K grows, GNN stays close to WMMSE.** The reason is architectural — the MLP's input/output dimensions are pinned to K, so training at K=10 doesn't transfer and parameter count explodes; the GNN shares weights across nodes and is node-count-agnostic. This is *the* result that justifies going GNN-only in everything that follows.
2. **JSAC — Application.** The GNN is applied to a richer, constrained vehicular setting (Blue / Yellow / Green cars, per-group power budget, soft SINR constraint on sensing links). Demonstrates that the same architecture extends cleanly to a harder problem.

Both scenarios compare their learned model(s) against the iterative **WMMSE** baseline and a naive **equal-power** lower bound. The website should make this arc legible to a reader who has never opened the code, and give the reviewer something to click.

Think of the site as three concentric circles:

1. **A static poster** — what this is, why it matters, the result.
2. **Figures & tables** — the same deliverable as `save_main/jsac_results.png` and `save_test/jsac_*.png`, `saves/*.png`, `saves_QoS/*.png` but web-native (clean, zoomable, with captions).
3. **Small interactive bits** — widgets that let the reader *see* the problem, not just read about it.

### Candidate sections

The page tells a linear argument — *problem → D2D finding → why GNN → JSAC application → deep dive → refs* — but as of v1.1 the *shell* is a tab-based SPA rather than a single scrolling page (see **Navigation & entry animation** below). The section inventory below is unchanged; each section is now its own tab rather than a scroll target.

- **Hero** — title, one-line pitch (something like *"Approximating WMMSE with graph neural networks — for the classical interference channel, and for joint sensing-and-communication."*), author, affiliation.
- **Problem** — why wireless power allocation is hard in one diagram. WMMSE is near-optimal but iterative and slow; inference-time matters in fast-changing channels. Frame both scenarios briefly: D2D is the classical interference channel (K transceiver pairs, simple version); JSAC adds Rx types, per-group budgets, and SINR constraints (real version).

- **Scenario 1 · D2D (Foundation)** — the scenario where we discovered *what to keep* about ML for this problem.
  - Setup diagram: K transceiver pairs, Gaussian IC / IMAC.
  - Two models we trained: **DNN (MLP, supervised against WMMSE)** and **GNN (IGCNet)**. Small architecture cards for each.
  - **Headline figure — the scaling finding.** Sum-rate (or sum-rate / WMMSE ratio) as a function of K, three lines: WMMSE (reference), GNN, DNN. The DNN line falls away as K grows; the GNN tracks WMMSE. This single chart earns the entire JSAC section.
  - Quick architectural explainer: **why** the DNN scales poorly (fixed input/output dimensions tied to K, parameter count grows quadratically, no weight sharing) and why the GNN doesn't (one shared message-passing module runs over however many nodes you hand it).
  - Optional sub-panel: the **QoS variant** (`test_QoS.py`) — same setup with a minimum-rate constraint per user, showing the GNN still tracks WMMSE.

- **Method bridge · Why GNN** — a short, figure-driven "here's the generalizable lesson." IGCNet in one diagram: node features → 4 × IGConv (message-passing with `aggr='max'`) → output head → power vector. Call out: weight sharing across nodes, O(|E|) inference, K-independence. This is the joint between D2D and JSAC; keep it tight.

- **Scenario 2 · JSAC (Application)** — the GNN handed a harder problem.
  - Setup diagram: Blue (Tx) / Yellow (sensing Rx) / Green (comm Rx), orthogonal channels inside a cluster, same-channel interference across clusters.
  - What's *new* vs. D2D: two edge types (intra-group + inter-cluster interference), per-group softmax output (budget enforced by construction), squared-hinge penalty on Yellow SINR.
  - **Results** — 2×2 grid from `jsac_results.png` (Green SR & Yellow violation, vs. B and vs. (M_y, M_g)) + compact runtime table. Headline: GNN trails WMMSE on Green sum-rate by a small margin, wins orders-of-magnitude on latency.

- **Deep dive** — side-by-side lower-density section:
  - **JSAC** — the three panels from `jsac_deep_dive.png` (per-link power bars, per-group budget utilization, Yellow-vs-Green share) and a layout snapshot strip where Rx brightness encodes allocated power.
  - **D2D (QoS)** — a smaller strip showing how the GNN respects the minimum-rate constraint vs. unconstrained WMMSE. Optional for v1 if time is tight.

- **References** — the three papers in README (W1P1, W2P2, W4P2) with DOIs, rendered as a monospaced bibliography block.

### Interactive ideas (pick a subset — the more ambitious ones are sandbox fodder)

Ordered by increasing effort:

- **A. Layout gallery (easy).** A small carousel of pre-rendered PNG/SVG layouts (from `save_test/jsac_layout_snapshots.png`) with method-name toggles (Equal / WMMSE / GNN). Pure image swap, no JS math.
- **B. Method comparison toggle (easy).** Same plot area, three buttons that fade in one method's line at a time on the sum-rate curve. Teaches the reader to read the figure. Useful in *both* the D2D scaling section and the JSAC results section.
- **C. Topology slider — JSAC (medium).** Slider over `B ∈ {3, 5, 7, 10, 13}` (or `(M_y, M_g)` pairs). On change, sum-rate / violation numbers update from a prebaked JSON dump of `results_sweep_B.pkl`. No model runs in the browser; just lookup.
- **C′. Scaling slider — D2D (medium).** The companion to C, for the *foundation* scenario. Slider over `K ∈ {10, 20, 30, 50, ...}` from the D2D sweep. On change, three lines (WMMSE / GNN / DNN) and three number readouts update from a prebaked JSON dump of the D2D results. **This is the widget that makes the DNN-doesn't-scale story land viscerally.** Probably the single highest-leverage interactive in v1.
- **D. Power-allocation viewer (medium).** Drop a single layout JSON (Blue/Yellow/Green positions + allocated powers per method) in. Render it as an SVG map where Rx dot size or brightness = allocated power, Tx arrows colored by target link type. Clicking a Blue car highlights its cluster. JSAC-flavored; a D2D analogue (link-pair map, per-link power) is trivially similar.
- **E. Interference sandbox (medium-hard).** Let the reader drag a Tx around a 2D field; recompute path loss in JS (the `(200/d)^3 × L` formula is trivial) and show the resulting interference matrix heatmap. Learn-by-feel that moving Tx close together wrecks the channels. *No GNN inference*, just channel physics. Works for either scenario.
- **F. WMMSE iteration visualizer (hard).** Port the WMMSE inner loop to JS for a small K (say K=10). Step-through: show power vector evolving over iterations, with sum-rate climbing. Anchor the "WMMSE is iterative, that's the cost we're paying" story. Natural home: the problem/motivation section near the top.
- **G. ONNX-exported GNN in the browser (hard, stretch).** Export `IGCNet` to ONNX, run it with `onnxruntime-web`. Reader draws a layout, GNN predicts powers, site shows resulting SINRs. High wow-factor, real risk the export doesn't handle PyG's message-passing cleanly. If it works, wire it in for both D2D and JSAC — same model, two input shapes.

**Sensible v1 scope.** Sections: *hero + problem + D2D (foundation) + method bridge + JSAC (application) + deep dive + references.* Interactive widgets: **C′ (D2D scaling slider)** as the D2D section's centerpiece, **A (layout gallery)** and **C (JSAC topology slider)** in the JSAC section. Save D and E for `sandbox/` experimentation; F and G are stretch/research projects.

Both C and C′ share the same component pattern — a slider + a small multi-line chart + a numeric readout panel — so they can be built once as a `<sweep-slider>` Web Component and fed different JSON.

### Navigation & entry animation

The v1.1 shell replaces the single scrolling page with a **multi-view, tab-based SPA** (still one HTML file, still no framework — just a ~40-line hash router in `script.js`).

- **Landing view (Home).** On first load, the reader sees the hero content only — title, tagline, author/affiliation/date, soft orange radial glow. No stacked sections below; the six numbered sections live behind tabs.
- **Entrance animation — option 2, boot-plate overlay with load bar.** A full-viewport overlay (same near-black as `--bg`) carries two elements, stacked and centered: a **large monospace title plate** ("WIRELESS POWER CONTROL · CAPSTONE", uppercase, tracked `~0.12em`, `clamp(22px, 4vw, 44px)`) that fades up first (~400ms), and below it a **thin horizontal track** (~220–340px wide, 3px tall) whose **orange fill grows from left to right** over ~1200ms, reading as a load gauge. Once the fill completes, the overlay **wipes horizontally off-screen** (~400ms, `cubic-bezier(0.2, 0.8, 0.2, 1)`, same direction the bar traced) to reveal the home view. Total overlay budget: ~2.0s; the left sidebar then slides in (~360ms) and the hero title + anchor row fade up with short stagger delays (~180ms + 320ms). Skipped entirely under `prefers-reduced-motion: reduce`.
- **Left sidebar.** Persistent after the entry animation. Contains, top-to-bottom: a small monospaced logo block; a `◀ Home` link; a thin divider; the six numbered section links (`01 · Problem … 06 · References`); and a small `● LIVE · GNN ~58ms`-style metric pill at the bottom. ~240px wide on desktop; collapses to a horizontal top bar on <860px.
- **Tab switching.** Clicking a sidebar link updates `location.hash`; the router listens to `hashchange` and runs a two-phase fade (~160ms out, ~160ms in — total ~320ms) between views. URLs are deep-linkable (`#problem`, `#d2d`, …) and the browser back button works without extra code. Under `prefers-reduced-motion`, the swap is instantaneous.
- **Per-view reveals.** The existing `.reveal` fade-up pattern still applies, but fires when a view *activates* (all `.reveal` children of the entering view get `is-visible`) rather than on scroll — each view is short enough that staggering per element isn't needed.

**Future improvement — option 3 (canvas/SVG title animation).** Instead of a flat horizontal wipe, build the entrance as a slowly-resolving "interference field" that condenses into the title: animated particle dots or a noisy heat-map over canvas that crystallises into the logo mark as the overlay dissolves. Character-by-character reveal for the title, timed to the field clearing. Higher wow-factor but easily half a day to a day of polish — keep in the sandbox queue, ship v1.1 with the flat wipe.

### Vibe

**"Modern, cool-tech feel."** Think Linear / Vercel / Anthropic docs / research-lab microsites — the look of a thing that was *built*, not *typeset*. Concretely:

- **Dark by default.** Deep near-black background (`#0a0b0e`–`#111218`), high-contrast off-white body (`~#e8e8ea`), muted grey for secondary text. A light-mode toggle is optional; the canonical version is dark.
- **Typography.** Geometric sans for headings (Inter / Geist / Space Grotesk, tight tracking, 600–700 weight), clean sans for body (Inter, 400–500), **monospace accents** (JetBrains Mono / IBM Plex Mono / Geist Mono) for eyebrows, labels, code, numeric readouts, and axis ticks. Monospace is the "this is technical" tell; use it liberally for small metadata.
- **Palette — keep the codebase scenario colors, reframe them as accents on dark.** `Blue #2196F3` (WMMSE / Blue cars / Tx), `Orange #FF5722` (GNN / primary accent), `Yellow #F6C445` (sensing), `Green #4CAF50` (comm), `#888888` (naive) — from `COLORS` in `Scenario_JSAC/main.py:393`. On dark, nudge saturation: GNN-orange can run slightly hotter (`#FF6A3D`), WMMSE-blue slightly cooler (`#4DA3FF`). Add one or two near-neutral steel tones for UI chrome.
- **Surfaces & depth.** One or two elevation levels via subtle 1px borders (`rgba(255,255,255,0.06)`) and very quiet inner glows — not drop-shadows. Cards, charts, and interactive panels sit on faint panels (`#141620`-ish) against the near-black background. Rounded corners 8–12px.
- **Accent glow.** Sparingly: a soft radial tint behind the hero, a faint colored halo around the primary CTA / active chart line. Never neon-party; think "data center at 2am."
- **Grid, not column.** Break out of the single-column poster feel. Use a responsive grid — hero spans full width, results panels can be 2-up / 3-up, figures can go full-bleed. Generous negative space.
- **Motion as signal, not decoration.** Tiny reveals on scroll (fade + 4–8px rise, ≤300ms, easing `cubic-bezier(0.2, 0.8, 0.2, 1)`). Chart lines draw in once when they enter the viewport. Toggling a method crossfades in ~200ms. Nothing bounces, nothing spins. Respect `prefers-reduced-motion`.
- **Data-native UI chrome.** Small live-looking touches that signal "this is engineering": monospace axis ticks, numeric readouts with tabular figures (`font-variant-numeric: tabular-nums`), a tiny status dot next to "GNN inference: 8.2ms" style metrics, keyboard hint pills (`[ / ]` to navigate), command-palette aesthetic for any search/filter.
- **Interactivity framing.** Every interactive widget labels itself — a small eyebrow that says "interactive · drag to explore" or "live data · topology sweep" — so the reader knows to engage.
- **Typography scale.** Display (hero): ~clamp(40px, 6vw, 84px), tight leading, tight tracking. H2: ~28–32px. Body: 16–17px, line-height 1.65. Captions / eyebrows: 12–13px mono, tracked out `0.08–0.14em`.
- **Footer.** A references list rendered like a bibliography but styled as a monospaced block — DOIs rendered as links in the accent color.

What to avoid: gradients-as-backgrounds, pastel watercolor blobs, playful illustrations, heavy shadows, anything that reads "marketing landing page."

### What goes into `site/` vs `sandbox/`

- `site/` — the polished one-page showcase. Build toward the v1 scope above.
- `sandbox/` — a scratch area. Same scaffold, no commitment to polish. Use it to try ideas D/E/F before promoting anything to `site/`. For the first pass, the sandbox will implement a **small "clean academic poster" demo**: hero card + a JSAC layout SVG + a method-comparison bar chart rendered from a tiny hard-coded results object. That proves the stack works end-to-end without a build step.

### Open questions to resolve before v1

- Do we want audio narration or a short screen-recording video walkthrough? (Not for v1.)
- Hosting: is this offline-only (capstone defense on the user's machine) or do we push to GitHub Pages? The initial direction is **offline** (stated in setup instructions), so no deploy step — but the scaffold should be GitHub-Pages-clean so we can flip later by just pushing the folder.
- Do we dump model artifacts (`.pth`, `.pkl`) into `site/assets/`? No — keep the site data-only. Convert the pickles we need into small JSON files.
- **D2D data export.** The D2D scaling chart (and the C′ slider) needs a JSON that doesn't currently exist pre-formatted. We'll write a tiny `Scenario_D2D/export_for_site.py` that loads whatever sweep pickles live in `saves/` and dumps `{K, method, sum_rate, rate_ratio_vs_wmmse, inference_time}` tuples. If the D2D saves got wiped by `.gitignore`, we may need to re-run the D2D sweep once to regenerate them — minor, since no retraining of the already-saved models is required.
- **D2D visual parity.** Do we render D2D link layouts the same way as JSAC snapshots (dots + lines with power-encoded brightness)? Probably yes — one shared SVG layout component, two JSON shapes. That keeps the site's visual vocabulary consistent across scenarios.
