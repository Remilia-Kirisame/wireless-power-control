In this branch `feat-visual`, we are going to develop a running website to show-off the result of this capstone project, and maybe some interactive features.

First, I'll brainstorm with AI and see what we can do with a website.

After I got a full precise prompt of the website (the vibe, the features, etc.), ask AI to scaffold the site version 1. Get it running locally.

Then decide whether Figma would help you refine specific screens. Repeat the workflow (Figma+Claude Code) to refine the website until a final deliverable.

---

## Brainstorm — what the website could be

The capstone has two scenarios: **D2D** (interference-channel power allocation, GNN + DNN) and **JSAC** (joint sensing-and-communication on vehicular links, GNN-only). Both compare a learned model against the iterative **WMMSE** baseline and a naive **equal-power** lower bound. The website should make this legible to a reader who has never opened the code, and give the reviewer something to click.

Think of the site as three concentric circles:

1. **A static poster** — what this is, why it matters, the result.
2. **Figures & tables** — the same deliverable as `save_main/jsac_results.png` and `save_test/jsac_*.png`, but web-native (clean, zoomable, with captions).
3. **Small interactive bits** — widgets that let the reader *see* the problem, not just read about it.

### Candidate sections (one page, scroll-driven)

- **Hero** — title, one-line pitch ("approximate WMMSE with a GNN, ~30× faster at inference, with a hard per-group power budget and a soft SINR constraint"), author, logos.
- **Problem** — the interference channel in one diagram. For D2D: K transceiver pairs. For JSAC: Blue-Yellow-Green cars, orthogonal channels inside a Blue cluster, same-channel interference across Blue clusters. A static SVG is enough; the D2D case can be the "simple version" and JSAC the "real version."
- **Method** — IGCNet in a single figure: node features → 4 × IGConv → per-group softmax → power vector. Call out the two edge types (interference, intra-group) and the `apply_group_softmax` trick that enforces the budget *by construction*.
- **Results** — 2×2 grid from `jsac_results.png` (Green SR & Yellow violation vs. B and vs. (M_y, M_g)). Plus a compact runtime table — the headline is that GNN trails WMMSE in sum-rate by a small margin but wins by orders of magnitude in latency.
- **Deep dive** — the three panels from `jsac_deep_dive.png` (per-link power bars, per-group budget utilization, Yellow-vs-Green share) and a layout snapshot strip where brightness of each Rx dot encodes allocated power.
- **D2D (archived)** — smaller section. Same shape as above but for the IC/IMAC setup; label it "prior work, kept for reference."
- **References** — the three papers in README (W1P1, W2P2, W4P2) with DOIs.

### Interactive ideas (pick a subset — the more ambitious ones are sandbox fodder)

Ordered by increasing effort:

- **A. Layout gallery (easy).** A small carousel of pre-rendered PNG layouts (from `save_test/jsac_layout_snapshots.png`) with method-name toggles (Equal / WMMSE / GNN). Pure image swap, no JS math.
- **B. Method comparison toggle (easy).** Same plot area, three buttons that fade in one method's line at a time on the sum-rate curve. Teaches the reader to read the figure.
- **C. Topology slider (medium).** Slider over `B ∈ {3, 5, 7, 10, 13}` (or `(M_y, M_g)` pairs). On change, the sum-rate / violation numbers update from a prebaked JSON dump of `results_sweep_B.pkl`. No model runs in the browser; just lookup.
- **D. Power-allocation viewer (medium).** Drop a single layout JSON (Blue/Yellow/Green positions + allocated powers per method) in. Render it as an SVG map where Rx dot size or brightness = allocated power, Tx arrows colored by target link type. Clicking a Blue car highlights its cluster.
- **E. Interference sandbox (medium-hard).** Let the reader drag a Blue car around a 2D field; recompute path loss in JS (the `(200/d)^3 × L` formula is trivial) and show the resulting interference matrix heatmap. Learn-by-feel that moving Blues close together wrecks the channels. *No GNN inference*, just channel physics.
- **F. WMMSE iteration visualizer (hard).** Port the WMMSE inner loop to JS for a small K (say K=10). Step-through: show power vector evolving over iterations, with sum-rate climbing. Anchor the "WMMSE is iterative, that's the cost we're paying" story.
- **G. ONNX-exported GNN in the browser (hard, stretch).** Export `IGCNet` to ONNX, run it with `onnxruntime-web`. Reader draws a layout, GNN predicts powers, site shows resulting SINRs. High wow-factor, real risk the export doesn't handle PyG's message-passing cleanly.

Sensible v1 scope: **hero + problem + method + results + deep dive + references**, static, with **A (layout gallery)** and **C (topology slider)** as the only interactive bits. Save D and E for `sandbox/` experimentation; F and G are research projects of their own.

### Vibe

"Clean academic poster feel." Concretely:

- Serif headings (e.g. system serif / Charter / Georgia), sans-serif body, generous line height.
- Muted palette tied to the scenario colors the codebase already uses: **Blue** `#2196F3` (WMMSE / Blue cars / Tx), **Orange** `#FF5722` (GNN / primary accent), **Yellow** `#F6C445` (sensing links), **Green** `#4CAF50` (comm links), **grey** `#888888` (naive). These match the `COLORS` dict in `Scenario_JSAC/main.py:393`.
- Wide margins, figures first, body text explains the figure — not the other way around.
- No shadows, no gradients, no animation for decoration. Transitions only where they aid comprehension (e.g. fading between methods).
- Single column, max ~720px reading width, figures can break out to ~1100px.
- Footer with DOI-style reference list.

### What goes into `site/` vs `sandbox/`

- `site/` — the polished one-page showcase. Build toward the v1 scope above.
- `sandbox/` — a scratch area. Same scaffold, no commitment to polish. Use it to try ideas D/E/F before promoting anything to `site/`. For the first pass, the sandbox will implement a **small "clean academic poster" demo**: hero card + a JSAC layout SVG + a method-comparison bar chart rendered from a tiny hard-coded results object. That proves the stack works end-to-end without a build step.

### Open questions to resolve before v1

- Do we want audio narration or a short screen-recording video walkthrough? (Probably not for v1.)
- Hosting: is this offline-only (capstone defense on the user's machine) or do we push to GitHub Pages? The initial direction is **offline** (stated in setup instructions), so no deploy step — but the scaffold should be GitHub-Pages-clean so we can flip later by just pushing the folder.
- Do we dump model artifacts (`.pth`, `.pkl`) into `site/assets/`? No — keep the site data-only. Convert the pickles we need into small JSON files.
