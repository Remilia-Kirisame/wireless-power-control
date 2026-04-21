# WEBSITE.md — how this site works, for someone new to web

This project has a website living next to the research code. It's a **static website**: three text files (HTML, CSS, JavaScript) plus an `assets/` folder of images and icons. No server-side code, no database, no "build step." If you can open a file in a browser, you can run it.

There are two parallel scaffolds:

- **`site/`** — the real capstone showcase (clean, polished deliverable).
- **`sandbox/`** — a scratch copy for trying ideas before promoting them to `site/`.

Both have the same shape:

```
site/                       sandbox/
├── index.html              ├── index.html
├── styles.css              ├── styles.css
├── script.js               ├── script.js
└── assets/                 └── assets/
    ├── images/                 ├── images/
    └── icons/                  └── icons/
```

## The three files, in one paragraph each

**`index.html`** is the skeleton. It's what the browser *reads first*. It declares the page's structure — headings, sections, images, buttons — as a tree of "tags" like `<h1>`, `<p>`, `<img>`. The `<head>` at the top contains metadata (title, character set) and **links to the other two files**: a `<link rel="stylesheet" href="styles.css">` tells the browser "load this CSS file and use it to style the page," and a `<script src="script.js" defer>` tells the browser "load this JavaScript file and run it after the HTML is parsed." That's how the three files become one running page.

**`styles.css`** is the visuals. CSS (Cascading Style Sheets) is a list of rules: *"for every `<h1>` on the page, use Georgia, 2rem, dark grey"*. It controls typography, colors, layout (grid, flex), spacing, and responsive behavior (different styling on mobile vs. desktop). You can change the look of the entire site by editing this one file — the HTML stays the same.

**`script.js`** is the behavior. JavaScript runs *inside the browser* once the page has loaded. It can read user input (clicks, slider drags), mutate the HTML tree on the fly ("when the user clicks this button, add that element"), fetch small JSON files, and redraw SVG charts. For a poster-style site, JS is optional flavor; for interactive demos (like the topology slider we plan to build), it's where the logic lives.

## How to run it locally

You have two options. Pick the second one unless you really just want to peek.

### Option 1 — double-click `index.html`

macOS Finder → open `site/index.html` → it opens in Safari/Chrome. **This works for pure HTML/CSS.** It breaks the moment your JavaScript tries to load another file (like `fetch('data.json')`), because browsers refuse to load local files from a page opened via the `file://` protocol — it's a security rule. You'll see CORS errors in the console.

### Option 2 — run a tiny local server (recommended)

A "local server" sounds heavy but is one command. It hands files to your browser over `http://localhost:...` instead of `file://...`, which satisfies the browser's security rules and makes everything behave like a real deployed website.

Python comes with one. From this repo's root:

```bash
cd site
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser. Ctrl-C in the terminal to stop. For the sandbox, do the same from `sandbox/` (use a different port so they can run side by side):

```bash
cd sandbox
python -m http.server 8001
# then open http://localhost:8001
```

Edit any of the three files, save, **refresh the browser tab**. That's the whole development loop. No compile, no bundle, no dependency install.

If you prefer VS Code, the extension **Live Server** does the same thing with a "Go Live" button and auto-refreshes on save. Totally optional.

## How the three files collaborate — a quick walkthrough

1. You open `http://localhost:8000/`. The server sees no specific file requested, so it sends `index.html`.
2. Browser reads `index.html` top-to-bottom. Hits `<link href="styles.css">` — makes a second request for the CSS, applies it.
3. Hits `<script src="script.js" defer>`. `defer` means: download now, but don't execute until the HTML tree is fully built. This keeps the page from freezing while JS loads.
4. Once the HTML is done, the JS runs. It can now `document.querySelector(...)` to find elements and attach click handlers, etc.
5. Images and icons referenced from either the HTML (via `<img src="assets/images/foo.png">`) or the CSS (`background-image: url("assets/icons/bar.svg")`) are fetched as the page renders.

All paths in HTML/CSS/JS are **relative to the HTML file** unless they start with `/` (absolute). So `assets/images/foo.png` works from `site/index.html` because the browser resolves it to `site/assets/images/foo.png`.

## What goes in `assets/`

- `assets/images/` — photographs, screenshots, exported figures from the scenario scripts (e.g. `jsac_results.png`). Anything pixel-based: PNG/JPG/WebP.
- `assets/icons/` — small vector graphics. Prefer SVG: crisp at any size, style-able from CSS, tiny file size. UI glyphs (arrows, logos, info marks) live here.

Keeping these separate isn't enforced by the browser — it's for your sanity.

## Why no framework? Why no build step?

Capstone showcases don't need React / Vue / Next.js. Those tools solve problems this site doesn't have (thousands of components, shared state, server-side rendering). The cost of bringing them in is big: Node.js, a `package.json`, a `node_modules/` folder of ~100MB, a build command, and a different deployment story. For one page that reads a few JSON blobs, **plain HTML/CSS/JS is the right tool**, and it'll still work ten years from now with zero maintenance.

When you *do* need frameworks: large apps with dynamic routing, multi-user state, or hundreds of reusable components. None of that applies here.

## Going from "local" to "published"

If you ever want the site live on the internet, the simplest path is **GitHub Pages**: push the folder to a GitHub repo, flip the Pages toggle, done. No config. Because this scaffold has no build step, nothing extra is needed — the same files that work locally will work on Pages. For now, the plan is offline-only (capstone defense on the user's laptop), so there's nothing to do.

## Summary

- `index.html` is what loads. It pulls in `styles.css` (look) and `script.js` (behavior).
- To preview: `cd site && python -m http.server 8000`, open `localhost:8000`, refresh after edits.
- `sandbox/` is a twin scaffold for experiments — use it to prototype before touching `site/`.
- No frameworks, no build, no server code. Three files and a folder of assets.

---

## Q&A log

Running notes from design conversations. Each item keeps the question and the short answer; expand later if it becomes a real section.

### Q1 — The two figures in the sandbox: are they drawn by JS, not images? What's the best practice for figures that come from Python?

Yes — correct. Both sandbox figures are **SVG elements built by `sandbox/script.js` at page load**, not images pulled from `assets/`. `drawLayout()` and `drawBars()` call `document.createElementNS(...)` to add `<rect>`, `<circle>`, `<line>`, `<text>` nodes into the empty `<svg>` placeholders in `index.html`.

For research figures that are *produced by Python*, there isn't one right answer — there are four common paths, and the best choice depends on whether you want the figure to be **locked** or **responsive/interactive**:

1. **Export as PNG or SVG from Python, reference as `<img>`.** Simplest. In matplotlib: `plt.savefig('fig.svg')` or `plt.savefig('fig.png', dpi=200)`. Drop into `assets/images/`, reference from HTML. **SVG is usually the better pick** for plots — crisp at any zoom, small file, works offline, no JS required. Good for: the hero / headline / final-deliverable figures you don't want to rebuild.
2. **Dump data as JSON from Python, draw in JS.** What the sandbox does. `pickle.load(...)` → `json.dump(...)` into `assets/data/*.json`, then the site reads it and draws SVG with its own code (or a small library like D3 / Chart.js / uPlot). Good for: any figure that needs to be **interactive** (a slider over B, toggling methods, hover tooltips). Downside: you rewrite the plotting code in JS.
3. **Use a Python-to-web plotting library.** Plotly / Bokeh / Altair can produce a standalone HTML fragment with interactivity baked in. You run it once in Python, embed the HTML chunk. Downside: bigger JS payload; less stylistic control over the "modern cool-tech" look.
4. **Headless screenshot pipeline.** Keep matplotlib figures, export as SVG into `assets/images/`, re-run whenever results change. Good middle ground if you never need interaction.

**Practical plan for this project.** For the v1 site:

- Take the **polished deliverable figures** (`jsac_results.png`, `jsac_deep_dive.png`, `jsac_layout_snapshots.png`) — **export them from the Python pipeline as SVG** (`savefig('...svg')`) and drop into `site/assets/images/`. Reference as `<img>` in HTML. These are "frozen" results; no need to rebuild in JS.
- For the **interactive bits** (topology slider over `B`, method toggles) — dump a small JSON from the relevant pickle (e.g. `results_sweep_B.pkl` → `site/assets/data/sweep_B.json`) and draw the chart in JS. A `Scenario_JSAC/export_for_site.py` helper can do the Pickle→JSON conversion in a couple of dozen lines.

Rule of thumb: **if the figure can be re-rendered from a table of numbers that fits in a few KB, make it interactive with JSON + JS. If it's a complex matplotlib figure with custom layout, export as SVG.**

### Q2 — How do I save an interactive feature as a reusable component so changes don't break it later?

Plain HTML/CSS/JS has no built-in "component" language construct the way React does — but the browser has a native mechanism called **Web Components** (a.k.a. custom elements) and there are a few lightweight conventions on top. Ordered from simplest to most robust:

1. **Folder convention.** Make `site/components/<name>/` with its own `name.html` snippet (or just a template inside JS), `name.css`, and `name.js`. The main page includes them via `<link rel="stylesheet">` and `<script src="...">`. Each component exposes a `mount(element, props)` function. Cheap, works immediately, but styles are global — you have to namespace class names yourself (e.g. `.topo-slider__bar`).
2. **Web Components (recommended).** Native browser feature, no libraries. You define a class:
   ```js
   // site/components/topology-slider/topology-slider.js
   class TopologySlider extends HTMLElement {
       connectedCallback() { /* build DOM, wire up events */ }
   }
   customElements.define('topology-slider', TopologySlider);
   ```
   Then use it in HTML as `<topology-slider data-src="assets/data/sweep_B.json"></topology-slider>`. Attach a **Shadow DOM** inside to get **scoped styles** — the component's CSS can't leak out, and the page's CSS can't leak in. Encapsulated, reusable, no build step, no framework. This is the "modern vanilla" way.
3. **ES modules.** Separate concern from packaging: write each component as an ES module (`export class ...`), load with `<script type="module">`. Pairs naturally with Web Components. Needs a local server (which you already have via `python -m http.server`).
4. **Freeze by copy.** When a component is in a known-good state, **copy** it to `site/components/_frozen/<name>-v1/` and pin imports in `index.html` to the frozen path. Future edits happen against a new `_frozen/<name>-v2/`. Crude but effective and zero infrastructure. Git commits / tags give you the same with more discipline.

**Practical plan for this project.**

- When we build the first interactive widget (likely the topology slider or method toggle), put it in `site/components/topology-slider/` as a Web Component (`<topology-slider>`) using Shadow DOM for style isolation.
- When a component behaves the way we want and we don't want to touch it again, **copy the folder to `_frozen/topology-slider-v1/`** and reference the frozen path. Keep the non-frozen version around for iteration.
- Each component folder gets a one-line `README` or top-of-file comment stating: what props it accepts (data-attributes), what events it emits, and what DOM it produces.

### Q3 — Can we run Python in the website for live model inference? (Letting the reader experience the GNN predict powers.)

Short answer: **yes, with caveats** — and for this project the honest recommendation is usually "don't run Python; pre-compute results or export the model to a web-native format." Your options, in rough order of practicality:

1. **Pre-compute and lookup (simplest, recommended default).** Run inference in Python for, say, 200 layouts × 3 methods, dump `{layout, method, powers, metrics}` tuples to `assets/data/*.json`. The site lets the reader click through *real results* — no runtime, no server, offline-friendly. Feels interactive even though no model is running. This is the right answer for a capstone defense on a single laptop.
2. **Export the GNN to ONNX, run with `onnxruntime-web` in the browser.** `onnxruntime-web` ships a WebAssembly runtime that executes ONNX models client-side — no Python, no server. `torch.onnx.export(model, sample_input, 'igcnet.onnx')` is one line, but PyTorch-Geometric's message-passing layers don't always export cleanly (custom ops, dynamic graphs). Realistic effort: a few evenings of debugging. If it works, the reader can sketch a layout and see the GNN's powers *actually computed live* in their browser.
3. **Pyodide (Python in the browser, via WebAssembly).** Ships CPython + NumPy + SciPy + pandas as a ~10–30MB WASM bundle. **PyTorch is not cleanly supported** (there's experimental work; not production-ready). Works great for Python math you want to demo live (e.g. the **WMMSE iterations on a tiny K=5–10 toy problem** — idea F in the brainstorm). Not the path for running the real GNN.
4. **TensorFlow.js.** Similar story to ONNX Runtime: convert PyTorch → ONNX → TF.js, run in the browser. More steps, similar outcome.
5. **Local Python backend.** Run a small Flask / FastAPI server on the user's laptop; the browser makes HTTP calls to `localhost:5000/predict`. Pro: uses the real research code, zero export pain. Con: no longer a "static" site — the capstone reviewer has to start a backend. Fine for an in-person defense, clumsy for anyone opening a ZIP later.

**Practical plan for this project.**

- **v1 — pre-computed lookup.** Dump 50–200 layouts' worth of Blue/Yellow/Green positions and three methods' power vectors to `assets/data/layouts.json`. The site renders any layout the reader picks; the reader feels like the model is running, but we just read JSON. Zero risk.
- **v2 — ONNX in the browser (stretch goal, optional).** Attempt the ONNX export of IGCNet. If it works, build a small widget where the reader drags Blue cars around a 2D field and the page recomputes channel losses (pure JS) + runs the GNN via `onnxruntime-web` + shows SINRs. If the export fights us, fall back to v1 — we've already got it working.
- **Not pursuing for v1:** Pyodide, TF.js, local backend. If a reviewer specifically wants to see Python code execute, they can run `python test_JSAC.py` locally — that's what the repo is for.
