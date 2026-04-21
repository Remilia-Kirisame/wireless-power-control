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
