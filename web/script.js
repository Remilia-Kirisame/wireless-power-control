/* web/script.js — page-level orchestration.
 * - Entry overlay animation (title plate + orange load bar, then wipe).
 * - Hash-based tab router across eight views (home + 01..07).
 * - Per-view reveal trigger + sidebar active-link sync.
 * - Shared data cache for components that dedupe fetches.
 */

const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const VIEWS = ['home', 'problem', 'd2d', 'method', 'jsac', 'deepdive', 'liverun', 'refs'];
const VIEW_FADE_MS = 160;       // half-crossfade duration
const ENTRY_DURATION_MS = 2000; // title fade-in + progress fill + wipe total

// --------------------------------------------------------------------
// Site meta — single source of truth for hero + footer byline.
// Edit these three values and both the Home view and the global footer
// pick them up on next load.
// --------------------------------------------------------------------
const SITE_META = {
    author:      'Remilia',
    affiliation: 'SDM',
    date:        'May 2026',
};

// --------------------------------------------------------------------
// Shared data cache — components can read/write here to dedupe fetches.
// --------------------------------------------------------------------
window.__dataCache = window.__dataCache || new Map();

window.fetchJSONCached = async function (url) {
    if (window.__dataCache.has(url)) return window.__dataCache.get(url);
    const promise = fetch(url, { cache: 'no-cache' })
        .then((r) => {
            if (!r.ok) throw new Error(`HTTP ${r.status} — ${url}`);
            return r.json();
        })
        .catch((err) => {
            window.__dataCache.delete(url);
            throw err;
        });
    window.__dataCache.set(url, promise);
    return promise;
};

// --------------------------------------------------------------------
// View router — swap which <section class="view"> is active.
// --------------------------------------------------------------------
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

function revealAll(view) {
    view.querySelectorAll('.reveal').forEach((el) => el.classList.add('is-visible'));
}

function viewIdFromHash() {
    const raw = (location.hash || '#home').replace('#', '').trim();
    return VIEWS.includes(raw) ? raw : 'home';
}

function syncSidebar(activeId) {
    document.querySelectorAll('.tab-link').forEach((link) => {
        link.classList.toggle('is-active', link.dataset.view === activeId);
    });
}

async function switchView(nextId) {
    const next = document.getElementById(nextId);
    if (!next) return;
    const current = document.querySelector('.view.is-active');
    if (current === next) return;

    syncSidebar(nextId);

    if (prefersReduced || !current) {
        current?.classList.remove('is-active', 'is-ready');
        next.classList.add('is-active', 'is-ready');
        revealAll(next);
        window.scrollTo(0, 0);
        return;
    }

    // Phase 1 — fade current out.
    current.classList.remove('is-ready');
    await wait(VIEW_FADE_MS);
    current.classList.remove('is-active');

    // Phase 2 — fade next in. Force a reflow so the opacity transition
    // actually starts from 0 instead of being collapsed by the browser.
    next.classList.add('is-active');
    void next.offsetHeight;
    next.classList.add('is-ready');

    window.scrollTo(0, 0);
    revealAll(next);
}

function initRouter() {
    const initialId = viewIdFromHash();
    const initial = document.getElementById(initialId);
    if (initial) {
        initial.classList.add('is-active', 'is-ready');
        revealAll(initial);
        syncSidebar(initialId);
    }

    window.addEventListener('hashchange', () => switchView(viewIdFromHash()));

    // Any in-page anchor whose target is one of our views routes through
    // the view switcher instead of the default browser jump-scroll.
    document.querySelectorAll('a[href^="#"]').forEach((a) => {
        a.addEventListener('click', (ev) => {
            const target = a.getAttribute('href').slice(1);
            if (!VIEWS.includes(target)) return;
            ev.preventDefault();
            if (location.hash.replace('#', '') === target) {
                // Same hash — hashchange won't fire; drive it manually.
                switchView(target);
            } else {
                location.hash = target;
            }
        });
    });
}

// --------------------------------------------------------------------
// Entry animation — overlay holds briefly, wipes out, body marked loaded.
// --------------------------------------------------------------------
function initEntry() {
    const overlay = document.querySelector('.entry-overlay');
    if (prefersReduced || !overlay) {
        document.body.classList.add('is-loaded');
        overlay?.remove();
        return;
    }
    setTimeout(() => {
        document.body.classList.add('is-loaded');
        overlay.remove();
    }, ENTRY_DURATION_MS);
}

// --------------------------------------------------------------------
// Site meta — fill any [data-site-meta] element from SITE_META.
// `byline` collapses author/affiliation/date into one separated string.
// --------------------------------------------------------------------
function initSiteMeta() {
    const byline = `${SITE_META.author} · ${SITE_META.affiliation} · ${SITE_META.date}`;
    const slots = { ...SITE_META, byline };
    document.querySelectorAll('[data-site-meta]').forEach((el) => {
        const key = el.dataset.siteMeta;
        if (key in slots) el.textContent = slots[key];
    });
}

// --------------------------------------------------------------------
// Live readout in the sidebar — cycle through a few real-ish samples.
// --------------------------------------------------------------------
function initLiveReadout() {
    const els = document.querySelectorAll('[data-live-ms]');
    if (!els.length || prefersReduced) return;
    const samples = [58, 62, 55, 71, 64, 49, 67, 53];
    let i = 0;
    setInterval(() => {
        i = (i + 1) % samples.length;
        els.forEach((el) => (el.textContent = samples[i]));
    }, 2800);
}

// --------------------------------------------------------------------
// Bootstrap.
// --------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    initSiteMeta();
    initEntry();
    initRouter();
    initLiveReadout();
});
