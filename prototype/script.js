/* prototype/script.js — page-level orchestration.
 * - Scroll-triggered reveal (respects prefers-reduced-motion).
 * - Simple data cache for components that want to share fetches.
 */

const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

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
// Scroll-reveal — IntersectionObserver, disabled under reduced-motion.
// --------------------------------------------------------------------
function initReveal() {
    const els = document.querySelectorAll('.reveal');
    if (prefersReduced || !('IntersectionObserver' in window)) {
        els.forEach((el) => el.classList.add('is-visible'));
        return;
    }
    const io = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('is-visible');
                    io.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
    );
    els.forEach((el) => io.observe(el));
}

// --------------------------------------------------------------------
// Live readout in the hero — cycle through a few real numbers so the
// "LIVE" chip isn't a lie.
// --------------------------------------------------------------------
function initLiveReadout() {
    const el = document.querySelector('[data-live-ms]');
    if (!el || prefersReduced) return;
    const samples = [58, 62, 55, 71, 64, 49, 67, 53];
    let i = 0;
    setInterval(() => {
        i = (i + 1) % samples.length;
        el.textContent = samples[i];
    }, 2800);
}

// --------------------------------------------------------------------
// Bootstrap
// --------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    initReveal();
    initLiveReadout();
});
