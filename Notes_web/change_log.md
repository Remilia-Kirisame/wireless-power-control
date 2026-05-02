version 1 `prototype` done, and renamed to `web/`. that's v1.0.0.

future changes logged below.

- **v1.0.1** — sidebar collapse toggle. 
  corner chevron slides the side rail off-screen so the main content reclaims the freed width via a synced animation; state persists across reloads. mobile (top-bar mode) is unaffected.
- **v1.0.2** — Live Run map legends. 
  D2D and JSAC maps now label the scenario-specific nodes, links, interference traces, and constraint/power cues in the top-right corner of the map.
- **v1.0.3** — Live Run grouped controls and safer presets.
  Reworked the layout of the map controls (droplist and buttons) for user friendly interaction. Also improved the preset to layouts that GNN works well.

---

- **v1.1.0** — Live Run cockpit refinement.
  compact intro, map-aligned rail, animated JSAC selected-group allocation, click-empty focus exit, D2D/JSAC heatmap linking, compact JSAC power glows, and compact solver drawer.
- **v1.1.1** — Live Run normalized demo latency.
  Latency pills are normalized for demonstration: WMMSE/Greedy are live JS elapsed, while GNN is mapped from live WMMSE time using D2D benchmark ratios. Raw browser time is available in the tooltip; this is not a fair runtime benchmark.
