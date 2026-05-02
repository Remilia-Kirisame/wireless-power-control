# Live Run Candidate Interactions

Context for next session: Chapter 06 Live Run is now a cockpit-style component. The draggable map is the main surface, the right rail carries method controls and animated allocation/score summaries, and the bottom drawer carries deeper diagnostics.

Current modified baseline (`v1.1.0`):

- D2D and JSAC share the large-map, right-rail, bottom-drawer layout.
- Chapter 06 has a compact intro so the live map starts higher in the first viewport.
- Right rail and map use the same responsive height; rail content scrolls internally.
- JSAC separates map focus from selected allocation group: clicking a Blue/Rx focuses the map and updates the group gauge; clicking empty map space exits focus while preserving the last group gauge.
- JSAC selected-group allocation gauges animate across method changes and group changes.
- JSAC map markers are compact; allocated power is shown with modest glows rather than large rings.
- D2D and JSAC heatmaps both highlight the corresponding map path on hover.
- D2D and JSAC drag handlers preserve the initial cursor-to-marker offset, so nodes do not jump when the drag starts.
- JSAC GNN trace is presented as message-passing replay. It is not an optimizer iteration; it shows the learned logits after each GNN layer, after per-Blue softmax.
- WMMSE is compacted into a convergence sparkline/checkpoint strip.

Candidate interactions to keep considering:

- WMMSE iteration scrubber that previews checkpoint allocations on the map.
- Method ghosting: selected method solid, other methods faintly visible as ghost power rings.
- Diff mode: show where GNN gives more/less power than WMMSE.
- Constraint-first focus: click a Yellow SINR alert to show the strongest interferers and the method that resolves it best.
- Heatmap-to-map linking in both directions: hover a map interference line to highlight the matrix cell.
- Presentation mode: hide low-level controls and keep only map, method switch, score delta, selected-group allocation, and replay.

Implementation context:

- Main site files are in `web/`; the relevant JSAC live component is `web/components/live-run-lab/live-run-jsac-lab.js`.
- The wrapper `web/components/live-run-lab/live-run-lab.js` still contains the D2D version and imports the JSAC component.
- Version/changelog convention: append each user-visible website change to `Notes_web/change_log.md`, bump semver, and keep `web/index.html` cache-busting query strings aligned.
- Existing animation hooks in the JSAC component include `resultAnimationT`, `displayResults`, method interpolation through `visualPower` / `visualGroupUtil`, selected-group interpolation through `visualAllocationPower`, animated history append mode, replay state through `replayLayerIndex`, and pulsing map interference edges.
