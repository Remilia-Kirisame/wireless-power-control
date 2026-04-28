# Brainstorm - 06 Live Run

This note collects the detailed design ideas for interactive idea **G. ONNX-exported GNN in the browser**. It is intentionally separate from `BRAINSTORM.md` because this feature is large enough to need its own working note.

Navigation change:

- Add a new chapter: **06 Live Run**.
- Rename current **06 References** to **07 References**.
- Home anchor row, sidebar navigation, hash router, and section labels all need to shift from six content tabs to seven.

## Goal

The Live Run chapter should feel like the climax of the showcase. Earlier chapters explain the problem, show the D2D scaling result, and demonstrate JSAC as the application. This chapter lets the visitor draw a wireless layout and watch the learned optimizer run inside the static website.

The pitch:

> Run the learned optimizer in your browser. Draw a wireless layout, move the nodes, and watch the GNN reallocate power in real time.

This is not just another chart. It should make the GNN story tactile: the reader creates a topology, the browser builds the graph, ONNX Runtime Web runs the exported model, and the map plus metrics react immediately.

## Important Method Sets

Keep the baseline naming clear. The two modes do **not** use the same comparison set.

### D2D Mode

D2D compares:

- **WMMSE** - iterative reference solver from `Scenario_D2D/baselines.py`.
- **GNN** - exported D2D IGCNet.
- **Greedy** - the D2D greedy baseline from `Scenario_D2D/baselines.py`, especially `simple_greedy`, which activates the strongest links according to weighted direct-channel strength and a sparsity level estimated from a reference allocation.

Do not label the D2D third baseline as Equal Power. Equal power belongs to JSAC, not D2D.

Possible UI labels:

- `WMMSE`
- `GNN`
- `Greedy`

For the first browser implementation, WMMSE can be handled in one of three ways:

- live JS/WASM implementation for small K only,
- precomputed comparison for built-in presets,
- omitted from the immediate live loop but shown as a "run WMMSE" button with an explicit slower update.

The ideal end state is live GNN plus live Greedy, with WMMSE available for small K or preset layouts so the comparison remains honest.

### JSAC Mode

JSAC compares:

- **WMMSE** - `batch_WMMSE2_JSAC`, including per-group power projection and Yellow SINR dual updates.
- **GNN** - exported JSAC IGCNet plus per-Blue-car softmax in the browser.
- **Naive equal power** - `naive_equal_power`, which splits each Blue car's budget equally across its Yellow and Green receivers.

Possible UI labels:

- `WMMSE`
- `GNN`
- `Naive`

For JSAC, the naive baseline is pedagogically useful because it makes the per-group budget obvious: every link in a Blue cluster starts with the same share, then GNN/WMMSE rebalance across Yellow and Green links.

## Core Screen Concept

The Live Run view should read as a lab, not a static article section.

Suggested layout:

- Left: a large dark interactive field where the user draws or drags the wireless layout.
- Right: live method comparison panel with metrics, method toggles, inference time, and constraint status.
- Bottom: compact diagnostics strip with power bars, SINR/rate chart, channel matrix, or allocation history.
- Top control rail: mode switch, randomize, reset, freeze fading, method toggles, and maybe "compare to WMMSE" for small layouts.

The user should be able to immediately understand:

- what nodes exist,
- which links are intended,
- which links are interference,
- which method is currently visualized,
- how much power each link receives,
- what the resulting sum-rate/SINR/violation metrics are,
- how fast the GNN inference was.

## Interaction Model

### Shared Interactions

Both modes should support:

- drag nodes to reshape the topology,
- randomize layout,
- reset to a clean preset,
- freeze or reshuffle fading/shadowing,
- switch between methods,
- hover/click a link to inspect direct gain, interference gain, power, SINR, and rate,
- keyboard nudging for focused nodes,
- a compact status chip like `ONNX / GNN / 4.8 ms`.

When the layout changes:

1. JS recomputes distances.
2. JS recomputes channel gains using the same formula family as the research code.
3. JS normalizes using exported scaler statistics.
4. JS builds graph tensors.
5. ONNX Runtime Web runs the GNN.
6. JS postprocesses powers.
7. JS computes SINR, rates, Yellow violations where relevant.
8. The map, bars, and metrics animate to the new state.

The interaction should feel continuous, but it does not need to run inference on every mousemove. A good implementation can debounce or throttle model calls, while still updating visual drag positions immediately.

### D2D Drawing

D2D mode should be the first implementation because it is cleaner.

User actions:

- click or button to add a Tx-Rx pair,
- drag Tx and Rx independently,
- remove a selected pair,
- randomize K pairs,
- choose K from a small supported range,
- optionally toggle an interference threshold ring.

Visuals:

- each pair gets a numbered Tx and Rx,
- intended Tx-Rx links are orange,
- interference lines are grey/blue and opacity-scaled by harmfulness,
- allocated power appears as a ring around Tx or glow around Rx,
- power bars show $p_1, p_2, \ldots, p_K$,
- sum-rate and per-link rate update live.

What it teaches:

- moving one transmitter can change another receiver's SINR,
- turning every link up is often bad,
- Greedy can shut off weaker links,
- GNN learns a smoother allocation than Greedy,
- WMMSE remains the reference but is slower.

The strong demo moment: place two transmitters close to one victim receiver and watch the GNN reduce power on the harmful link while preserving total sum-rate.

### JSAC Drawing

JSAC mode should be the richer stretch after D2D.

User actions:

- add or remove Blue-car clusters,
- drag a Blue car to move its whole cluster,
- drag individual Yellow or Green receivers,
- adjust $M_y$ and $M_g$ within a small supported range,
- toggle or reshuffle channel assignment,
- reset to realistic highway-like presets.

Visuals:

- Blue nodes are transmitters,
- Yellow nodes are sensing receivers,
- Green nodes are communication receivers,
- each Blue cluster has a budget meter,
- intra-cluster links are solid and colored by receiver type,
- same-channel inter-cluster interference is dashed grey,
- Yellow nodes show a warning outline when SINR is below target,
- Green sum-rate and Yellow violation are displayed side by side.

What it teaches:

- JSAC is not just D2D with colors; it has group budgets and sensing constraints,
- naive equal power is simple and stable but leaves performance on the table,
- GNN reallocates within each Blue car while preserving the group budget,
- Yellow constraints can force power away from Green links,
- WMMSE is slower and iterative, while GNN is one forward pass.

The strong demo moment: drag a Yellow receiver into a difficult location and watch the GNN shift more of that Blue car's budget toward sensing, while the Green sum-rate panel reacts.

## Animation And Visual Feel

This feature should be animated, but the motion should explain the computation.

Useful animated states:

- power rings expand/contract when allocation changes,
- metric numbers count smoothly to the new value,
- bad Yellow SINR states get a restrained warning pulse,
- interference lines brighten only when they are currently influential,
- method switching crossfades map powers and bars,
- dragging a node temporarily leaves a faint trace or ghost line to show before/after.

Highest-impact idea: expose intermediate GNN layer outputs.

If feasible, export or reproduce the four message-passing iterations so the UI can animate:

1. edges pulse as messages flow,
2. node logits update after layer 1,
3. allocations sharpen after layers 2 and 3,
4. final power settles after layer 4.

This would make the GNN feel alive without faking the math. The user sees information flowing through the graph before the final allocation appears.

Respect `prefers-reduced-motion`: all pulses, transitions, and animated counting should become instant state updates.

## Technical Architecture

Expected browser pipeline:

1. Layout editor produces Tx/Rx or Blue/Yellow/Green coordinates.
2. Channel builder computes distance, path loss, shadowing/fading, direct gains, and cross gains.
3. Normalizer applies exported scaler statistics from the training pipeline.
4. Graph builder creates node features, edge indices or dense masks, and edge features.
5. ONNX Runtime Web runs the exported model.
6. Postprocessor converts model output into method powers.
7. Metrics module computes SINR, per-link rates, Green sum-rate, Yellow violations, and budget utilization.
8. Renderer updates SVG/canvas visual state.

Important export detail: PyG `MessagePassing` may be awkward to export directly. A web-friendly wrapper is probably safer.

Recommended approach:

- Rebuild the trained GNN as a dense-mask Torch module for export.
- Use the same learned MLP weights.
- Compute pairwise messages over all node pairs.
- Mask invalid edges.
- Use max-reduction over incoming messages.
- Run the same unrolled IGConv iterations.

For browser demo sizes, dense $O(K^2)$ message computation is acceptable and much easier to export reliably than PyG scatter operations.

For D2D:

- node features: normalized direct distance, normalized direct channel gain, power placeholder,
- edge features: normalized interference distance,
- output: per-link normalized power via sigmoid,
- method outputs: WMMSE, GNN, Greedy.

For JSAC:

- node features: normalized direct distance, normalized direct channel gain, is-Green flag, power placeholder,
- edge features: normalized distance, edge type, normalized channel gain,
- output: raw logits,
- postprocess: per-Blue-car softmax,
- method outputs: WMMSE, GNN, Naive equal power.

## Three-Stage Plan

### Stage 1 - D2D Live Run MVP

Goal: prove the browser-inference loop.

Build:

- new `06 Live Run` view,
- D2D layout editor,
- D2D channel builder,
- D2D scaler export,
- ONNX-exported D2D GNN,
- live GNN powers,
- live Greedy powers,
- WMMSE comparison for presets or small K,
- sum-rate, per-link rates, SINR, and inference-time panel.

Why first:

- D2D graph is simpler,
- no group softmax,
- no Yellow/Green masks,
- the visual story directly extends the D2D scaling chapter.

### Stage 2 - JSAC Live Run

Goal: bring the capstone application into the same live lab.

Build:

- mode switch from D2D to JSAC,
- Blue/Yellow/Green layout editor,
- JSAC channel builder and metadata builder,
- JSAC scaler export,
- ONNX-exported JSAC GNN,
- per-Blue-car softmax postprocess,
- Naive equal-power baseline,
- WMMSE comparison for presets or small supported layouts,
- Green sum-rate, Yellow SINR violation, and per-group budget meters.

Why second:

- JSAC has more state and more constraints,
- the interaction is richer only after the inference loop is stable,
- it requires careful explanation so the user does not confuse equal power with the D2D baseline.

### Stage 3 - Polish Layer

Goal: make it memorable for a defense/demo.

Build:

- message-passing pulse animation,
- layer-by-layer GNN state visualization,
- presets like "crowded corner", "hidden terminal", "sensing stress", and "balanced highway",
- saved example layouts,
- WMMSE iteration overlay for small layouts,
- comparison history strip showing how metrics changed as the user dragged nodes,
- keyboard shortcuts and refined accessibility states,
- local performance optimizations with Web Workers if needed.

## Risks And Design Constraints

Main technical risks:

- ONNX export may not handle PyG `MessagePassing` directly.
- Dynamic graph shapes can be painful in browser inference.
- JSAC WMMSE may be too slow to run live in JavaScript for arbitrary layouts.
- Browser performance can degrade if every drag event triggers full channel rebuild plus inference.
- The UI can become visually noisy if all interference lines are shown at once.

Mitigations:

- use a dense-mask export wrapper,
- support a small maximum K in Live Run even though the GNN concept is size-agnostic,
- throttle inference while dragging,
- run expensive comparisons on button press rather than continuously,
- let users isolate a link, cluster, or channel,
- keep WMMSE live only for small layouts or preset comparisons,
- make the method set explicit in the UI for each mode.

## Ready-To-Build Checklist

Before implementation, confirm these items:

- Which D2D trained checkpoint should be exported?
- Which JSAC trained checkpoint should be exported?

A: Pick as you believe is appropriate. You may also run the main files with `ENABLE_SAVING=True` to get newest trained checkpoint.

- What maximum live K is acceptable for browser performance?

  A: lets do 20 for now.

- Should WMMSE run live, only on demand, or only for presets in Stage 1?

  A: Run on live.

- Should the first Stage 1 UI ship with D2D only, with JSAC tab visibly marked "coming next"?

  A: Yes.

- Where should ONNX files live? Likely `prototype/assets/models/`, with a small JSON manifest.

  A: Up to you where you believe works best with the website structure.

- Should `PROMPT_WEB.md` be updated after the Live Run feature is built, or before implementation as a revised spec?

  A: No, this note `BRAINSTORM_LIVERUN.md` is the only note to be synced with our ideas.

## Current Recommendation

Proceed with the three-stage plan.

Stage 1 should build a D2D-only Live Run MVP with method comparison explicitly labeled **WMMSE / GNN / Greedy**. Once that feels good and the ONNX path is proven, Stage 2 should add JSAC with **WMMSE / GNN / Naive**. Stage 3 should focus on polish, especially message-passing animation and demo presets.

## Stage 1 Implementation Status

Stage 1 is now implemented in `prototype/`.

Files added or changed:

- `prototype/components/live-run-lab/live-run-lab.js` - D2D browser lab component.
- `Scenario_D2D/export_live_run.py` - dense-mask D2D GNN export helper.
- `prototype/assets/models/d2d_igcnet_k20.onnx` - exported D2D GNN for `K <= 20`.
- `prototype/assets/models/d2d_live_manifest.json` - model, scaler, and physics manifest.
- `prototype/assets/models/d2d_igcnet_k20_weights.json` - JS fallback weights.
- `prototype/assets/vendor/onnxruntime-web/` - minimal ONNX Runtime Web WASM runtime.
- `prototype/index.html` and `prototype/script.js` - new `06 Live Run` route; References moved to `07`.

Stage 1 shipped behavior:

- D2D-only live editor.
- K selector from 2 to 20.
- Add/remove pair.
- Randomize layout.
- Freeze or reshuffle fading.
- Live ONNX GNN inference in the browser.
- Live JS WMMSE.
- Live JS Greedy.
- Method toggle: WMMSE / GNN / Greedy.
- Sum-rate, active-link count, and method runtime readouts.
- Power allocation gauges.
- Channel matrix heatmap.
- JSAC tab is visible but disabled as "coming next".

Post-implementation fixes:

- K changes and Add Pair used to leave the map blank until Shuffle Fading was clicked. Cause: stale previous-K channel/results state was reused during the immediate redraw before the new compute ran. Fix: topology changes now clear stale `last` and `results`, invalidate pending compute tickets, redraw safely, and then recompute.
- Randomize after a K change now works for the same reason.
- Power gauges were not visibly filled because the fill element was inline. Fix: the fill is now block-level, method-colored, and the right-side readout shows normalized power instead of link rate.

Verified:

- `node --check prototype/components/live-run-lab/live-run-lab.js`
- `node --check prototype/script.js`
- `python -m py_compile Scenario_D2D/export_live_run.py`
- Browser smoke test at `http://127.0.0.1:8000/#liverun`.
- Browser state reaches `ONNX / GNN`.
- Add Pair from K=8 to K=9 keeps the map populated and adds Receiver/Transmitter 8.
- Randomize works after K changes.
- Power gauges visibly fill.

## Stage 2 Handoff - JSAC Live Run

Stage 2 should add JSAC mode inside the same Live Run component or as a sibling component behind the existing disabled tab.

Method set must be:

- **WMMSE** - `Scenario_JSAC/baselines.py:batch_WMMSE2_JSAC`
- **GNN** - exported JSAC IGCNet plus browser-side per-Blue-car softmax
- **Naive** - `Scenario_JSAC/baselines.py:naive_equal_power`

Do not call the JSAC baseline "Greedy", and do not introduce Equal/Naive into D2D.

Suggested Stage 2 build order:

1. Export the JSAC GNN with the same dense-mask strategy used for D2D. The JSAC wrapper needs 4D node features, 3D edge features, raw logits, and `Kmax` padding.
2. Export a JSAC manifest with scaler stats, field settings, `sinr_min`, `Pmax`, default `B`, `M_y`, `M_g`, `group_ids`, `green_mask`, and `yellow_mask` conventions.
3. Add a JSAC layout editor: Blue Tx clusters, Yellow sensing Rx, Green communication Rx, and drag behavior.
4. Build browser metadata from the layout: same-channel inter-Blue interference mask, intra-group edges, `group_ids`, `green_mask`, and `yellow_mask`.
5. Run ONNX JSAC GNN, then apply per-Blue-car softmax in JavaScript.
6. Implement Naive equal power as a simple browser baseline.
7. Port `batch_WMMSE2_JSAC` to JavaScript for small supported layouts or run it on demand if it feels too heavy.
8. Render JSAC-specific readouts: Green sum-rate, Yellow SINR violation rate/count, min Yellow SINR, and per-Blue budget utilization.
9. Add JSAC-specific visuals: Blue budget rings, Yellow warning outlines, Green/Yellow link colors, and dashed same-channel interference.
10. Keep the D2D Stage 1 behavior unchanged while adding JSAC.

Stage 2 UI should preserve the current Live Run shell:

- same top mode toggle,
- same left map / right metrics layout,
- same diagnostics strip,
- D2D remains WMMSE / GNN / Greedy,
- JSAC becomes WMMSE / GNN / Naive.
