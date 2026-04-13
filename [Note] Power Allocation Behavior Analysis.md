# Power Allocation Behavior Analysis

Observations from Step 5 smoke test (B=10, M_y=2, M_g=3, field=225, no channel shuffling).

## Empirical Results

```
Method          Yellow avg  Yellow std   Green avg   Green std  Group total
----------------------------------------------------------------------
Equal Power       0.2000      0.0000      0.2000      0.0000       1.0000
WMMSE             0.1271      0.1606      0.1021      0.1387       0.5606
GNN               0.4413      0.3250      0.0391      0.0510       1.0000
```

Performance was comparable: GNN/WMMSE = 97.0%, violation 18.1% vs 19.6%.

## Key System Context

Two properties of the current JSAC setup are critical to understanding these results:

1. **Interference-limited regime.** Noise is at -169 dBm/Hz with 5 MHz bandwidth, so the noise floor is extremely low. SINR effectively equals SIR (signal-to-interference ratio). In SIR, scaling all powers within a group by the same constant changes nothing — numerator and denominator scale together.

2. **Disjoint Yellow/Green channel sets (no shuffle).** With fixed channel ordering (`channel_id = link_index mod M`), Yellow links always share channels with other Yellow links across Blue cars, and Green with Green. Yellow power from one Blue car creates zero interference on Green links from another. The power split between Yellow and Green within a group has no cross-effect on each other's SIR.

## Observation 1: WMMSE Uses Only ~56% of Power Budget

**Why:** WMMSE's iterative update equations compute optimal power levels. In the interference-limited regime, the rate function `log(1 + SIR)` has near-zero gradient with respect to uniform power scaling — if you double every link's power, SIR stays the same. The per-group power constraint (sum ≤ Pmax) is never binding because there's no incentive to push power upward. The algorithm converges to a solution where the *relative allocations* are optimized but the absolute scale settles wherever the update equations naturally land.

This is mathematically correct behavior — WMMSE correctly identifies that the power constraint is slack and doesn't waste effort saturating it.

**Why GNN always uses full budget:** The GNN uses per-group softmax as its output layer, which structurally constrains each group's power to sum to exactly 1.0 (= Pmax normalized). It cannot produce a solution that uses less than the full budget — this is a hard architectural constraint, not a learned behavior.

**Implication:** In the interference-limited regime, this difference is benign — absolute power doesn't matter, only relative allocations. But if we move to a regime where noise matters (e.g., lower Tx power, higher noise, or longer distances), WMMSE would naturally start using more of the budget while the GNN would already be at full budget by construction. The GNN's softmax might actually be beneficial in that regime.

## Observation 2: GNN Allocates Most Power to Yellow (Sensing)

The GNN gives Yellow links ~44% of per-link power vs Green's ~4%, despite Yellow being only 2/5 of the links per group (expected equal share: 20% per link).

**Why:** The GNN's loss function has two terms:
- **Green sum-rate** (objective): maximize `sum(log(1 + SIR_green))`
- **Yellow SINR penalty** (constraint): penalize `max(0, sinr_min - SIR_yellow)^2`

Under the two system properties above:

1. The Yellow SINR penalty creates a **strong gradient** to increase Yellow power — directly reduces violation.
2. Green SIR is determined by the *ratio among Green links' powers*, not their absolute values — because SIR and because Green links don't interfere with Yellow links (disjoint channels).
3. The softmax forces a budget split: more Yellow → less Green in absolute terms.
4. But reducing absolute Green power while keeping the same relative Green allocation doesn't change Green SIR at all.

So the GNN finds the path of least resistance:
- Give Yellow a lot → strongly satisfies sensing constraint (low violation)
- Give Green a little → doesn't hurt Green SIR (rates preserved)

This is actually a **correct optimization** under current assumptions. The 97% WMMSE ratio and 18.1% violation rate confirm that the GNN's aggressive Yellow allocation doesn't sacrifice Green performance.

**Why WMMSE doesn't show this pattern:** WMMSE uses `alpha` weights (0.1 for Yellow, 1.0 for Green) plus a Lagrange multiplier for the SINR constraint. The alpha weights suppress Yellow's contribution to the objective, while the dual variable provides just enough Yellow power boost to (partially) satisfy constraints. WMMSE reaches a more "balanced" allocation because it doesn't have the softmax structural constraint — it can (and does) use less total power, distributing it more evenly.

**Why Equal Power doesn't show this pattern:** It has no optimization at all — flat 0.20 per link by definition.

## When This Behavior Would Break

The current power allocation patterns are correct under the current assumptions but would change or break if:

| Change | Effect on WMMSE budget | Effect on GNN Yellow bias |
|---|---|---|
| **Higher noise / lower Tx power** (noise-limited regime) | Would start using more budget — absolute power now affects SNR | Green would need more absolute power too — Yellow bias would hurt Green rates |
| **Channel shuffling enabled** | No direct effect on budget | Yellow/Green cross-interference appears — Yellow power from Blue_a hurts Green SIR at Blue_b — Yellow bias would increase Green violations |
| **Triple path-loss model** (d^4 for sensing) | May shift regime for sensing links | Sensing links become much weaker — may need even more Yellow power, or may change the SIR balance |
| **Larger field / fewer Blue cars** | Lower interference → more noise-limited | Same as higher noise |

## Potential Future Fixes

These are not needed now (performance is good) but should be revisited if the system regime changes:

1. **For WMMSE under-utilization:** Not really a problem — it's mathematically correct. If we want WMMSE to use full budget for fair comparison, we could post-hoc scale each group's allocation to sum to Pmax. But this would only matter in a noise-limited regime.

2. **For GNN Yellow bias:** If we move to a regime where absolute power matters:
   - Option A: Add a regularization term encouraging more balanced Yellow/Green split
   - Option B: Replace per-group softmax with separate softmax for Yellow and Green subgroups within each Blue car (guarantees each subgroup gets a fair share of the budget)
   - Option C: Use sigmoid + per-group projection (Option A from WORKPLAN Step 4C) which allows the GNN to learn the budget split rather than forcing full utilization

3. **For fair comparison across methods:** Normalize all methods to use the same total power per group before computing metrics. This isolates the effect of *relative allocation quality* from *budget utilization*.
