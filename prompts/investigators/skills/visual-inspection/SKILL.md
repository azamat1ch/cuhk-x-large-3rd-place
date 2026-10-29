---
name: visual-inspection
description: Plan chronological coverage, targeted dense views, crops and source comparisons; diagnose missing detail without inventing pixels.
---
# Inspect to separate a concrete choice

1. Read all questions, then describe available sources. Compare durations, field of
   view and native timing. IR is a useful first overview, not an authority ranking.
   If it ends early, inspect the longer permitted source for missing late events.
   A successful decode does not prove the whole routine is present.
2. Cover the recording chronologically, including first and last decoded frames.
   Choose overview sampling for its duration and likely action lengths. Sparse
   samples localize broad episodes; they cannot exclude a brief wrist check or pickup.
3. Keep a compact ledger of covered/uninspected intervals, observable features,
   candidate actions, onset/offset uncertainty and delivered references. Repeated
   views of one event do not create additional events.
4. State what would separate the closest alternatives before requesting detail:
   page crossing versus stationary-paper wiping; bottle tipping versus reaching;
   fingers on a broad base versus on a hand-sized upright device. Widen for scale
   and hand destination; densify for onset, short gestures and disputed order.
   Separate supporting motion, contradictory motion and merely hidden detail.
5. Inspect before, during and after transitions. A utensil seen in use does not
   locate its pickup; already sitting does not establish a sit-down transition.
6. Reconcile answers and stop when the decisive distinction is supported, or when
   remaining uncertainty is occlusion/naming rather than inspectable information.
   Repeating identical views adds no evidence. There is no prescribed number of
   views or frames that makes an investigation sufficient.

View delivery may page a selection into groups of 24 frames. This is a transport
page size, not a sampling limit or an instruction to discard the rest. Follow the
tool's returned paging information until the intended interval is covered. Use
`fps=None` for every native frame in a selected interval when a brief event needs
it; use `timestamps_s` for specific revisits. Prefer short dense intervals over
needlessly dense whole recordings, but do not impose a fixed evidence cap.

Read [detail-and-alignment](references/detail-and-alignment.md) when dark detail,
small geometry, source meaning/limits, alignment or apparent rhythm matters.

Track both hands and their target together. For tiny hand/object motion, use
`layout="detail"`, a crop and `zoom=4` before treating it as unresolvable.
After a fresh pickup, reassess the item and manipulation from visible evidence;
an earlier utensil interpretation does not automatically carry into the new episode.
