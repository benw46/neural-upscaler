# Spec 3 of 4: Temporal Model

Prerequisite: Spec 2 gate passed. Read `CLAUDE.md` first.

**Goal:** integrate frame history so output is stable and detailed across
long sequences, not just accurate per-frame. This is the hardest and least
predictable phase — treat it as an experiment log, not a fixed build.

---

1. Feed in the previous frame's *output*, warped into the current frame by
   motion vectors (reuse the Spec 1 reprojection logic).
2. Disocclusion mask: compare reprojected depth against current depth; feed
   the mask to the network as a signal to fall back to spatial-only
   reconstruction where history is invalid.
3. Recurrent training on short sequences (4–8 frames), backpropagating
   through the recurrence.
4. Temporal consistency loss term: penalise frame-to-frame difference in
   regions motion vectors mark as static.
5. **Treat loss weighting as empirical, not derivable.** Log every
   configuration tried in `/notebook` — including ones that were discarded —
   with what was observed and why it was rejected. Don't just report the
   final config; the discarded attempts are as informative as the winner.
6. Explicitly test for and report on: degenerate "just copy history"
   solutions, feedback-loop brightness drift or artifact accumulation over
   long sequences, and ghosting at disocclusions.
7. Test on sequences of several hundred frames, not just short clips — drift
   and instability often only appear over length.

---

## Gate

- Stable over 300+ frame sequences: no brightness drift, no accumulating
  artifacts. Show a long-sequence result, not just a short clip.
- The `/notebook` log of attempted configurations exists and is legible —
  this phase's real output is as much the log as the model.

Write the `/docs` summary before stopping, including a short narrative of
what failed along the way, not just the final working config.
