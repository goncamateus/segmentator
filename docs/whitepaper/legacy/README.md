# The predecessor, and one change to it

`before/` is this project at commit `48ec930` (2026-08-18): one hardcoded
gas-plume recipe, 217 lines over six files. It is recovered from this
repository's own history, not rewritten.

`after/` is the *same architecture* with the fixed mean-of-N background swapped
for OpenCV's adaptive MOG2 subtractor. That change never actually happened —
the project became segmentator instead — so `after/` is a **reconstruction**:
the smallest edit that makes the predecessor do what `examples/baseline.yaml`
does by saying `type: mog2` instead of `type: mean_background`.

`legacy_diff.py` diffs the two into `generated/legacy-mog2.diff`, the left half
of the paper's comparison figure. The right half is the same change made to a
config, and it is one line.
