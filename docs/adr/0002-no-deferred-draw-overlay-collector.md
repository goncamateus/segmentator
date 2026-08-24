# No deferred-draw overlay collector, no dataset tooling

goncanalyser gathers deferred draw callables in `Result.ops` so an overlay can be
composited onto whichever canvas the user picks later, after the fact. Segmentator
deliberately doesn't port this: in a linear `source → stages → sinks` chain,
ordering *is* the composition — `canny` then `harris(draw_on: image)` puts the
corners on the edge map because of stage order, not a deferred callable choosing a
canvas at the end. Porting `Result.ops` would add a mechanism for a degree of
freedom (draw target chosen later) that this architecture doesn't have and isn't
meant to gain; if that need ever shows up, it's a sign the linear-chain assumption
itself needs revisiting, not that this one collector should be bolted back on.

`goncanalyser/dataset/` (COCO export, rosbag, Optuna parameter search, dataset
statistics) is likewise not ported — it's dataset tooling, not image processing,
and out of this project's scope on its face rather than by trade-off.
