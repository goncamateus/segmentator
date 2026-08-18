# The editor

**What this is.** A PyQt6 window for writing one of these YAML configs while
watching what it does, frame by frame. It is the same engine — the same stage
classes, the same `frame_for`, the same numbers — driven one frame at a time
instead of over a batch.

```bash
uv sync --extra gui
uv run segmentator-gui configs/motion.yaml
uv run segmentator-gui                       # opens a file picker
```

The config is a linear list of stages, so **the list is the graph**, drawn at 1:1
rather than as a straight line of boxes on a canvas. Where a config does branch
it branches by *name* — a `select` stage, or a sink reading a tap — and the
editor shows that as a named reference rather than as a second edge. The
reasoning behind that choice, and what a node canvas would have cost instead, is
in [Why a list and not a canvas](#why-a-list-and-not-a-canvas).

![The editor: stage list, generated form, preview](assets/gui-window.svg)

That is not a sketch of the intended design — it is what the window renders. The
editor forces one of its own two palettes and the Fusion style rather than
inheriting the desktop theme. The amber marks are the reason it is worth pinning
down: a `name:` tap and a stage that carries state between frames are the two
things you scan the list for while tuning, and a theme that recolours them is a
theme that hides them.

### Day and night

The `☀` in the corner of the menu bar switches; the glyph is the theme you are
in, so `☾` means night. The choice is remembered between sessions.

![The same editor at night](assets/gui-window-dark.svg)

Same layout, same meanings. Two roles do not simply invert, because a palette
that flips every channel mechanically loses the things the light one was chosen
for: the blue lightens, since `#1f6feb` on a dark panel reads as a hole rather
than a highlight, and the amber pill becomes a dark tint behind the same amber
border — a tap has to stay recognisably a tap. The video in the preview is never
themed. It is the footage, not the furniture.

## The three panels

**Left — the pipeline.** The stages in the order they run, then the sinks. An
amber dot marks a stage that remembers something between frames; a pill marks a
`name:` tap. Both matter while tuning, which is why they are on the list rather
than buried in the form.

**Middle — the parameters.** Generated from `inspect.signature` of the
registered class. There is no per-stage form to maintain: adding a stage to
`segmentator/stages/` adds its form here, with the right widget per type and the
constructor's defaults already in place.

**Right — the preview.** One tab per thing worth looking at: the selected stage,
the untouched source, and every image sink. Under it, `ctx.metrics` and the row
counts — what the `json` and `csv` sinks would have written for this frame.

## Opening and saving

`File ▸ Open` reads a config, `Ctrl+S` writes it back, `Ctrl+Shift+S` writes it
somewhere else. The document is read and written by ruamel's round-trip parser,
not by `yaml.safe_load`, and the difference is the point: `configs/*.yaml` are
heavily commented on purpose, and a save has to give those comments back.

![What a save does to a hand-written config](assets/gui-save.svg)

A `git diff` after a save shows the lines you changed and nothing else — key
order, quoting, flow style and blank lines all survive. The one thing that does
not is a comment block between two stages when you *reorder* one of them: it
belongs to whichever stage ruamel attached it to, and moves when that stage
moves.

## Adding, removing and reordering stages

![Adding, reordering and tapping a stage](assets/gui-stages.svg)

`+` opens the palette — half the main window's width, and laid out as a grid of
family cards (the same twelve `docs/stages.md` groups into: Adjust, Blur /
morphology, Motion — models, and so on) rather than one long combo. Every name
in it is `registered("stage")` or `registered("sink")` — 41 stages, 3 sources, 6
sinks, listed because they registered themselves, not because anyone typed them
here. Each card wears its family's colour from the
[catalogue](stages.md) — blue to prepare a frame, purple to find features in it,
green to mask it, orange to measure what moved — and lists its stages in the
same monospace the catalogue sets them in, since they are `type:` values to be
typed into a config rather than prose. A stage that carries state between frames
shows the same amber dot the main list marks it with, explained by a legend
under the cards; a card with more than three stages scrolls rather than growing,
so every card is the same size whatever is filed under it. A filter box above
them narrows every card at once, since a grid is still a lot to scan by eye for
one specific name. Clicking
a row selects it — across cards, only one row is ever selected — and a
double-click accepts immediately. The new stage is inserted after the selected
one and written out with only its *required* parameters; everything else is a
default, and a default stays out of the file.

Reorder by dragging inside the list, or with `Ctrl+↑` / `Ctrl+↓`. Order is
composition: `canny` then `harris(draw_on: image)` puts the corners on the edge
map, and the same two the other way round does not.

`−` (or `Ctrl+D`) removes whichever list has the focus.

## Editing a parameter

![How the form is generated, and how live knobs are told from construction ones](assets/gui-params.svg)

Each row writes its key into the YAML only when it differs from the constructor
default, and deletes the key when it goes back — so a terse config stays terse
instead of growing every default the first time it is clicked.

Some labels are amber. Those are **construction parameters**: values the stage
consumes when it is built and does not keep — `mog2`'s `history` disappears into
an OpenCV subtractor, `frame_diff`'s `lag` into a deque's `maxlen`. They cannot
be assigned to a running stage, so moving one rebuilds it, and a stage that had
learnt a background starts over. Everything else is assigned live, and a paused
frame simply re-renders as you drag.

Nothing in the codebase declares which is which. The editor asks the built
instance whether it still carries an attribute of that name, because a parameter
a stage needs at run time is a parameter it kept. The single exception is
`StaticMask`, which stores `threshold` but caches the mask it fitted from it, and
says so with one line: `RECONSTRUCT = ("threshold", "invert")`.

## Naming a stage

The last row of every stage's form is `name (tap)`. Filling it in publishes that
stage's output under that name, which is what makes it reachable — by a sink's
`input:`, by a `select` stage, and by the `draw_on:` of anything that paints.
Taps are opt-in; an unnamed stage costs nothing.

> **Gotcha, and the editor cannot save you from it:** a tap holds `ctx.image`
> *after* the stage ran. `static_mask` publishes a mask into `ctx.store` but
> leaves the frame alone, so naming it taps its *input* image. Reach the mask
> itself with `input: mask`, which the dropdown offers.

## Sinks

Sinks are edited with the same generated form — `path`, `input`, `fps`, `dir`,
whatever that sink's constructor takes. `input:` is a dropdown of every key that
currently resolves.

**Sinks are shown, never run.** The editor builds no sink object and opens no
file. What a sink's tab shows is its `input:` resolved against the current
frame, which is the whole of what a `display` or `ffmpeg` sink would do with it.
Nothing is encoded, nothing is written, and pointing a sink at
`outputs/final.mp4` while you tune cannot truncate yesterday's render.

## Previewing

![What each preview tab is looking at](assets/gui-preview.svg)

Every stage is previewable whether the config named it or not: the editor taps
each position as `#n` as well as by name. `#2` exists for the editor; `mask`
exists for the sinks and the rest of the config.

A key resolves exactly as it does in a batch run — `image`, then `source`, then
a named tap, then any image left in `ctx.store` — because it is the same
`frame_for()` call. That is what makes the preview worth trusting: the tab is
not a rendering of what the sink would write, it *is* what the sink would write.

Only the visible tab is converted to a picture. Nothing off-screen is ever
turned into a `QImage`, which is why there is no thumbnail throttle to tune.

## Moving through the video

`◀◀ ◀ ▶ ▶▶` step ten frames back, one back, play/pause, one forward; the slider
seeks. Everything expensive — decoding, the chain, the conversion — happens on a
worker thread, so dragging never competes with a 200 ms HOG for the main loop.

Tuning while paused is the case worth understanding, because it is the one where
a preview can quietly lie:

![What happens when a knob moves while the preview is paused](assets/gui-transport.svg)

Three rules, all of them goncanalyser's, restated for a chain whose state lives
on the stage instances:

1. **The same frame must not go through a stateful stage twice.** Feeding MOG2
   the frame on screen fifty times while you drag teaches it that the plume *is*
   the background, and it fades out as you work. The editor caches each stage's
   output for the current frame and re-runs only from the edit down, so a model
   above the knob you are turning is never re-fed.
2. **A jump resets.** After a seek the previous frame is no longer the previous
   frame, and differencing across it would light up the whole image.
3. **A changed model resets.** See the amber labels above.

After a reset, the eight frames before the current one are replayed through the
chain — but only as far as the last stage that remembers anything, since
everything below that is a pure function of the frame it is handed. Without the
replay, a rebuilt `farneback` would paint black under your hands.

The status bar names what reset, because eight frames warms a frame-differencer
and does nothing at all for `mog2 history: 500`, and a number you cannot trust is
worse than one that says so:

```
frame 1232/22875   320x240   97 ms   paused   reset: motion_objects
```

The millisecond figure is the whole cost of the edit, replay included.

## Running the batch

The editor does not. `File ▸ Copy run command` puts

```bash
uv run segmentator configs/motion.yaml
```

on the clipboard, and the run happens where runs belong — in a terminal, with a
progress line, restartable, and with the ffmpeg encoder writing at full rate
instead of competing with a window. The editor's job is the recipe; the CLI's job
is the batch.

## Keyboard

| | |
|---|---|
| `Ctrl+O` / `Ctrl+S` / `Ctrl+Shift+S` | open, save, save as |
| `Ctrl+N` | add a stage |
| `Ctrl+D` | remove the selected stage or sink |
| `Ctrl+↑` / `Ctrl+↓` | move the selected stage |
| `Ctrl+Q` | quit |

## Why a list and not a canvas

A node canvas over this config format would draw a mostly straight line. Every
stage has exactly one input — `Pipeline.run` carries a single `ctx.image` down an
ordered list — so the graph is a *tree*: one producer, many consumers, which
flattens to a list plus taps. The gesture a node editor exists for, dragging a
second wire into a node, is precisely the one the format cannot express, and an
editor whose central gesture has to be refused is worse than no canvas at all.

If merges ever become expressible — stages taking an `input:` too, named buffers
instead of one `ctx.image`, a topological sort — a canvas stops being decoration.
Until then the list is the honest drawing of what the file is, and it is the
drawing that costs a third as much.

## Troubleshooting

**The window will not start: "could not load the Qt platform plugin xcb".**
Importing `cv2` points `QT_QPA_PLATFORM_PLUGIN_PATH` at the Qt5 plugins the
opencv-python wheel ships, and PyQt6 then refuses to load its own. The entry
point clears that variable before Qt starts; if you are launching the window
from your own script, do the same:

```python
import os
import cv2  # noqa: F401  — must be imported before Qt
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
```

**"no frames in …".** The source path in the config is relative to where the
editor was launched, exactly as it is for a batch run.

**The window ignores my desktop theme.** By design — it has its own two, and the
`☀` in the menu bar's corner picks between them. What the desktop is set to
never enters into it. If you are constructing `MainWindow` yourself rather than
running `segmentator-gui`, call `style.apply(app, theme)` before building it, or
you get whatever the platform style hands you.

**A stage raises while I am typing.** Expected, and not fatal — the status bar
shows the exception and keeps the last good frame. `farneback` on a colour frame
is the usual one; it wants a `gray` above it.

## Where the code is

| | |
|---|---|
| [gui/spec.py](../segmentator/gui/spec.py) | signatures, the live-vs-rebuild rule, YAML round-trip. No Qt. |
| [gui/style.py](../segmentator/gui/style.py) | both palettes, as a stylesheet and a `QPalette` |
| [gui/worker.py](../segmentator/gui/worker.py) | the preview thread: prefix cache, the three rules, warm-up replay |
| [gui/window.py](../segmentator/gui/window.py) | the window and every operation above |
| [gui/main.py](../segmentator/gui/main.py) | entry point, and the Qt plugin fix |
| [tests/test_gui.py](../tests/test_gui.py) | runs headless: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_gui.py` |
