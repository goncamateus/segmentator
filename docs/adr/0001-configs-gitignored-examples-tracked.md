# Local configs are gitignored; runnable samples live in `examples/`

`configs/` was untracked in `3812eee` ("Local config files vary per run/environment
and don't belong in version control") and stays gitignored — it's where a user's
own per-run YAML lives, and committing those would mean constant unrelated diffs as
paths/params change run to run.

That decision left every doc's quickstart command pointing at `configs/*.yaml`
files that don't exist in a fresh clone. `examples/` is the fix: a small,
*tracked* set of runnable sample configs, kept deliberately separate from
`configs/` so the "don't commit your local run config" rule doesn't get relitigated
every time someone wants a working example to start from. If a reader only sees
`configs/` in `.gitignore` and wonders where the README's example files went, this
is why — they moved to `examples/`, not back into version control.
