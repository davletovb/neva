# Historical reviewer measurements — not independent R0 acceptance

Source: [issue #103 review, comment 6259171743](https://github.com/davletovb/hiyal/issues/103#issuecomment-6259171743),
reported against historical Python revision
[d0bb81b](https://github.com/davletovb/neva/commit/d0bb81b42efbf1658ada63ccceafba9095b12192).

| Actors | Observer on | Observer off |
| ---: | ---: | ---: |
| 20 | 69–92 µs/step | 10–15 µs/step |
| 100 | 142–165 µs/step | 11–14 µs/step |
| 1,000 | 1.9–2.6 ms/step | 11–12 µs/step |

The reviewer also reported approximately 233 MB retained-memory growth over
1,000 steps with 1,000 actors and a rising per-block cost at 20 actors.

These measurements did **not** record a machine identification/hardware
profile in the issue comment. They use a five-run procedure not identical to
our fresh-subprocess runner. Therefore **do not** use them to pass R0 or
calculate Rust-vs-Python speedup ratios. They motivate the observer-on/off
baseline and limit interpretation of any future runtime comparison.

Run `benchmarks/reference_python.py` on the intended reference hardware,
save raw samples under `benchmarks/results/` alongside a provenance manifest,
and publish measured evidence (not just this table) to close the R0 gate.
