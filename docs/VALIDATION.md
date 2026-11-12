# Release validation

Executed on 2026-11-12 with CPython 3.12 on Windows.

| Check | Result |
|---|---|
| `python -m unittest discover -s tests -v` | 6 tests passed |
| Randomized maintained vs fresh retrieval | 24 queries, 200 update epochs, requests every 4 epochs; candidates, evidence, answers equal |
| Reverse sphere-tree completeness | 70 watches, 300 update vectors; exact admitted sets equal |
| Shadow dependency deletion | Passed |
| Contiguous epoch gaps and delayed vector completion | Passed |
| Historical snapshot reads | Passed |
| Deterministic ties, document versions, evidence reuse | Passed |
| Cost admission and metadata range predicate | Passed |
| `python -m retrievalwatch examples/updates.jsonl` | Passed; 3 responses including historical read |
| `python -m pip install --no-deps -e .` | Passed |

The deterministic fixtures validate implementation behavior. No model-serving,
hardware, throughput, or manuscript speedup reproduction was performed.
