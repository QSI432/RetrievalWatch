# Implementation notes

| Manuscript component | Reference behavior |
|---|---|
| Exact two-stage retrieval | Full cosine scan, descending `(score, -id)`, deterministic reranker ties by ascending id |
| Exclusion-fence maintenance | Persistent fence, top-(L+B) truncation, fresh refill after guard exhaustion |
| Reverse routing | Sphere tree with conservative float rounding guard; exact filter/key leaf checks |
| Positive/shadow dependencies | Pool and pending IDs receive subsequent updates and deletion tombstones |
| Publication | Contiguous completion frontiers for documents, vectors, routing |
| Pinned snapshots | Retained complete immutable corpus dictionaries; historical requests use fresh search |
| Evidence ledger | Versioned pair scores and full canonical generation-record equality |
| Admission | Explicit estimated-cost function; manual registration/eviction |
| Registration/fence replacement | Global lock serializes updates and registration; no distributed handoff |

The service retains all corpus snapshots, events, score entries, and answers. It is intended for small correctness traces, not large memory or throughput benchmarks. Snapshots contain frozen document records; metadata is canonically serialized at construction. The sphere tree is rebuilt on watch registration/repair/eviction and does not implement predicate-summary pruning. This increases work conservatively.

Model callbacks receive fixed query signatures and immutable documents/evidence. The embedding model is external. The reference prompt is the ordered, complete evidence text in the canonical JSON record; applications needing truncation or prompt packing must implement it deterministically and version the policy. No KV state is represented. Stochastic generation returns the previously committed answer when its complete input matches; byte-equivalence to a new stochastic decode is not claimed.

The release was assembled as an executable reference after manuscript preparation. Its correctness checks do not establish provenance or reproduction of the manuscript's original experiments.
