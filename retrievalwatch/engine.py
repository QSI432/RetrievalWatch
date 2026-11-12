"""Sequential executable reference of the exact two-stage retrieval contract.

All operations are serialized by one lock. Full immutable corpus snapshots and
an unbounded event history intentionally replace a production MVCC/WAL service.
No ANN, GPU, remote model, or distributed recovery performance is simulated.
"""
from dataclasses import dataclass, field
import json
from math import fsum, sqrt, isfinite, inf
from threading import RLock


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def normalize(vector):
    vector = tuple(float(x) for x in vector)
    if not vector or not all(isfinite(x) for x in vector):
        raise ValueError("embedding must be nonempty and finite")
    norm = sqrt(fsum(x*x for x in vector))
    if not isfinite(norm) or norm == 0:
        raise ValueError("invalid vector norm")
    return tuple(x/norm for x in vector)


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("embedding dimensions differ")
    return fsum(x*y for x, y in zip(a, b))


@dataclass(frozen=True)
class Document:
    id: int
    version: int
    vector: tuple
    text: str
    attrs: object = "{}"

    def __post_init__(self):
        if type(self.id) is not int or type(self.version) is not int or self.version < 1:
            raise ValueError("integer id and positive version required")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        object.__setattr__(self, "vector", normalize(self.vector))
        attrs = json.loads(self.attrs) if isinstance(self.attrs, str) else self.attrs
        if not isinstance(attrs, dict):
            raise ValueError("metadata must be an object")
        object.__setattr__(self, "attrs", canonical(attrs))

    def record(self):
        return {"id": self.id, "version": self.version, "vector": self.vector,
                "text": self.text, "attrs": json.loads(self.attrs)}


@dataclass(frozen=True)
class Query:
    text: str
    vector: tuple
    L: int = 10
    k: int = 3
    guards: int = 4
    filters: object = "{}"
    embedding_model: str = "external-v1"
    reranker: str = "external-v1"
    generator: str = "external-v1"
    prompt: str = "json-evidence-v1"

    def __post_init__(self):
        if not 1 <= self.k <= self.L or self.guards < 0:
            raise ValueError("require 1 <= k <= L and guards >= 0")
        object.__setattr__(self, "vector", normalize(self.vector))
        filters = json.loads(self.filters) if isinstance(self.filters, str) else self.filters
        for rule in filters.values():
            if not isinstance(rule, dict) or len(rule) != 1 or next(iter(rule)) not in ("eq", "in", "range"):
                raise ValueError("filters support eq, in, or inclusive range")
        object.__setattr__(self, "filters", canonical(filters))

    def eligible(self, doc):
        attrs = json.loads(doc.attrs)
        for key, rule in json.loads(self.filters).items():
            if key not in attrs:
                return False
            op, value = next(iter(rule.items()))
            if op == "eq" and attrs[key] != value:
                return False
            if op == "in" and attrs[key] not in value:
                return False
            if op == "range" and not value[0] <= attrs[key] <= value[1]:
                return False
        return True

    @property
    def signature(self):
        return canonical(self.__dict__)

    def key(self, doc):
        return dot(self.vector, doc.vector), -doc.id


class EpochGate:
    stages = ("documents", "vectors", "routing")

    def __init__(self):
        self.done = {s: set() for s in self.stages}
        self.frontiers = {s: 0 for s in self.stages}

    def complete(self, epoch, stage):
        if stage not in self.done or epoch < 1:
            raise ValueError("invalid completion marker")
        self.done[stage].add(epoch)
        while self.frontiers[stage]+1 in self.done[stage]:
            self.frontiers[stage] += 1

    @property
    def published(self):
        return min(self.frontiers.values())


@dataclass
class Watch:
    query: Query
    epoch: int
    pool: dict
    fence: tuple
    pending: dict = field(default_factory=dict)  # id -> latest routed epoch


class FenceIndex:
    """Rebuilt sphere tree. Exact predicate/key tests occur at every leaf.

    Predicate summaries are deliberately omitted (conservative extra work).
    A rounding guard makes pruning conservative for normalized float64 inputs.
    """
    def __init__(self, watches):
        self.root = self.build(list(watches.values()))

    def build(self, watches):
        if not watches:
            return None
        dimension = len(watches[0].query.vector)
        center = tuple(fsum(w.query.vector[j] for w in watches)/len(watches)
                       for j in range(dimension))
        radius = max(sqrt(fsum((x-y)**2 for x, y in zip(center, w.query.vector)))
                     for w in watches)
        node = [center, radius, min(w.fence[0] for w in watches), watches, None]
        if len(watches) > 8:
            axis = max(range(dimension), key=lambda j: max(w.query.vector[j] for w in watches)
                       - min(w.query.vector[j] for w in watches))
            watches.sort(key=lambda w: w.query.vector[axis])
            mid = len(watches)//2
            node[3], node[4] = None, (self.build(watches[:mid]), self.build(watches[mid:]))
        return node

    def match(self, doc):
        result = []
        def visit(node):
            if node is None:
                return
            center, radius, bound, watches, children = node
            margin = 1e-12 * (len(center)+1)
            if dot(center, doc.vector) + radius + margin < bound:
                return
            if watches is not None:
                result.extend(w for w in watches if w.query.eligible(doc)
                              and w.query.key(doc) >= w.fence)
            else:
                for child in children:
                    visit(child)
        visit(self.root)
        return result


def should_admit(request_rate, fresh_cost, hit_cost, repair_probability,
                 repair_cost, matching_rate, update_rate, delivery_cost,
                 registration_rate):
    values = (request_rate, fresh_cost, hit_cost, repair_probability, repair_cost,
              matching_rate, update_rate, delivery_cost, registration_rate)
    if not all(isfinite(x) and x >= 0 for x in values) or repair_probability > 1:
        raise ValueError("invalid cost estimate")
    return request_rate*(fresh_cost-hit_cost-repair_probability*repair_cost) > (
        matching_rate+update_rate*delivery_cost+registration_rate)


class RetrievalWatch:
    def __init__(self, rerank, generate):
        self.rerank, self.generate = rerank, generate
        self.lock = RLock()
        self.gate = EpochGate()
        self.snapshots = {0: {}}
        self.events = {}
        self.watches = {}
        self.index = FenceIndex({})
        self.scores, self.answers = {}, {}
        self.stats = {"refills": 0, "pair_evaluations": 0, "generations": 0,
                      "answer_hits": 0, "routed": 0}

    def stage(self, epoch, changes):
        """Stage atomic upserts/deletes; publish after all three markers.

        Changes map integer IDs to Document or None. Versions increase even
        across deletion/reinsertion; this avoids stale pair-score identities.
        """
        with self.lock:
            if epoch != len(self.events)+1:
                raise ValueError("stage contiguous epochs starting at one")
            previous = self.snapshots[max(self.snapshots)]
            # Validate against staged predecessors too, before accepting an epoch.
            history = dict(previous)
            for e in range(max(self.snapshots)+1, epoch):
                for i, doc in self.events[e].items():
                    if doc is None:
                        history.pop(i, None)
                    else:
                        history[i] = doc
            versions = {i: d.version for events in self.events.values()
                        for i, d in events.items() if d is not None}
            dimensions = {len(d.vector) for d in history.values()}
            dimensions.update(len(w.query.vector) for w in self.watches.values())
            for i, doc in changes.items():
                if type(i) is not int:
                    raise ValueError("IDs must be integers")
                if doc is not None:
                    if doc.id != i or doc.version <= versions.get(i, 0):
                        raise ValueError("upsert needs matching ID and increasing version")
                    dimensions.add(len(doc.vector))
            if len(dimensions) > 1:
                raise ValueError("embedding dimensions differ")
            self.events[epoch] = dict(changes)

    def complete(self, epoch, stage):
        with self.lock:
            if epoch not in self.events:
                raise ValueError("epoch not staged")
            self.gate.complete(epoch, stage)
            for e in range(max(self.snapshots)+1, self.gate.published+1):
                corpus = dict(self.snapshots[e-1])
                for i, doc in self.events[e].items():
                    affected = {sig for sig, w in self.watches.items()
                                if i in w.pool or i in w.pending}
                    if doc is not None:
                        affected.update(w.query.signature for w in self.index.match(doc))
                        corpus[i] = doc
                    else:
                        corpus.pop(i, None)
                    for sig in affected:
                        self.watches[sig].pending[i] = e
                        self.stats["routed"] += 1
                self.snapshots[e] = corpus

    def commit(self, changes):
        with self.lock:
            epoch = len(self.events)+1
            self.stage(epoch, changes)
            for stage in self.gate.stages:
                self.complete(epoch, stage)
            return epoch

    def fresh_pool(self, query, epoch):
        docs = [d for d in self.snapshots[epoch].values() if query.eligible(d)]
        docs.sort(key=query.key, reverse=True)
        M = query.L+query.guards
        retained = docs[:M]
        fence = query.key(retained[-1]) if len(docs) >= M else (-inf, -inf)
        return {d.id: d for d in retained}, fence

    def register(self, query):
        with self.lock:
            if self.snapshots[self.gate.published]:
                doc = next(iter(self.snapshots[self.gate.published].values()))
                dot(query.vector, doc.vector)
            if any(len(w.query.vector) != len(query.vector) for w in self.watches.values()):
                raise ValueError("query dimensions differ")
            pool, fence = self.fresh_pool(query, self.gate.published)
            self.watches[query.signature] = Watch(query, self.gate.published, pool, fence)
            self.index = FenceIndex(self.watches)

    def evict(self, query):
        with self.lock:
            self.watches.pop(query.signature, None)
            self.index = FenceIndex(self.watches)

    def candidates(self, query, epoch, maintained=True):
        watch = self.watches.get(query.signature) if maintained else None
        # Historical reads never overwrite the latest watch materialization.
        if watch is None or epoch != self.gate.published or epoch < watch.epoch:
            pool, _ = self.fresh_pool(query, epoch)
        else:
            pool = dict(watch.pool)
            for i in watch.pending:
                pool.pop(i, None)
                doc = self.snapshots[epoch].get(i)
                if doc is not None and query.eligible(doc) and query.key(doc) >= watch.fence:
                    pool[i] = doc
            ordered = sorted(pool.values(), key=query.key, reverse=True)
            M = query.L+query.guards
            fence = watch.fence
            if len(ordered) >= M:
                ordered = ordered[:M]
                fence = max(fence, query.key(ordered[-1]))
            if fence[0] != -inf and (len(ordered) < query.L or query.key(ordered[query.L-1]) < fence):
                pool, fence = self.fresh_pool(query, epoch)
                self.stats["refills"] += 1
            else:
                pool = {d.id: d for d in ordered}
            watch.pool, watch.fence, watch.epoch = pool, fence, epoch
            watch.pending.clear()
            # Lock serialization closes fence-decrease registration races.
            self.index = FenceIndex(self.watches)
        return sorted(pool.values(), key=query.key, reverse=True)[:query.L]

    def request(self, query, epoch=None, *, maintained=True):
        with self.lock:
            epoch = self.gate.published if epoch is None else epoch
            if epoch not in self.snapshots:
                raise ValueError("epoch is not published/retained")
            candidates = self.candidates(query, epoch, maintained)
            for doc in candidates:
                key = (query.signature, doc.id, doc.version)
                if key not in self.scores:
                    score = float(self.rerank(query, doc))
                    if not isfinite(score):
                        raise ValueError("reranker score must be finite")
                    self.scores[key] = score
                    self.stats["pair_evaluations"] += 1
            evidence = sorted(candidates, key=lambda d: (
                -self.scores[(query.signature, d.id, d.version)], d.id))[:query.k]
            # Full canonical bytes, including model settings and ordered versions,
            # are the cache key. Hash equality alone never authorizes answer reuse.
            record = canonical({"signature": query.signature,
                                "evidence": [[d.id, d.version, d.text] for d in evidence]})
            if record not in self.answers:
                self.answers[record] = self.generate(query, tuple(evidence))
                self.stats["generations"] += 1
            else:
                self.stats["answer_hits"] += 1
            return {"epoch": epoch, "candidates": [(d.id, d.version) for d in candidates],
                    "evidence": [(d.id, d.version, d.text) for d in evidence],
                    "answer": self.answers[record]}
