import random
import unittest
from retrievalwatch import Document, Query, RetrievalWatch, EpochGate, should_admit
from retrievalwatch.engine import FenceIndex, Watch


def service():
    return RetrievalWatch(lambda q, d: float(d.text.split(":")[0]),
                          lambda q, docs: "|".join(d.text for d in docs))


class RetrievalTests(unittest.TestCase):
    def test_shadow_deletion_and_historical_pin(self):
        engine = service()
        engine.commit({i: Document(i, 1, (1, i/10), f"{i}:old") for i in range(10)})
        query = Query("q", (1, 0), L=2, k=1, guards=1)
        engine.register(query)
        before = engine.request(query)
        engine.commit({50: Document(50, 1, (1, 0), "50:new")})
        engine.commit({50: None})
        self.assertEqual(engine.request(query), engine.request(query, maintained=False))
        self.assertEqual(engine.request(query, 1), before)
        self.assertEqual(engine.stats["routed"], 2)

    def test_epoch_gate_does_not_bridge_gaps(self):
        engine = service()
        engine.stage(1, {1: Document(1, 1, (1, 0), "1:one")})
        engine.stage(2, {2: Document(2, 1, (0, 1), "2:two")})
        for stage in EpochGate.stages:
            engine.complete(2, stage)
        self.assertEqual(engine.gate.published, 0)
        engine.complete(1, "documents")
        engine.complete(1, "routing")
        self.assertEqual(engine.gate.published, 0)
        with self.assertRaises(ValueError):
            engine.request(Query("q", (1, 0), L=1, k=1), 1)
        engine.complete(1, "vectors")
        self.assertEqual(engine.gate.published, 2)
        self.assertEqual(len(engine.snapshots[2]), 2)

    def test_random_mutations_equal_fresh(self):
        rng = random.Random(891)
        engine = service()
        versions = {}
        queries = [Query(str(i), tuple(rng.uniform(-1, 1) for _ in range(5)), L=4, k=2,
                         guards=i%4, filters={"kind": {"in": [i%2]}}) for i in range(24)]
        for q in queries:
            engine.register(q)
        for epoch in range(200):
            changes = {}
            for _ in range(4):
                i = rng.randrange(45)
                if rng.random() < .3:
                    changes[i] = None
                else:
                    versions[i] = versions.get(i, 0)+1
                    changes[i] = Document(i, versions[i], tuple(rng.uniform(-1, 1) for _ in range(5)),
                                          f"{rng.randrange(20)}:v{versions[i]}", {"kind": rng.randrange(2)})
            engine.commit(changes)
            if epoch%4 == 0:
                for q in queries:
                    self.assertEqual(engine.request(q), engine.request(q, maintained=False))
            if epoch%29 == 0:
                q = queries[0]
                self.assertEqual(engine.request(q, epoch//2),
                                 engine.request(q, epoch//2, maintained=False))

    def test_fence_index_no_false_negative(self):
        rng = random.Random(77)
        watches = {}
        for i in range(70):
            q = Query(str(i), tuple(rng.uniform(-1, 1) for _ in range(4)), L=2, k=1)
            watches[q.signature] = Watch(q, 0, {}, (rng.uniform(-1, 1), -i))
        index = FenceIndex(watches)
        for i in range(300):
            d = Document(i, 1, tuple(rng.uniform(-1, 1) for _ in range(4)), "1:x")
            expected = {w.query.signature for w in watches.values() if w.query.key(d) >= w.fence}
            self.assertEqual(expected, {w.query.signature for w in index.match(d)})

    def test_ties_versions_evidence_and_no_digest_only_reuse(self):
        engine = service()
        engine.commit({i: Document(i, 1, (1, 0), "1:equal") for i in range(5)})
        q = Query("q", (1, 0), L=3, k=1, guards=1)
        engine.register(q)
        self.assertEqual(engine.request(q)["evidence"][0][0], 0)
        self.assertEqual(engine.stats["generations"], 1)
        engine.commit({4: Document(4, 2, (1, 0), "9:excluded-tie")})
        engine.request(q)
        self.assertEqual(engine.stats["generations"], 1)
        engine.commit({0: Document(0, 2, (1, 0), "1:edited")})
        self.assertEqual(engine.request(q)["evidence"][0][1], 2)
        self.assertEqual(engine.stats["generations"], 2)
        with self.assertRaises(ValueError):
            engine.commit({0: Document(0, 1, (1, 0), "1:stale")})

    def test_admission_and_filters(self):
        self.assertTrue(should_admit(10, 10, 1, .1, 2, 1, 2, 1, 1))
        self.assertFalse(should_admit(0, 10, 1, .1, 2, 1, 2, 1, 1))
        q = Query("q", (1, 0), L=1, k=1, filters={"v": {"range": [2, 4]}})
        self.assertTrue(q.eligible(Document(1, 1, (1, 0), "1:x", {"v": 2})))
        self.assertFalse(q.eligible(Document(1, 1, (1, 0), "1:x", {})))


if __name__ == "__main__":
    unittest.main()
