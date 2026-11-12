import argparse
import json
from . import Document, Query, RetrievalWatch

parser = argparse.ArgumentParser(description="Replay JSONL document commits and requests")
parser.add_argument("trace")
args = parser.parse_args()
service = RetrievalWatch(lambda q, d: sum(q.text.lower().count(w) for w in d.text.lower().split()),
                         lambda q, docs: "\n".join(d.text for d in docs))
queries = {}
with open(args.trace, encoding="utf8") as stream:
    for line in stream:
        event = json.loads(line)
        if event["op"] == "commit":
            service.commit({int(i): None if d is None else Document(**d)
                            for i, d in event["changes"].items()})
        elif event["op"] == "register":
            queries[event["name"]] = Query(**event["query"])
            service.register(queries[event["name"]])
        elif event["op"] == "request":
            print(json.dumps(service.request(queries[event["name"]], event.get("epoch"))))
        else:
            raise ValueError("unknown operation")
