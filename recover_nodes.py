import json
import shutil

SRC = "orchestrator/_state/nodes.jsonl"
DST = "orchestrator/_state/nodes.jsonl.recovered"

content = open(SRC, encoding="utf-8").read()
print(f"read {len(content):,} chars from {SRC}")

records = []
i = 0
decoder = json.JSONDecoder()
n = len(content)
skipped_chunks = 0

while i < n:
    # skip whitespace
    while i < n and content[i].isspace():
        i += 1
    if i >= n:
        break
    try:
        obj, end = decoder.raw_decode(content, i)
        records.append(obj)
        i = end
    except json.JSONDecodeError:
        # Skip forward to the next '{"iter":' marker (start of next record)
        next_marker = content.find('{"iter":', i + 1)
        if next_marker == -1:
            print(f"no more records after char {i}, stopping")
            break
        skipped_chunks += 1
        i = next_marker

print(f"parsed {len(records)} records, skipped {skipped_chunks} malformed chunks")

with open(DST, "w", encoding="utf-8") as fh:
    for r in records:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"wrote {DST}")