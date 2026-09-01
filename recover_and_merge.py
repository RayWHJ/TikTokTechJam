import json

BACKUP = 'orchestrator/_state/nodes_backup.jsonl'  # your 1-7 data (corrupted)
CURRENT = 'orchestrator/_state/nodes.jsonl'        # your 8-11 data (clean)
OUTPUT = 'orchestrator/_state/nodes.jsonl.merged'

# Recover iter 1-7 from the corrupted backup
content = open(BACKUP, encoding='utf-8').read()
print(f"read {len(content):,} chars from {BACKUP}")

records_old = []
decoder = json.JSONDecoder()
i = 0
skipped = 0
while i < len(content):
    while i < len(content) and content[i].isspace():
        i += 1
    if i >= len(content):
        break
    try:
        obj, end = decoder.raw_decode(content, i)
        records_old.append(obj)
        i = end
    except json.JSONDecodeError:
        nm = content.find('{"iter":', i + 1)
        if nm == -1:
            break
        skipped += 1
        i = nm

print(f"recovered {len(records_old)} records from backup, skipped {skipped} malformed chunks")

# Load the current (post-resume) records
records_new = []
with open(CURRENT, encoding='utf-8') as fh:
    for line in fh:
        try:
            records_new.append(json.loads(line))
        except ValueError:
            continue

print(f"loaded {len(records_new)} records from current")

# Order matters: old records first (iter 1-7 land there), then new records (iter 8-11).
# Same id in both means the newer wins — but iter numbers don't overlap so this
# should be a clean union in practice.
seen = {}
for r in records_old:
    seen[r['id']] = r
for r in records_new:
    seen[r['id']] = r

merged = sorted(seen.values(), key=lambda r: (r.get('iter', 0), r.get('id', '')))

# Report iter coverage so we can eyeball whether iters 1-7 actually landed
iters_present = sorted(set(r.get('iter') for r in merged))
print(f"merged {len(merged)} unique nodes, iters present: {iters_present}")

with open(OUTPUT, 'w', encoding='utf-8') as fh:
    for r in merged:
        fh.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"wrote {OUTPUT}")