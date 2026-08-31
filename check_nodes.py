import json

lines = list(open('orchestrator/_state/nodes.jsonl', encoding='utf-8'))
print(f"lines: {len(lines)}")

promoted = []
open_scored = []
for line in lines:
    r = json.loads(line)
    has_pu = bool(r.get("per_user_by_seed"))
    if r["status"] == "promoted":
        promoted.append((r["iter"], r["id"][:8], has_pu))
    if r["status"] == "open" and has_pu and r["operation"] != "draft":
        open_scored.append((r["iter"], r["id"][:8], r.get("mean_delta"), r.get("lower_95")))

print(f"\npromoted nodes: {len(promoted)}")
for it, i, has_pu in sorted(promoted):
    print(f"  iter={it} id={i} has_per_user={has_pu}")

print(f"\nopen scored nodes: {len(open_scored)}")
for it, i, md, l95 in sorted(open_scored):
    print(f"  iter={it} id={i} mean_delta={md} lower_95={l95}")

print(f"\nlast node in file:")
last = json.loads(lines[-1])
print(f"  iter={last['iter']} id={last['id'][:8]} status={last['status']} "
      f"has_per_user={bool(last.get('per_user_by_seed'))}")