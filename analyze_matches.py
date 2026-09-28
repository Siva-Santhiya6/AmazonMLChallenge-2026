import csv
import sys
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

print("="*70, flush=True)
print("ANALYZING TRUE MATCH EXAMPLES FROM TRAINING SET", flush=True)
print("="*70, flush=True)

# First, read a sample of ground truth rows
gt_samples = []
with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader, None)
    for i, row in enumerate(reader):
        if len(row) >= 2 and row[1].strip():
            matches = row[1].strip().split(",")
            gt_samples.append((row[0], matches))
            if len(gt_samples) >= 25:
                break

target_s1_ids = {s1 for s1, _ in gt_samples}
target_s23_ids = {m for _, matches in gt_samples for m in matches}

s1_data = {}
with open("dataset/train/train_source1.tsv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader, None)
    for row in reader:
        if row[0] in target_s1_ids:
            s1_data[row[0]] = (row[1], row[2], row[3])

s23_data = {}
with open("dataset/train/train_source2.tsv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader, None)
    for row in reader:
        if row[0] in target_s23_ids:
            s23_data[row[0]] = (row[1], row[2], row[3])

with open("dataset/train/train_source3.tsv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader, None)
    for row in reader:
        if row[0] in target_s23_ids:
            s23_data[row[0]] = (row[1], row[2], row[3])

for s1_id, matches in gt_samples[:15]:
    if s1_id not in s1_data:
        continue
    name1, addr1, ctry1 = s1_data[s1_id]
    print(f"\n[S1 Entity] ID: {s1_id} | Country: {ctry1}", flush=True)
    print(f"   Name   : {name1}", flush=True)
    print(f"   Address: {addr1}", flush=True)
    print(f"   Matches ({len(matches)}):", flush=True)
    for mid in matches:
        if mid in s23_data:
            mname, maddr, mctry = s23_data[mid]
            name_sim = fuzz.token_set_ratio(name1, mname)
            addr_sim = fuzz.token_set_ratio(addr1, maddr)
            print(f"     -> [{mid}] ({mctry}) NameSim={name_sim:.1f}, AddrSim={addr_sim:.1f}", flush=True)
            print(f"        Name   : {mname}", flush=True)
            print(f"        Address: {maddr}", flush=True)
        else:
            print(f"     -> [{mid}] NOT FOUND", flush=True)

