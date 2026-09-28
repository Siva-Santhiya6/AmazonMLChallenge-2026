import os
import csv
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

train_dir = "dataset/train"
test_dir = "dataset/test"

print("="*60, flush=True)
print("INSPECTING DATASET STATISTICS", flush=True)
print("="*60, flush=True)

for folder, name in [
    (train_dir, "train_source1.tsv"),
    (train_dir, "train_source2.tsv"),
    (train_dir, "train_source3.tsv"),
    (train_dir, "train_ground_truth.tsv"),
    (test_dir, "test_source1.tsv"),
    (test_dir, "test_source2.tsv"),
    (test_dir, "test_source3.tsv"),
]:
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        print(f"File not found: {path}", flush=True)
        continue
    
    line_count = 0
    countries = Counter()
    match_counts = Counter()
    sample_rows = []
    
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            line_count += 1
            if len(sample_rows) < 3:
                sample_rows.append(row)
            if len(row) >= 4:
                countries[row[3]] += 1
            if name == "train_ground_truth.tsv":
                if len(row) >= 2 and row[1].strip():
                    n_matches = len(row[1].strip().split(","))
                    match_counts[n_matches] += 1
                else:
                    match_counts[0] += 1
    
    print(f"\n[{name}] Total Rows: {line_count:,}", flush=True)
    print(f"Header: {header}", flush=True)
    if countries:
        print(f"Countries: {dict(countries)}", flush=True)
    if match_counts:
        print(f"Match counts distribution (matches per S1 entity): {sorted(match_counts.items())[:10]}", flush=True)
        total_s1 = line_count
        singletons = match_counts[0]
        with_matches = total_s1 - singletons
        print(f"Singletons: {singletons:,} ({singletons/total_s1*100:.1f}%) | With matches: {with_matches:,} ({with_matches/total_s1*100:.1f}%)", flush=True)
    print("Sample Rows:", flush=True)
    for i, r in enumerate(sample_rows, 1):
        print(f"  {i}. {r}", flush=True)

