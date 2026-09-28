import csv
import sys
import os
import re
import unicodedata
from collections import defaultdict
import time
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

print("="*70, flush=True)
print("TESTING ENHANCED MULTI-PASS BLOCKING (NAME + ADDRESS INDEXING)", flush=True)
print("="*70, flush=True)

STOP_WORDS_NAME = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'services', 'service', 'solutions',
    'technologies', 'technology', 'enterprises', 'enterprise', 'group',
    'holdings', 'holding', 'industries', 'industry', 'sa', 'sarl', 'sas',
    'gmbh', 'dba', 'the', 'and', 'llp', 'pllc', 'center', 'centre',
    'associates', 'partners', 'international', 'consulting', 'consultants',
    'management', 'global', 'systems', 'system', 'india', 'usa', 'france',
    'societe', 'et', 'cie', 'de', 'du', 'la', 'le', 'les', 'des', 'of', 'in', 'at'
}

STOP_WORDS_ADDR = {
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'drive', 'dr', 'lane', 'ln',
    'court', 'ct', 'boulevard', 'blvd', 'way', 'circle', 'cir', 'place', 'pl',
    'highway', 'hwy', 'parkway', 'pkwy', 'suite', 'ste', 'unit', 'apt', 'apartment',
    'floor', 'fl', 'block', 'blk', 'building', 'bldg', 'near', 'behind', 'opp',
    'opposite', 'plot', 'shop', 'no', 'number', 'door', 'h', 'house', 'sec', 'sector',
    'phase', 'east', 'west', 'north', 'south', 'e', 'w', 'n', 's', 'us', 'usa',
    'india', 'france', 'de', 'du', 'la', 'le', 'rue', 'r', 'bd', 'av', 'chem',
    'chemin', 'route', 'rte', 'cedex', 'delhi', 'new', 'hq', 'region'
}

def remove_accents(text):
    if not text:
        return ""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def clean_text(text):
    if not text:
        return ""
    text = remove_accents(text)
    text = text.lower()
    text = re.sub(r'\[+.*?\]+|\(+.*?\)+|<+.*?>+', ' ', text)
    text = re.sub(r'\b(d\.?b\.?a\.?|doing business as)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 0]
    return " ".join(tokens)

def get_core_tokens(cleaned_name):
    tokens = cleaned_name.split()
    core = [t for t in tokens if t not in STOP_WORDS_NAME and len(t) > 1]
    if not core:
        core = [t for t in tokens if len(t) > 1]
    if not core:
        core = tokens
    return core

def get_address_tokens(cleaned_addr):
    tokens = cleaned_addr.split()
    core = [t for t in tokens if t not in STOP_WORDS_ADDR and len(t) > 2]
    return core

def extract_numbers(text):
    if not text:
        return []
    return re.findall(r'\b\d+\b', text)

# Load 10,000 validation S1 entities
NUM_VAL = 10000
val_gt = {}
with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader, None)
    for row in reader:
        if len(row) >= 2:
            s1_id = row[0]
            matches = set(row[1].strip().split(",")) if row[1].strip() else set()
            val_gt[s1_id] = matches
        if len(val_gt) >= NUM_VAL:
            break

val_s1_ids = set(val_gt.keys())
val_s1_records = {}
with open("dataset/train/train_source1.tsv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader, None)
    for row in reader:
        if row[0] in val_s1_ids:
            cn = clean_text(row[1])
            ca = clean_text(row[2])
            val_s1_records[row[0]] = {
                'id': row[0], 'name': row[1], 'addr': row[2], 'country': row[3],
                'name_clean': cn,
                'core_name': get_core_tokens(cn),
                'addr_clean': ca,
                'core_addr': get_address_tokens(ca),
                'numbers': extract_numbers(row[2])
            }

all_true_target_ids = {m for matches in val_gt.values() for m in matches}
target_records = {}
def load_targets(filename, max_records=200000):
    count = 0
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            eid = row[0]
            if eid in all_true_target_ids or count < max_records:
                cn = clean_text(row[1])
                ca = clean_text(row[2])
                target_records[eid] = {
                    'id': eid, 'name': row[1], 'addr': row[2], 'country': row[3],
                    'name_clean': cn,
                    'core_name': get_core_tokens(cn),
                    'addr_clean': ca,
                    'core_addr': get_address_tokens(ca),
                    'numbers': extract_numbers(row[2])
                }
                count += 1

load_targets("dataset/train/train_source2.tsv", max_records=200000)
load_targets("dataset/train/train_source3.tsv", max_records=200000)
print(f"Total Target Records in Pool: {len(target_records):,}", flush=True)

# Build Enhanced Inverted Indices
print("\nBuilding Enhanced Indices (Name + Address)...", flush=True)
t0 = time.time()

idx_exact_name = defaultdict(list)
idx_name_token = defaultdict(list)
idx_name_prefix = defaultdict(list)
idx_addr_num_street = defaultdict(list)
idx_addr_num = defaultdict(list)
idx_addr_token = defaultdict(list)

for eid, rec in target_records.items():
    country = rec['country']
    nc = rec['name_clean']
    core_n = rec['core_name']
    core_a = rec['core_addr']
    nums = rec['numbers']
    
    # 1. Exact Name
    if nc:
        idx_exact_name[(country, nc)].append(eid)
    
    # 2. Name Tokens & Prefixes
    for t in core_n:
        if len(t) >= 3:
            idx_name_token[(country, t)].append(eid)
            if len(t) >= 4:
                idx_name_prefix[(country, t[:4])].append(eid)
    
    # 3. Address Number + Street Token
    for num in nums:
        idx_addr_num[(country, num)].append(eid)
        for at in core_a:
            if len(at) >= 3:
                idx_addr_num_street[(country, num, at)].append(eid)
    
    # 4. Address Significant Tokens
    for at in core_a:
        if len(at) >= 4:
            idx_addr_token[(country, at)].append(eid)

print(f"Indices built in {time.time()-t0:.2f}s", flush=True)

# Test Enhanced Querying
total_true_matches = sum(len(m) for m in val_gt.values())
print(f"Evaluating recall on {len(val_s1_records):,} validation queries...", flush=True)

for TOP_K in [6, 8, 10, 12, 15]:
    recalled_matches = 0
    total_cands = 0
    
    for s1_id, s1_rec in val_s1_records.items():
        country = s1_rec['country']
        nc = s1_rec['name_clean']
        core_n = s1_rec['core_name']
        core_a = s1_rec['core_addr']
        nums = s1_rec['numbers']
        name_orig = s1_rec['name']
        addr_orig = s1_rec['addr']
        
        cand_scores = defaultdict(float)
        
        # 1. Exact Name match
        if nc:
            for eid in idx_exact_name.get((country, nc), []):
                cand_scores[eid] += 60.0
        
        # 2. Name Token match
        for t in core_n:
            if len(t) >= 3:
                hits = idx_name_token.get((country, t), [])
                if len(hits) <= 400:
                    w = 20.0 / (1.0 + 0.05 * len(hits))
                    for eid in hits:
                        cand_scores[eid] += w
        
        # 3. Address Number + Street match
        for num in nums:
            for at in core_a:
                if len(at) >= 3:
                    hits = idx_addr_num_street.get((country, num, at), [])
                    if len(hits) <= 200:
                        w = 30.0 / (1.0 + 0.05 * len(hits))
                        for eid in hits:
                            cand_scores[eid] += w
        
        # 4. Address Number match (if number is long like 4+ digits)
        for num in nums:
            if len(num) >= 4:
                hits = idx_addr_num.get((country, num), [])
                if len(hits) <= 100:
                    for eid in hits:
                        cand_scores[eid] += 15.0
        
        # 5. Name Prefix fallback if few candidates
        if len(cand_scores) < 4:
            for t in core_n:
                if len(t) >= 4:
                    hits = idx_name_prefix.get((country, t[:4]), [])
                    if len(hits) <= 150:
                        for eid in hits:
                            cand_scores[eid] += 8.0
        
        # 6. Address Token fallback if still few candidates
        if len(cand_scores) < 4:
            for at in core_a:
                if len(at) >= 5:
                    hits = idx_addr_token.get((country, at), [])
                    if len(hits) <= 50:
                        for eid in hits:
                            cand_scores[eid] += 10.0
        
        # Re-score candidates with rapid string similarity
        if not cand_scores:
            top_candidates = []
        else:
            scored = []
            for eid, base_score in cand_scores.items():
                cand_rec = target_records[eid]
                nsim = fuzz.token_set_ratio(name_orig, cand_rec['name'])
                asim = fuzz.token_set_ratio(addr_orig, cand_rec['addr']) if addr_orig and cand_rec['addr'] else 40.0
                
                # Composite relevance score
                total_sim = base_score + (0.55 * nsim + 0.45 * asim)
                scored.append((total_sim, eid))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            top_candidates = [eid for _, eid in scored[:TOP_K]]
        
        total_cands += len(top_candidates)
        true_m = val_gt.get(s1_id, set())
        recalled_matches += len(set(top_candidates) & true_m)
    
    recall_pct = (recalled_matches / total_true_matches) * 100.0 if total_true_matches > 0 else 0.0
    avg_cands = total_cands / len(val_s1_records)
    print(f"Top-K={TOP_K:2d} -> Recall: {recall_pct:6.2f}% ({recalled_matches:,}/{total_true_matches:,}) | Avg Candidates/S1: {avg_cands:4.1f}", flush=True)

