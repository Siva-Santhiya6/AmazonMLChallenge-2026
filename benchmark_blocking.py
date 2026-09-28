import csv
import sys
import os
import re
import unicodedata
from collections import defaultdict, Counter
import time
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

print("="*70, flush=True)
print("BENCHMARKING BLOCKING STRATEGY ON VALIDATION SPLIT", flush=True)
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

INDIC_TRANSLIT = {
    'प्राइवेट': 'private', 'लिमिटेड': 'limited', 'एंटरप्राइजेज': 'enterprises',
    'इंटरप्राइजेज': 'enterprises', 'सर्विसेज': 'services', 'सॉल्यूशंस': 'solutions',
    'टेक्नोलॉजीज': 'technologies', 'कंपनी': 'company', 'प्रॉपर्टीज': 'properties',
    'एसोसिएट्स': 'associates', 'उद्योग': 'udyog', 'इन्फ्राटेक': 'infratech',
    'मार्केटिंग': 'marketing', 'फाइनेंस': 'finance', 'ग्रुप': 'group',
    'इंटरनेशनल': 'international', 'कंसल्टेंट्स': 'consultants', 'कार्पोरेशन': 'corporation',
    'कारपोरेशन': 'corporation', 'वेंचर्स': 'ventures', 'होटल': 'hotel',
    'इंडस्ट्रीज': 'industries', 'एलएलपी': 'llp', 'रेड': 'red', 'आदित्य': 'aditya',
    'राम': 'ram', 'शर्मा': 'sharma', 'सिंह': 'singh', 'कुमार': 'kumar', 'गुप्ता': 'gupta'
}

def remove_accents(text):
    if not text:
        return ""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def normalize_indic(text):
    if not text:
        return ""
    words = text.split()
    mapped = [INDIC_TRANSLIT.get(w, w) for w in words]
    return " ".join(mapped)

def clean_name(name):
    if not name:
        return ""
    name = normalize_indic(name)
    name = remove_accents(name)
    name = name.lower()
    name = re.sub(r'\[+.*?\]+|\(+.*?\)+|<+.*?>+', ' ', name)
    name = re.sub(r'\b(d\.?b\.?a\.?|doing business as)\b', ' ', name)
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    tokens = [t for t in name.split() if len(t) > 0]
    return " ".join(tokens)

def get_core_tokens(cleaned_name):
    tokens = cleaned_name.split()
    core = [t for t in tokens if t not in STOP_WORDS_NAME and len(t) > 1]
    if not core:
        core = [t for t in tokens if len(t) > 1]
    if not core:
        core = tokens
    return core

def clean_address(addr):
    if not addr:
        return ""
    addr = remove_accents(addr)
    addr = addr.lower()
    addr = re.sub(r'[^a-z0-9\s]', ' ', addr)
    tokens = [t for t in addr.split() if len(t) > 0]
    return " ".join(tokens)

def extract_numbers(text):
    if not text:
        return []
    return re.findall(r'\b\d+\b', text)

# 1. Load a validation split of 10,000 S1 records from train_ground_truth.tsv
NUM_VAL_S1 = 10000
val_gt = {}
print(f"Loading first {NUM_VAL_S1:,} records from train_ground_truth.tsv...", flush=True)
with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader, None)
    for i, row in enumerate(reader):
        if len(row) >= 2:
            s1_id = row[0]
            matches = set(row[1].strip().split(",")) if row[1].strip() else set()
            val_gt[s1_id] = matches
        if len(val_gt) >= NUM_VAL_S1:
            break

val_s1_ids = set(val_gt.keys())
total_true_matches = sum(len(m) for m in val_gt.values())
singletons_count = sum(1 for m in val_gt.values() if len(m) == 0)
print(f"Validation Set: {len(val_gt):,} S1 entities | {total_true_matches:,} true target matches | {singletons_count:,} singletons", flush=True)

# 2. Load corresponding S1 records
val_s1_records = {}
print("Loading S1 records...", flush=True)
with open("dataset/train/train_source1.tsv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader, None)
    for row in reader:
        if row[0] in val_s1_ids:
            val_s1_records[row[0]] = {
                'id': row[0],
                'name': row[1],
                'addr': row[2],
                'country': row[3],
                'name_clean': clean_name(row[1]),
                'core_tokens': get_core_tokens(clean_name(row[1])),
                'addr_clean': clean_address(row[2]),
                'numbers': extract_numbers(row[2])
            }

print(f"Loaded {len(val_s1_records):,} S1 records.", flush=True)

# 3. Load a slice of S2 and S3 target records (e.g. first 500k from S2 and S3, ensuring all true matches are included)
all_true_target_ids = {m for matches in val_gt.values() for m in matches}
print(f"Total distinct true target IDs needed: {len(all_true_target_ids):,}", flush=True)

target_records = {}
def load_targets(filename, max_records=200000):
    print(f"Loading records from {filename}...", flush=True)
    count = 0
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            eid = row[0]
            if eid in all_true_target_ids or count < max_records:
                target_records[eid] = {
                    'id': eid,
                    'name': row[1],
                    'addr': row[2],
                    'country': row[3],
                    'name_clean': clean_name(row[1]),
                    'core_tokens': get_core_tokens(clean_name(row[1])),
                    'addr_clean': clean_address(row[2]),
                    'numbers': extract_numbers(row[2])
                }
                count += 1

load_targets("dataset/train/train_source2.tsv", max_records=200000)
load_targets("dataset/train/train_source3.tsv", max_records=200000)
print(f"Total Target Records in Pool: {len(target_records):,}", flush=True)

# 4. Build Multi-Index Inverted Indices
print("\nBuilding Inverted Blocking Indices...", flush=True)
t0 = time.time()

index_exact_name = defaultdict(list)
index_token = defaultdict(list)
index_num_token = defaultdict(list)
index_prefix = defaultdict(list)

for eid, rec in target_records.items():
    country = rec['country']
    n_clean = rec['name_clean']
    core = rec['core_tokens']
    numbers = rec['numbers']
    
    # 1. Exact cleaned name
    if n_clean:
        index_exact_name[(country, n_clean)].append(eid)
    
    # 2. Core tokens
    for t in core:
        if len(t) >= 3:
            index_token[(country, t)].append(eid)
            # Prefix of token (first 4 chars)
            if len(t) >= 4:
                index_prefix[(country, t[:4])].append(eid)
    
    # 3. Number + Token prefix
    for num in numbers:
        for t in core:
            if len(t) >= 3:
                index_num_token[(country, num, t[:3])].append(eid)

print(f"Indices built in {time.time()-t0:.2f}s", flush=True)

# 5. Query Candidates for S1 entities and evaluate Recall vs K
print("\nQuerying candidates for validation S1 entities...", flush=True)
t_query = time.time()

for TOP_K in [5, 8, 10, 15, 20]:
    recalled_matches = 0
    total_candidates_generated = 0
    
    for s1_id, s1_rec in val_s1_records.items():
        country = s1_rec['country']
        n_clean = s1_rec['name_clean']
        core = s1_rec['core_tokens']
        numbers = s1_rec['numbers']
        name_orig = s1_rec['name']
        addr_orig = s1_rec['addr']
        
        # Candidate accumulator with frequency / bonus score
        cand_scores = defaultdict(float)
        
        # Pass 1: Exact cleaned name
        if n_clean:
            for eid in index_exact_name.get((country, n_clean), []):
                cand_scores[eid] += 50.0
        
        # Pass 2: Number + Token
        for num in numbers:
            for t in core:
                if len(t) >= 3:
                    for eid in index_num_token.get((country, num, t[:3]), []):
                        cand_scores[eid] += 25.0
        
        # Pass 3: Core Token matching (penalize overly frequent tokens)
        for t in core:
            if len(t) >= 3:
                matches_t = index_token.get((country, t), [])
                if len(matches_t) <= 500: # skip overly generic tokens
                    weight = 15.0 / (1.0 + 0.05 * len(matches_t))
                    for eid in matches_t:
                        cand_scores[eid] += weight
        
        # Pass 4: Token Prefix matching if very few candidates found
        if len(cand_scores) < 5:
            for t in core:
                if len(t) >= 4:
                    matches_p = index_prefix.get((country, t[:4]), [])
                    if len(matches_p) <= 200:
                        for eid in matches_p:
                            cand_scores[eid] += 5.0
        
        # If still very few candidates, try char 3-grams or fallback
        if not cand_scores:
            top_candidates = []
        else:
            # Re-rank candidate subset with fast rapidfuzz string matching
            scored_candidates = []
            for eid, base_score in cand_scores.items():
                cand_rec = target_records[eid]
                nsim = fuzz.token_set_ratio(name_orig, cand_rec['name'])
                asim = fuzz.token_set_ratio(addr_orig, cand_rec['addr']) if addr_orig and cand_rec['addr'] else 50.0
                total_sim = base_score + (0.6 * nsim + 0.4 * asim)
                scored_candidates.append((total_sim, eid))
            
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            top_candidates = [eid for _, eid in scored_candidates[:TOP_K]]
        
        total_candidates_generated += len(top_candidates)
        true_m = val_gt.get(s1_id, set())
        recalled_matches += len(set(top_candidates) & true_m)
    
    recall_pct = (recalled_matches / total_true_matches) * 100.0 if total_true_matches > 0 else 0.0
    avg_cands = total_candidates_generated / len(val_s1_records)
    print(f"Top-K={TOP_K:2d} -> Recall: {recall_pct:6.2f}% ({recalled_matches:,}/{total_true_matches:,}) | Avg Candidates/S1: {avg_cands:4.1f}", flush=True)

print(f"\nTotal benchmark run completed in {time.time()-t_query:.2f}s", flush=True)

