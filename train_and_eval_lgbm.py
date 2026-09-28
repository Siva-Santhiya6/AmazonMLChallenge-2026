import csv
import sys
import os
import re
import unicodedata
import numpy as np
from collections import defaultdict
import time
from rapidfuzz import fuzz
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8')

print("="*70, flush=True)
print("TRAINING & EVALUATING LIGHTGBM ENTITY RESOLUTION PIPELINE", flush=True)
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
        return set()
    return set(re.findall(r'\b\d+\b', text))

def get_char_ngrams(text, n=3):
    if not text or len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text)-n+1)}

# 1. Load 15,000 S1 records for training & validation split
NUM_S1 = 15000
ground_truth = {}
print(f"Loading first {NUM_S1:,} records from train_ground_truth.tsv...", flush=True)
with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader, None)
    for row in reader:
        if len(row) >= 2:
            s1_id = row[0]
            matches = set(row[1].strip().split(",")) if row[1].strip() else set()
            ground_truth[s1_id] = matches
        if len(ground_truth) >= NUM_S1:
            break

all_s1_ids = list(ground_truth.keys())
train_s1_ids, val_s1_ids = train_test_split(all_s1_ids, test_size=0.33, random_state=42)
train_s1_set = set(train_s1_ids)
val_s1_set = set(val_s1_ids)

print(f"Dataset Split: {len(train_s1_set):,} Train S1 | {len(val_s1_set):,} Val S1", flush=True)

# Load S1 records
s1_records = {}
with open("dataset/train/train_source1.tsv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader, None)
    for row in reader:
        if row[0] in ground_truth:
            cn = clean_text(row[1])
            ca = clean_text(row[2])
            s1_records[row[0]] = {
                'id': row[0], 'name': row[1], 'addr': row[2], 'country': row[3],
                'name_clean': cn,
                'core_name': get_core_tokens(cn),
                'addr_clean': ca,
                'core_addr': get_address_tokens(ca),
                'numbers': extract_numbers(row[2]),
                'char3': get_char_ngrams(cn, 3)
            }

all_true_target_ids = {m for matches in ground_truth.values() for m in matches}
target_records = {}
def load_targets(filename, max_records=250000):
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
                    'numbers': extract_numbers(row[2]),
                    'char3': get_char_ngrams(cn, 3)
                }
                count += 1

load_targets("dataset/train/train_source2.tsv", max_records=250000)
load_targets("dataset/train/train_source3.tsv", max_records=250000)
print(f"Total Target Records in Pool: {len(target_records):,}", flush=True)

# Build Enhanced Inverted Indices
print("\nBuilding Inverted Blocking Indices...", flush=True)
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
    
    if nc:
        idx_exact_name[(country, nc)].append(eid)
    for t in core_n:
        if len(t) >= 3:
            idx_name_token[(country, t)].append(eid)
            if len(t) >= 4:
                idx_name_prefix[(country, t[:4])].append(eid)
    for num in nums:
        idx_addr_num[(country, num)].append(eid)
        for at in core_a:
            if len(at) >= 3:
                idx_addr_num_street[(country, num, at)].append(eid)
    for at in core_a:
        if len(at) >= 4:
            idx_addr_token[(country, at)].append(eid)

print("Inverted indices ready.", flush=True)

def generate_candidates(s1_id, top_k=12):
    s1_rec = s1_records[s1_id]
    country = s1_rec['country']
    nc = s1_rec['name_clean']
    core_n = s1_rec['core_name']
    core_a = s1_rec['core_addr']
    nums = s1_rec['numbers']
    name_orig = s1_rec['name']
    addr_orig = s1_rec['addr']
    
    cand_scores = defaultdict(float)
    if nc:
        for eid in idx_exact_name.get((country, nc), []):
            cand_scores[eid] += 60.0
    for t in core_n:
        if len(t) >= 3:
            hits = idx_name_token.get((country, t), [])
            if len(hits) <= 400:
                w = 20.0 / (1.0 + 0.05 * len(hits))
                for eid in hits:
                    cand_scores[eid] += w
    for num in nums:
        for at in core_a:
            if len(at) >= 3:
                hits = idx_addr_num_street.get((country, num, at), [])
                if len(hits) <= 200:
                    w = 30.0 / (1.0 + 0.05 * len(hits))
                    for eid in hits:
                        cand_scores[eid] += w
    for num in nums:
        if len(num) >= 4:
            hits = idx_addr_num.get((country, num), [])
            if len(hits) <= 100:
                for eid in hits:
                    cand_scores[eid] += 15.0
    if len(cand_scores) < 4:
        for t in core_n:
            if len(t) >= 4:
                hits = idx_name_prefix.get((country, t[:4]), [])
                if len(hits) <= 150:
                    for eid in hits:
                        cand_scores[eid] += 8.0
    if len(cand_scores) < 4:
        for at in core_a:
            if len(at) >= 5:
                hits = idx_addr_token.get((country, at), [])
                if len(hits) <= 50:
                    for eid in hits:
                        cand_scores[eid] += 10.0
    
    if not cand_scores:
        return []
    
    scored = []
    for eid, base_score in cand_scores.items():
        cand_rec = target_records[eid]
        nsim = fuzz.token_set_ratio(name_orig, cand_rec['name'])
        asim = fuzz.token_set_ratio(addr_orig, cand_rec['addr']) if addr_orig and cand_rec['addr'] else 40.0
        total_sim = base_score + (0.55 * nsim + 0.45 * asim)
        scored.append((total_sim, eid))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def extract_features(s1_rec, cand_rec, block_score, block_rank, top1_score):
    name1, name2 = s1_rec['name'], cand_rec['name']
    addr1, addr2 = s1_rec['addr'], cand_rec['addr']
    nc1, nc2 = s1_rec['name_clean'], cand_rec['name_clean']
    ca1, ca2 = s1_rec['addr_clean'], cand_rec['addr_clean']
    
    # 1. Name Similarities
    n_ratio = fuzz.ratio(name1, name2)
    n_part_ratio = fuzz.partial_ratio(name1, name2)
    n_tok_sort = fuzz.token_sort_ratio(name1, name2)
    n_tok_set = fuzz.token_set_ratio(name1, name2)
    nc_tok_sort = fuzz.token_sort_ratio(nc1, nc2)
    nc_tok_set = fuzz.token_set_ratio(nc1, nc2)
    exact_name = 1.0 if nc1 == nc2 and len(nc1) > 0 else 0.0
    
    # Jaccard on tokens
    toks1 = set(nc1.split())
    toks2 = set(nc2.split())
    tok_union = len(toks1 | toks2)
    tok_jaccard = len(toks1 & toks2) / tok_union if tok_union > 0 else 0.0
    
    # Char 3-gram Jaccard
    c1, c2 = s1_rec['char3'], cand_rec['char3']
    c_union = len(c1 | c2)
    char3_jaccard = len(c1 & c2) / c_union if c_union > 0 else 0.0
    
    # Length stats
    l1, l2 = len(name1), len(name2)
    len_diff = abs(l1 - l2)
    len_ratio = min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 1.0
    
    # 2. Address Similarities
    a_tok_sort = fuzz.token_sort_ratio(addr1, addr2) if addr1 and addr2 else 0.0
    a_tok_set = fuzz.token_set_ratio(addr1, addr2) if addr1 and addr2 else 0.0
    ac_tok_sort = fuzz.token_sort_ratio(ca1, ca2) if ca1 and ca2 else 0.0
    ac_tok_set = fuzz.token_set_ratio(ca1, ca2) if ca1 and ca2 else 0.0
    addr_empty = 1.0 if not addr2.strip() else 0.0
    
    # Numeric Overlap
    num1, num2 = s1_rec['numbers'], cand_rec['numbers']
    overlap_nums = len(num1 & num2)
    num_prec = overlap_nums / len(num2) if len(num2) > 0 else 0.0
    num_rec = overlap_nums / len(num1) if len(num1) > 0 else 0.0
    
    # Combined Text
    comb1 = name1 + " " + addr1
    comb2 = name2 + " " + addr2
    comb_tok_set = fuzz.token_set_ratio(comb1, comb2)
    
    is_s2 = 1.0 if cand_rec['id'].startswith('S2-') else 0.0
    score_diff = top1_score - block_score
    
    return [
        n_ratio, n_part_ratio, n_tok_sort, n_tok_set,
        nc_tok_sort, nc_tok_set, exact_name, tok_jaccard, char3_jaccard,
        len_diff, len_ratio,
        a_tok_sort, a_tok_set, ac_tok_sort, ac_tok_set, addr_empty,
        float(len(num1)), float(len(num2)), float(overlap_nums), num_prec, num_rec,
        comb_tok_set, is_s2, float(block_rank), block_score, score_diff
    ]

feature_names = [
    'n_ratio', 'n_part_ratio', 'n_tok_sort', 'n_tok_set',
    'nc_tok_sort', 'nc_tok_set', 'exact_name', 'tok_jaccard', 'char3_jaccard',
    'len_diff', 'len_ratio',
    'a_tok_sort', 'a_tok_set', 'ac_tok_sort', 'ac_tok_set', 'addr_empty',
    'num_cnt_s1', 'num_cnt_tgt', 'num_overlap', 'num_prec', 'num_rec',
    'comb_tok_set', 'is_s2', 'block_rank', 'block_score', 'score_diff'
]

# 2. Build Training Dataset
print("\nExtracting Features for Training Candidates...", flush=True)
X_train, y_train = [], []
for s1_id in train_s1_ids:
    cands = generate_candidates(s1_id, top_k=12)
    if not cands:
        continue
    top1_score = cands[0][0]
    true_m = ground_truth[s1_id]
    for rank, (b_score, tgt_id) in enumerate(cands, 1):
        cand_rec = target_records[tgt_id]
        feat = extract_features(s1_records[s1_id], cand_rec, b_score, rank, top1_score)
        label = 1 if tgt_id in true_m else 0
        X_train.append(feat)
        y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
print(f"Training set: {len(X_train):,} pairs (Positives: {np.sum(y_train):,}, Negatives: {len(y_train)-np.sum(y_train):,})", flush=True)

# 3. Train LightGBM Model
print("\nTraining LightGBM Classifier...", flush=True)
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

clf = lgb.LGBMClassifier(**params)
clf.fit(X_train, y_train)

# Feature Importances
imp = clf.feature_importances_
top_feat_idx = np.argsort(imp)[::-1][:10]
print("\nTop 10 Feature Importances:")
for i in top_feat_idx:
    print(f"  {feature_names[i]:25}: {imp[i]}")

# 4. Evaluate on Validation Set & Optimize Threshold for Macro-F0.5
print("\nEvaluating on Held-Out Validation Set...", flush=True)
val_candidate_data = {}
for s1_id in val_s1_ids:
    cands = generate_candidates(s1_id, top_k=12)
    if not cands:
        val_candidate_data[s1_id] = []
        continue
    top1_score = cands[0][0]
    cand_entries = []
    for rank, (b_score, tgt_id) in enumerate(cands, 1):
        cand_rec = target_records[tgt_id]
        feat = extract_features(s1_records[s1_id], cand_rec, b_score, rank, top1_score)
        cand_entries.append((tgt_id, feat))
    val_candidate_data[s1_id] = cand_entries

# Predict probabilities
all_val_feats = []
val_index_map = []
for s1_id, entries in val_candidate_data.items():
    for tgt_id, feat in entries:
        all_val_feats.append(feat)
        val_index_map.append((s1_id, tgt_id))

if all_val_feats:
    val_probs = clf.predict_proba(np.array(all_val_feats))[:, 1]
    
val_pred_probs = defaultdict(list)
for (s1_id, tgt_id), p in zip(val_index_map, val_probs):
    val_pred_probs[s1_id].append((tgt_id, p))

def compute_macro_f05(pred_dict, gt_dict):
    f05_list = []
    for s1_id, true_set in gt_dict.items():
        pred_set = set(pred_dict.get(s1_id, []))
        if len(true_set) == 0:
            f05 = 1.0 if len(pred_set) == 0 else 0.0
        else:
            if len(pred_set) == 0:
                f05 = 0.0
            else:
                tp = len(pred_set & true_set)
                p = tp / len(pred_set)
                r = tp / len(true_set)
                denom = 0.25 * p + r
                f05 = (1.25 * p * r) / denom if denom > 0 else 0.0
        f05_list.append(f05)
    return np.mean(f05_list)

print("\n--- Threshold Optimization for Macro-F_0.5 ---", flush=True)
best_thresh, best_f05 = 0.5, 0.0
for thresh in np.arange(0.20, 0.95, 0.05):
    pred_dict = {}
    for s1_id in val_s1_ids:
        preds = [tgt_id for tgt_id, p in val_pred_probs.get(s1_id, []) if p >= thresh]
        pred_dict[s1_id] = preds
    score = compute_macro_f05(pred_dict, {k: ground_truth[k] for k in val_s1_ids})
    print(f"Threshold = {thresh:.2f} -> Macro-F_0.5 = {score:.5f}", flush=True)
    if score > best_f05:
        best_f05 = score
        best_thresh = thresh

print(f"\n=======================================================", flush=True)
print(f"BEST VALIDATION MACRO-F_0.5: {best_f05:.5f} at Threshold = {best_thresh:.2f}", flush=True)
print(f"=======================================================", flush=True)

