import sys
import os
import gc
import time
import csv
import string
import unicodedata
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from rapidfuzz import fuzz
import joblib

sys.stdout.reconfigure(encoding='utf-8')

PUNCT_TRANS = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

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
    'india', 'france', 'de', 'du', 'la', 'le', 'rue', 'r', 'chem', 'chemin', 'route', 'rte',
    'cedex', 'delhi', 'new', 'hq', 'region'
}

INDIC_TRANSLIT = {
    'प्राइवेट': 'private', 'लिमिटेड': 'limited', 'एंटरप्राइजेज': 'enterprises',
    'इंटरप्राइजेज': 'enterprises', 'सर्विसेज': 'services', 'सॉल्यूशंस': 'solutions',
    'टेक्नोलॉजीज': 'technologies', 'कंपनी': 'company', 'प्रॉपर्टीज': 'properties',
    'एसोसिएट्स': 'associates', 'उद्योग': 'udyog', 'इन्फ्राटेक': 'infratech',
    'मार्केटिंग': 'marketing', 'फाइनेंस': 'finance', 'ग्रुप': 'group',
    'इंटरनेशनल': 'international', 'कंसल्टेंट्स': 'consultants', 'कार्पोरेशन': 'corporation',
    'कारपोरेशन': 'corporation', 'वेंचर्स': 'ventures', 'होटल': 'hotel',
    'इंडस्ट्रीज': 'industries', 'एलएलपी': 'llp'
}

def remove_accents(text):
    if not text:
        return ""
    if text.isascii():
        return text
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def normalize_indic(text):
    if not text:
        return ""
    if text.isascii():
        return text
    words = text.split()
    mapped = [INDIC_TRANSLIT.get(w, w) for w in words]
    return " ".join(mapped)

def clean_fast(text):
    if not text:
        return ""
    if not text.isascii():
        text = normalize_indic(text)
        text = remove_accents(text)
    text = text.lower().translate(PUNCT_TRANS)
    return " ".join(text.split())

def extract_record(name, addr):
    cn = clean_fast(name)
    ca = clean_fast(addr)
    n_toks = cn.split()
    core_n = tuple(t for t in n_toks if t not in STOP_WORDS_NAME and len(t) > 1) or tuple(t for t in n_toks if len(t) > 1) or tuple(n_toks)
    a_toks = ca.split()
    core_a = tuple(t for t in a_toks if t not in STOP_WORDS_ADDR and len(t) > 2)
    nums = tuple(t for t in a_toks if t.isdigit())
    return (name, addr, cn, core_n, ca, core_a, nums)

def extract_pairwise_features(s1_rec, cand_rec, tgt_id, block_score, block_rank, top1_score):
    name1, addr1, nc1, _, ca1, _, num1 = s1_rec
    name2, addr2, nc2, _, ca2, _, num2 = cand_rec
    
    n_ratio = fuzz.ratio(name1, name2)
    n_part_ratio = fuzz.partial_ratio(name1, name2)
    n_tok_sort = fuzz.token_sort_ratio(name1, name2)
    n_tok_set = fuzz.token_set_ratio(name1, name2)
    nc_tok_sort = fuzz.token_sort_ratio(nc1, nc2)
    nc_tok_set = fuzz.token_set_ratio(nc1, nc2)
    exact_name = 1.0 if nc1 == nc2 and len(nc1) > 0 else 0.0
    
    toks1 = set(nc1.split())
    toks2 = set(nc2.split())
    tok_union = len(toks1 | toks2)
    tok_jaccard = len(toks1 & toks2) / tok_union if tok_union > 0 else 0.0
    
    c1 = {nc1[i:i+3] for i in range(len(nc1)-2)} if len(nc1) >= 3 else set()
    c2 = {nc2[i:i+3] for i in range(len(nc2)-2)} if len(nc2) >= 3 else set()
    c_union = len(c1 | c2)
    char3_jaccard = len(c1 & c2) / c_union if c_union > 0 else 0.0
    
    l1, l2 = len(name1), len(name2)
    len_diff = abs(l1 - l2)
    len_ratio = min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 1.0
    
    a_tok_sort = fuzz.token_sort_ratio(addr1, addr2) if addr1 and addr2 else 0.0
    a_tok_set = fuzz.token_set_ratio(addr1, addr2) if addr1 and addr2 else 0.0
    ac_tok_sort = fuzz.token_sort_ratio(ca1, ca2) if ca1 and ca2 else 0.0
    ac_tok_set = fuzz.token_set_ratio(ca1, ca2) if ca1 and ca2 else 0.0
    addr_empty = 1.0 if not addr2.strip() else 0.0
    
    s_num1 = set(num1)
    s_num2 = set(num2)
    overlap_nums = len(s_num1 & s_num2)
    num_prec = overlap_nums / len(s_num2) if len(s_num2) > 0 else 0.0
    num_rec = overlap_nums / len(s_num1) if len(s_num1) > 0 else 0.0
    
    comb1 = name1 + " " + addr1
    comb2 = name2 + " " + addr2
    comb_tok_set = fuzz.token_set_ratio(comb1, comb2)
    
    is_s2 = 1.0 if tgt_id.startswith('S2-') else 0.0
    score_diff = top1_score - block_score
    
    # Enhanced distinguishing features:
    # 1. Exact address match indicator
    exact_addr = 1.0 if ca1 == ca2 and len(ca1) > 0 else 0.0
    # 2. Name prefix match
    p1 = nc1[:4] if len(nc1) >= 4 else nc1
    p2 = nc2[:4] if len(nc2) >= 4 else nc2
    prefix_match = 1.0 if p1 == p2 and len(p1) >= 3 else 0.0
    # 3. Numeric exact match indicator
    num_exact = 1.0 if num1 and num2 and num1 == num2 else 0.0
    # 4. Token difference count
    tok_diff = float(abs(len(toks1) - len(toks2)))
    
    return [
        n_ratio, n_part_ratio, n_tok_sort, n_tok_set,
        nc_tok_sort, nc_tok_set, exact_name, tok_jaccard, char3_jaccard,
        len_diff, len_ratio,
        a_tok_sort, a_tok_set, ac_tok_sort, ac_tok_set, addr_empty,
        float(len(num1)), float(len(num2)), float(overlap_nums), num_prec, num_rec,
        comb_tok_set, is_s2, float(block_rank), block_score, score_diff,
        exact_addr, prefix_match, num_exact, tok_diff
    ]

FEATURE_NAMES = [
    'n_ratio', 'n_part_ratio', 'n_tok_sort', 'n_tok_set',
    'nc_tok_sort', 'nc_tok_set', 'exact_name', 'tok_jaccard', 'char3_jaccard',
    'len_diff', 'len_ratio',
    'a_tok_sort', 'a_tok_set', 'ac_tok_sort', 'ac_tok_set', 'addr_empty',
    'num_cnt_s1', 'num_cnt_tgt', 'num_overlap', 'num_prec', 'num_rec',
    'comb_tok_set', 'is_s2', 'block_rank', 'block_score', 'score_diff',
    'exact_addr', 'prefix_match', 'num_exact', 'tok_diff'
]

def load_data_and_create_splits(train_dir, total_s1=30000, val_size=10000, max_target_pool=600000):
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA & CREATING CLEAN TRAIN/VAL SPLIT (NO LEAKAGE)")
    print("="*70, flush=True)
    t0 = time.time()
    
    # 1. Load ground truth
    ground_truth = {}
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                s1_id = parts[0]
                matches = set(parts[1].split(",")) if parts[1].strip() else set()
                ground_truth[s1_id] = matches
            if len(ground_truth) >= total_s1:
                break
                
    s1_all_ids = list(ground_truth.keys())
    train_ids, val_ids = train_test_split(s1_all_ids, test_size=val_size, random_state=42, shuffle=True)
    train_set = set(train_ids)
    val_set = set(val_ids)
    
    print(f"Loaded {len(s1_all_ids):,} S1 Entities -> Train: {len(train_ids):,} | Validation: {len(val_ids):,}", flush=True)
    
    # 2. Load S1 records
    s1_records = {}
    with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[0] in ground_truth:
                s1_records[parts[0]] = (parts[3], extract_record(parts[1], parts[2]))
                
    # 3. Load Target records (Dense pool with both true matches and negative distractors)
    all_true_targets = {m for matches in ground_truth.values() for m in matches}
    target_records = {}
    
    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        count = 0
        with open(os.path.join(train_dir, fname), "r", encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    if parts[0] in all_true_targets or count < (max_target_pool // 2):
                        target_records[parts[0]] = (parts[3], extract_record(parts[1], parts[2]))
                        count += 1
                        
    print(f"Loaded {len(target_records):,} Target Records in pool ({time.time()-t0:.2f}s total)", flush=True)
    
    # 4. Build Inverted Indices
    print("\nBuilding Inverted Blocking Indices...", flush=True)
    t_idx = time.time()
    idx_exact = defaultdict(list)
    idx_token = defaultdict(list)
    idx_prefix = defaultdict(list)
    idx_num_street = defaultdict(list)
    idx_num = defaultdict(list)
    
    for eid, (country, rec) in target_records.items():
        _, _, nc, core_n, _, core_a, nums = rec
        if nc:
            idx_exact[(country, nc)].append(eid)
        for t in core_n:
            if len(t) >= 3:
                idx_token[(country, t)].append(eid)
                if len(t) >= 4:
                    idx_prefix[(country, t[:4])].append(eid)
        for num in nums:
            idx_num[(country, num)].append(eid)
            for at in core_a:
                if len(at) >= 3:
                    idx_num_street[(country, num, at)].append(eid)
                    
    print(f"Inverted indices built in {time.time()-t_idx:.2f}s", flush=True)
    
    indices = {
        'exact': idx_exact,
        'token': idx_token,
        'prefix': idx_prefix,
        'num_street': idx_num_street,
        'num': idx_num
    }
    
    return train_ids, val_ids, ground_truth, s1_records, target_records, indices

def generate_candidates_for_query(s1_id, s1_records, target_records, indices, top_k=10):
    country, s1_rec = s1_records[s1_id]
    name1, addr1, nc1, core_n, ca1, core_a, num1 = s1_rec
    
    idx_exact = indices['exact']
    idx_token = indices['token']
    idx_prefix = indices['prefix']
    idx_num_street = indices['num_street']
    idx_num = indices['num']
    
    cand_scores = defaultdict(float)
    if nc1:
        for eid in idx_exact.get((country, nc1), ()):
            cand_scores[eid] += 60.0
            
    for t in core_n:
        if len(t) >= 3:
            hits = idx_token.get((country, t), ())
            if len(hits) <= 400:
                w = 20.0 / (1.0 + 0.05 * len(hits))
                for eid in hits:
                    cand_scores[eid] += w
                    
    for num in num1:
        for at in core_a:
            if len(at) >= 3:
                hits = idx_num_street.get((country, num, at), ())
                if len(hits) <= 200:
                    w = 30.0 / (1.0 + 0.05 * len(hits))
                    for eid in hits:
                        cand_scores[eid] += w
                        
    if len(cand_scores) < 3:
        for num in num1:
            if len(num) >= 4:
                hits = idx_num.get((country, num), ())
                if len(hits) <= 100:
                    for eid in hits:
                        cand_scores[eid] += 15.0
                        
    if len(cand_scores) < 3:
        for t in core_n:
            if len(t) >= 4:
                hits = idx_prefix.get((country, t[:4]), ())
                if len(hits) <= 150:
                    for eid in hits:
                        cand_scores[eid] += 8.0
                        
    if not cand_scores:
        return []
        
    if len(cand_scores) > 40:
        top_items = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:40]
    else:
        top_items = list(cand_scores.items())
        
    scored = []
    for eid, base_score in top_items:
        _, cand_rec = target_records[eid]
        nsim = fuzz.token_set_ratio(name1, cand_rec[0])
        asim = fuzz.token_set_ratio(addr1, cand_rec[1]) if addr1 and cand_rec[1] else 40.0
        scored.append((base_score + 0.55 * nsim + 0.45 * asim, eid))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def evaluate_blocking_recall(s1_id_list, ground_truth, s1_records, target_records, indices, top_k_values=[6, 8, 10, 12, 15]):
    print("\n" + "="*70)
    print("STEP 2: DIAGNOSTIC — EVALUATING CANDIDATE-GENERATION (BLOCKING) RECALL")
    print("="*70, flush=True)
    
    total_true_matches = sum(len(ground_truth[s1_id]) for s1_id in s1_id_list)
    print(f"Evaluating across {len(s1_id_list):,} validation queries ({total_true_matches:,} true target links)...", flush=True)
    
    for top_k in top_k_values:
        recalled = 0
        total_cands = 0
        for s1_id in s1_id_list:
            cands = generate_candidates_for_query(s1_id, s1_records, target_records, indices, top_k=top_k)
            cand_eids = {eid for _, eid in cands}
            true_m = ground_truth[s1_id]
            recalled += len(cand_eids & true_m)
            total_cands += len(cand_eids)
            
        recall_pct = (recalled / total_true_matches * 100) if total_true_matches > 0 else 0.0
        avg_cands = total_cands / len(s1_id_list)
        print(f"  Top-K = {top_k:2d} -> Blocking Recall Ceiling: {recall_pct:6.2f}% ({recalled:,}/{total_true_matches:,}) | Avg Cands/Query: {avg_cands:.2f}")

def compute_metrics_and_confusion_matrix(y_true, y_pred, y_probs=None):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return {
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'Precision': prec, 'Recall': rec, 'F0.5': f05,
        'Accuracy': acc, 'F1': f1
    }

def compute_macro_f05_per_entity(val_pred_probs, val_ids, ground_truth, threshold):
    f05_scores = []
    for s1_id in val_ids:
        true_m = ground_truth[s1_id]
        cand_list = val_pred_probs.get(s1_id, [])
        pred_m = {eid for eid, p in cand_list if p >= threshold}
        
        if len(true_m) == 0:
            score = 1.0 if len(pred_m) == 0 else 0.0
        else:
            if len(pred_m) == 0:
                score = 0.0
            else:
                tp = len(pred_m & true_m)
                p = tp / len(pred_m)
                r = tp / len(true_m)
                denom = 0.25 * p + r
                score = (1.25 * p * r) / denom if denom > 0 else 0.0
        f05_scores.append(score)
    return float(np.mean(f05_scores))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default="dataset/train")
    parser.add_argument("--out-csv", type=str, default="output/tuning_results.csv")
    args = parser.parse_args()
    
    os.makedirs("output", exist_ok=True)
    
    # 1. Load Data
    train_ids, val_ids, ground_truth, s1_records, target_records, indices = load_data_and_create_splits(
        args.train_dir, total_s1=30000, val_size=10000, max_target_pool=600000
    )
    
    # 2. Evaluate Candidate Generation (Blocking) Recall
    evaluate_blocking_recall(val_ids, ground_truth, s1_records, target_records, indices, top_k_values=[6, 8, 10, 12, 15])
    
    # 3. Extract Training Features
    print("\n" + "="*70)
    print("STEP 3: EXTRACTING PAIRWISE 30-DIMENSIONAL FEATURES FOR TRAINING & VALIDATION")
    print("="*70, flush=True)
    
    t0 = time.time()
    X_train, y_train = [], []
    for s1_id in train_ids:
        cands = generate_candidates_for_query(s1_id, s1_records, target_records, indices, top_k=10)
        if not cands:
            continue
        top1_score = cands[0][0]
        true_m = ground_truth[s1_id]
        _, s1_rec = s1_records[s1_id]
        
        for rank, (b_score, tgt_id) in enumerate(cands, 1):
            _, cand_rec = target_records[tgt_id]
            feat = extract_pairwise_features(s1_rec, cand_rec, tgt_id, b_score, rank, top1_score)
            label = 1 if tgt_id in true_m else 0
            X_train.append(feat)
            y_train.append(label)
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"Training Set: {len(X_train):,} pairs (Positives: {np.sum(y_train):,}, Negatives: {len(y_train)-np.sum(y_train):,}) [{time.time()-t0:.2f}s]", flush=True)
    
    # Extract Validation Features
    t0 = time.time()
    val_cand_map = defaultdict(list)
    X_val, y_val = [], []
    val_pair_index = []
    
    for s1_id in val_ids:
        cands = generate_candidates_for_query(s1_id, s1_records, target_records, indices, top_k=10)
        if not cands:
            continue
        top1_score = cands[0][0]
        true_m = ground_truth[s1_id]
        _, s1_rec = s1_records[s1_id]
        
        for rank, (b_score, tgt_id) in enumerate(cands, 1):
            _, cand_rec = target_records[tgt_id]
            feat = extract_pairwise_features(s1_rec, cand_rec, tgt_id, b_score, rank, top1_score)
            label = 1 if tgt_id in true_m else 0
            X_val.append(feat)
            y_val.append(label)
            val_pair_index.append((s1_id, tgt_id))
            
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    print(f"Validation Set: {len(X_val):,} pairs (Positives: {np.sum(y_val):,}, Negatives: {len(y_val)-np.sum(y_val):,}) [{time.time()-t0:.2f}s]", flush=True)
    
    # 4. Systematic Experiment Tracking Table
    experiments = []
    
    def run_experiment(exp_id, params, fit_params=None, threshold=0.70):
        t_exp = time.time()
        clf = lgb.LGBMClassifier(**params)
        
        if fit_params:
            clf.fit(X_train, y_train, **fit_params)
        else:
            clf.fit(X_train, y_train)
            
        best_iter = getattr(clf, 'best_iteration_', params.get('n_estimators', 300))
        
        train_probs = clf.predict_proba(X_train)[:, 1]
        val_probs = clf.predict_proba(X_val)[:, 1]
        
        train_preds = (train_probs >= threshold).astype(int)
        val_preds = (val_probs >= threshold).astype(int)
        
        train_m = compute_metrics_and_confusion_matrix(y_train, train_preds)
        val_m = compute_metrics_and_confusion_matrix(y_val, val_preds)
        
        # Build val pred map for macro-F0.5 per entity
        v_map = defaultdict(list)
        for (s1_id, tgt_id), p in zip(val_pair_index, val_probs):
            v_map[s1_id].append((tgt_id, p))
        macro_f05 = compute_macro_f05_per_entity(v_map, val_ids, ground_truth, threshold)
        
        gap = train_m['F0.5'] - val_m['F0.5']
        
        row = {
            'experiment_id': exp_id,
            'boosting_type': params.get('boosting_type', 'gbdt'),
            'data_sample_strategy': params.get('data_sample_strategy', 'bagging'),
            'learning_rate': params.get('learning_rate', 0.05),
            'num_iterations': params.get('n_estimators', 300),
            'best_iteration': best_iter,
            'num_leaves': params.get('num_leaves', 31),
            'max_depth': params.get('max_depth', -1),
            'min_data_in_leaf': params.get('min_child_samples', 20),
            'min_split_gain': params.get('min_split_gain', 0.0),
            'feature_fraction': params.get('colsample_bytree', 1.0),
            'bagging_fraction': params.get('subsample', 1.0),
            'bagging_freq': params.get('subsample_freq', 0),
            'lambda_l1': params.get('reg_alpha', 0.0),
            'lambda_l2': params.get('reg_lambda', 0.0),
            'threshold': threshold,
            'train_f0_5': train_m['F0.5'],
            'validation_f0_5': val_m['F0.5'],
            'macro_f0_5': macro_f05,
            'precision': val_m['Precision'],
            'recall': val_m['Recall'],
            'TP': val_m['TP'],
            'FP': val_m['FP'],
            'FN': val_m['FN'],
            'TN': val_m['TN'],
            'train_validation_gap': gap,
            'fit_time_s': round(time.time() - t_exp, 2)
        }
        experiments.append(row)
        
        print(f"\n[{exp_id}] F0.5: {val_m['F0.5']:.5f} | Macro-F0.5: {macro_f05:.5f} | Prec: {val_m['Precision']:.4f} | Rec: {val_m['Recall']:.4f} | FP: {val_m['FP']:,} | FN: {val_m['FN']:,} | Gap: {gap:+.4f} ({time.time()-t_exp:.1f}s)", flush=True)
        print(f"  Confusion Matrix: TP={val_m['TP']:,} | FP={val_m['FP']:,} | FN={val_m['FN']:,} | TN={val_m['TN']:,}", flush=True)
        return clf, val_m, macro_f05, v_map
        
    # =========================================================================
    # STEP 0: ESTABLISH BASELINE
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 0: ESTABLISHING BASELINE")
    print("="*70, flush=True)
    
    base_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 300,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    base_clf, base_m, base_macro, base_vmap = run_experiment("EXP_00_BASELINE", base_params, threshold=0.70)
    
    print("\n" + "-"*50)
    print("BASELINE METRICS RECORDED:")
    print(f"  TP: {base_m['TP']:,} | FP: {base_m['FP']:,} | FN: {base_m['FN']:,} | TN: {base_m['TN']:,}")
    print(f"  Precision: {base_m['Precision']:.5f} | Recall: {base_m['Recall']:.5f}")
    print(f"  Pairwise F0.5: {base_m['F0.5']:.5f} | Macro F0.5: {base_macro:.5f}")
    print(f"  Accuracy: {base_m['Accuracy']:.5f} | F1: {base_m['F1']:.5f}")
    print("-"*50, flush=True)
    
    # =========================================================================
    # STEP 4: TUNING TREE GROWTH & LEAF PARAMETERS (max_depth & num_leaves)
    # Respecting num_leaves <= 2^max_depth
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: TUNING MAX_DEPTH & NUM_LEAVES (STRUCTURAL COMPLEXITY)")
    print("="*70, flush=True)
    
    depth_configs = [
        (4, 15),
        (6, 31),
        (6, 45),
        (8, 63),
        (8, 127),
        (10, 127),
        (12, 255)
    ]
    
    best_depth_params = base_params.copy()
    best_depth_f05 = base_macro
    
    for d, leaves in depth_configs:
        p = base_params.copy()
        p['max_depth'] = d
        p['num_leaves'] = leaves
        exp_name = f"EXP_TREE_D{d}_L{leaves}"
        _, _, score, _ = run_experiment(exp_name, p, threshold=0.70)
        if score > best_depth_f05:
            best_depth_f05 = score
            best_depth_params = p.copy()
            
    # =========================================================================
    # STEP 5: TUNING MIN_CHILD_SAMPLES / MIN_DATA_IN_LEAF
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: TUNING MIN_CHILD_SAMPLES (OVERFITTING CONTROL ON LEAF SPLITS)")
    print("="*70, flush=True)
    
    best_leaf_params = best_depth_params.copy()
    best_leaf_f05 = best_depth_f05
    
    for min_leaf in [10, 20, 40, 80, 150]:
        p = best_depth_params.copy()
        p['min_child_samples'] = min_leaf
        exp_name = f"EXP_LEAF_MIN{min_leaf}"
        _, _, score, _ = run_experiment(exp_name, p, threshold=0.70)
        if score > best_leaf_f05:
            best_leaf_f05 = score
            best_leaf_params = p.copy()
            
    # =========================================================================
    # STEP 6: TUNING LEARNING RATE & BOOSTING ROUNDS WITH EARLY STOPPING
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: TUNING LEARNING RATE & N_ESTIMATORS")
    print("="*70, flush=True)
    
    best_lr_params = best_leaf_params.copy()
    best_lr_f05 = best_leaf_f05
    
    for lr, n_est in [(0.10, 250), (0.05, 450), (0.03, 600), (0.02, 800)]:
        p = best_leaf_params.copy()
        p['learning_rate'] = lr
        p['n_estimators'] = n_est
        exp_name = f"EXP_LR_{lr}_EST_{n_est}"
        _, _, score, _ = run_experiment(exp_name, p, threshold=0.70)
        if score > best_lr_f05:
            best_lr_f05 = score
            best_lr_params = p.copy()
            
    # =========================================================================
    # STEP 7: REGULARIZATION (L1, L2, min_split_gain)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 7: REGULARIZATION (L1/L2 & MIN_SPLIT_GAIN)")
    print("="*70, flush=True)
    
    best_reg_params = best_lr_params.copy()
    best_reg_f05 = best_lr_f05
    
    reg_configs = [
        (0.0, 1.0, 0.0),
        (0.1, 2.0, 0.0),
        (0.5, 5.0, 0.05),
        (1.0, 10.0, 0.1),
        (2.0, 20.0, 0.2)
    ]
    
    for l1, l2, gain in reg_configs:
        p = best_lr_params.copy()
        p['reg_alpha'] = l1
        p['reg_lambda'] = l2
        p['min_split_gain'] = gain
        exp_name = f"EXP_REG_L1_{l1}_L2_{l2}_G_{gain}"
        _, _, score, _ = run_experiment(exp_name, p, threshold=0.70)
        if score > best_reg_f05:
            best_reg_f05 = score
            best_reg_params = p.copy()
            
    # =========================================================================
    # STEP 8: FEATURE & BAGGING FRACTIONS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 8: SUBSAMPLING & FEATURE FRACTION (SUBSPACE REGULARIZATION)")
    print("="*70, flush=True)
    
    best_sub_params = best_reg_params.copy()
    best_sub_f05 = best_reg_f05
    
    sub_configs = [
        (0.9, 0.9, 1),
        (0.8, 0.85, 2),
        (0.75, 0.8, 3),
        (0.85, 1.0, 0)
    ]
    
    for col, sub, freq in sub_configs:
        p = best_reg_params.copy()
        p['colsample_bytree'] = col
        p['subsample'] = sub
        p['subsample_freq'] = freq
        exp_name = f"EXP_SUBSPACE_COL{col}_SUB{sub}"
        _, _, score, _ = run_experiment(exp_name, p, threshold=0.70)
        if score > best_sub_f05:
            best_sub_f05 = score
            best_sub_params = p.copy()
            
    # =========================================================================
    # STEP 9: BOOSTING TYPE / SAMPLING STRATEGY (GBDT vs DART vs GOSS)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 9: COMPARING BOOSTING & SAMPLING STRATEGIES")
    print("="*70, flush=True)
    
    # 1. DART
    p_dart = best_sub_params.copy()
    p_dart['boosting_type'] = 'dart'
    p_dart.pop('subsample', None)
    p_dart.pop('subsample_freq', None)
    p_dart['n_estimators'] = min(p_dart['n_estimators'], 400)
    run_experiment("EXP_BOOST_DART", p_dart, threshold=0.70)
    
    # 2. GOSS
    p_goss = best_sub_params.copy()
    p_goss['boosting_type'] = 'gbdt'
    p_goss['data_sample_strategy'] = 'goss'
    p_goss.pop('subsample', None)
    p_goss.pop('subsample_freq', None)
    run_experiment("EXP_SAMPLING_GOSS", p_goss, threshold=0.70)
    
    # =========================================================================
    # STEP 10: DECISION THRESHOLD CALIBRATION FOR F0.5
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 10: DECISION THRESHOLD GRID CALIBRATION FOR MACRO-F0.5")
    print("="*70, flush=True)
    
    final_clf = lgb.LGBMClassifier(**best_sub_params)
    final_clf.fit(X_train, y_train)
    
    val_probs = final_clf.predict_proba(X_val)[:, 1]
    v_map = defaultdict(list)
    for (s1_id, tgt_id), p in zip(val_pair_index, val_probs):
        v_map[s1_id].append((tgt_id, p))
        
    print("\nThreshold Scan:")
    print("Thresh | Pairwise Prec | Pairwise Rec | Pairwise F0.5 | Macro F0.5 |   TP   |   FP   |   FN   |   TN   ")
    print("-" * 105)
    
    best_t, best_macro_score = 0.70, 0.0
    best_val_matrix = None
    
    for t in np.arange(0.40, 0.96, 0.05):
        val_preds_t = (val_probs >= t).astype(int)
        m_t = compute_metrics_and_confusion_matrix(y_val, val_preds_t)
        macro_t = compute_macro_f05_per_entity(v_map, val_ids, ground_truth, t)
        
        print(f" {t:.2f}  |    {m_t['Precision']:.4f}     |    {m_t['Recall']:.4f}    |    {m_t['F0.5']:.5f}    |  {macro_t:.5f}   | {m_t['TP']:6,d} | {m_t['FP']:6,d} | {m_t['FN']:6,d} | {m_t['TN']:6,d}")
        
        row = {
            'experiment_id': f"EXP_THRESH_{t:.2f}",
            'boosting_type': best_sub_params.get('boosting_type', 'gbdt'),
            'data_sample_strategy': best_sub_params.get('data_sample_strategy', 'bagging'),
            'learning_rate': best_sub_params.get('learning_rate', 0.05),
            'num_iterations': best_sub_params.get('n_estimators', 300),
            'best_iteration': getattr(final_clf, 'best_iteration_', best_sub_params.get('n_estimators', 300)),
            'num_leaves': best_sub_params.get('num_leaves', 31),
            'max_depth': best_sub_params.get('max_depth', -1),
            'min_data_in_leaf': best_sub_params.get('min_child_samples', 20),
            'min_split_gain': best_sub_params.get('min_split_gain', 0.0),
            'feature_fraction': best_sub_params.get('colsample_bytree', 1.0),
            'bagging_fraction': best_sub_params.get('subsample', 1.0),
            'bagging_freq': best_sub_params.get('subsample_freq', 0),
            'lambda_l1': best_sub_params.get('reg_alpha', 0.0),
            'lambda_l2': best_sub_params.get('reg_lambda', 0.0),
            'threshold': round(t, 2),
            'train_f0_5': 0.0,
            'validation_f0_5': m_t['F0.5'],
            'macro_f0_5': macro_t,
            'precision': m_t['Precision'],
            'recall': m_t['Recall'],
            'TP': m_t['TP'],
            'FP': m_t['FP'],
            'FN': m_t['FN'],
            'TN': m_t['TN'],
            'train_validation_gap': 0.0,
            'fit_time_s': 0.0
        }
        experiments.append(row)
        
        if macro_t > best_macro_score:
            best_macro_score = macro_t
            best_t = round(t, 2)
            best_val_matrix = m_t
            
    # Save experiments log
    df_exp = pd.DataFrame(experiments)
    df_exp.sort_values(by='macro_f0_5', ascending=False, inplace=True)
    df_exp.to_csv(args.out_csv, index=False)
    print(f"\nSaved {len(experiments)} experiment records to {args.out_csv}")
    
    # Save best model
    joblib.dump({'model': final_clf, 'threshold': best_t, 'params': best_sub_params}, "model_optimized.joblib")
    print(f"Optimized model serialized to model_optimized.joblib (Optimal Threshold = {best_t:.2f})")
    
    # Print Final Summary Table
    print("\n" + "="*70)
    print("FINAL TUNING REPORT & BEST VALIDATION RESULT")
    print("="*70)
    print(f"BASELINE MACRO-F0.5:        {base_macro:.5f}")
    print(f"BEST VALIDATION MACRO-F0.5: {best_macro_score:.5f}")
    print(f"IMPROVEMENT:                +{best_macro_score - base_macro:.5f}")
    print(f"Optimal Decision Threshold: {best_t:.2f}")
    print(f"Precision:                  {best_val_matrix['Precision']:.5f}")
    print(f"Recall:                     {best_val_matrix['Recall']:.5f}")
    print(f"Confusion Matrix:           TP={best_val_matrix['TP']:,}, FP={best_val_matrix['FP']:,}, FN={best_val_matrix['FN']:,}, TN={best_val_matrix['TN']:,}")
    print("\nBest Parameter Configuration:")
    for k, v in best_sub_params.items():
        print(f"  {k:25}: {v}")
    print("="*70, flush=True)

if __name__ == "__main__":
    main()
