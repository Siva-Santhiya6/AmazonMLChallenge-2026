import sys
import os
import gc
import time
import string
import unicodedata
import argparse
from collections import defaultdict
import numpy as np
import lightgbm as lgb
import joblib

sys.stdout.reconfigure(encoding='utf-8')

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import STOP_WORDS_NAME, STOP_WORDS_ADDR, INDIC_TRANSLIT, FEATURE_NAMES, DEFAULT_TOP_K, OPTIMAL_MATCH_THRESHOLD
from rapidfuzz import fuzz

PUNCT_TRANS = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

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

def extract_record_fast(name, addr):
    cn = clean_fast(name)
    ca = clean_fast(addr)
    
    n_toks = cn.split()
    core_n = tuple(t for t in n_toks if t not in STOP_WORDS_NAME and len(t) > 1) or tuple(t for t in n_toks if len(t) > 1) or tuple(n_toks)
        
    a_toks = ca.split()
    core_a = tuple(t for t in a_toks if t not in STOP_WORDS_ADDR and len(t) > 2)
    nums = tuple(t for t in a_toks if t.isdigit())
    
    return (name, addr, cn, core_n, ca, core_a, nums)

def extract_pairwise_features_fast(name1, addr1, nc1, ca1, num1,
                                   tgt_id, name2, addr2, nc2, ca2, num2,
                                   block_score, block_rank, top1_score):
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
    
    # Enhanced distinguishing features
    exact_addr = 1.0 if ca1 == ca2 and len(ca1) > 0 else 0.0
    p1 = nc1[:4] if len(nc1) >= 4 else nc1
    p2 = nc2[:4] if len(nc2) >= 4 else nc2
    prefix_match = 1.0 if p1 == p2 and len(p1) >= 3 else 0.0
    num_exact = 1.0 if num1 and num2 and num1 == num2 else 0.0
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

def train_model_quick(train_dir: str, num_samples: int = 20000, model_save_path: str = "model.joblib"):
    """Quickly train and serialize LightGBM model on ground truth."""
    print(f"\n[1/4] Training LightGBM Matcher on {num_samples:,} Ground Truth Entities...", flush=True)
    t0 = time.time()
    
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    ground_truth = {}
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                s1_id = parts[0]
                matches = set(parts[1].split(",")) if parts[1].strip() else set()
                ground_truth[s1_id] = matches
            if len(ground_truth) >= num_samples:
                break
                
    s1_set = set(ground_truth.keys())
    all_true_targets = {m for matches in ground_truth.values() for m in matches}
    
    s1_records = {}
    with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[0] in s1_set:
                s1_records[parts[0]] = (parts[3], extract_record_fast(parts[1], parts[2]))
                
    target_records = {}
    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        count = 0
        with open(os.path.join(train_dir, fname), "r", encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    if parts[0] in all_true_targets or count < 150000:
                        target_records[parts[0]] = (parts[3], extract_record_fast(parts[1], parts[2]))
                        count += 1
                        
    print(f"Loaded {len(s1_records):,} S1 and {len(target_records):,} targets in {time.time()-t0:.2f}s", flush=True)
    
    idx_exact = defaultdict(list)
    idx_token = defaultdict(list)
    idx_num_street = defaultdict(list)
    
    for eid, (country, rec) in target_records.items():
        _, _, nc, core_n, _, core_a, nums = rec
        if nc:
            idx_exact[(country, nc)].append(eid)
        for t in core_n:
            if len(t) >= 3:
                idx_token[(country, t)].append(eid)
        for num in nums:
            for at in core_a:
                if len(at) >= 3:
                    idx_num_street[(country, num, at)].append(eid)
                    
    X, y = [], []
    for s1_id, (country, s1_rec) in s1_records.items():
        name1, addr1, nc1, core_n, ca1, core_a, nums1 = s1_rec
        true_m = ground_truth[s1_id]
        
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
        for num in nums1:
            for at in core_a:
                if len(at) >= 3:
                    hits = idx_num_street.get((country, num, at), ())
                    if len(hits) <= 200:
                        w = 30.0 / (1.0 + 0.05 * len(hits))
                        for eid in hits:
                            cand_scores[eid] += w
                            
        if not cand_scores:
            continue
            
        if len(cand_scores) > 30:
            top_items = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:30]
        else:
            top_items = list(cand_scores.items())
            
        scored = []
        for eid, base_score in top_items:
            _, cand_rec = target_records[eid]
            nsim = fuzz.token_set_ratio(name1, cand_rec[0])
            asim = fuzz.token_set_ratio(addr1, cand_rec[1]) if addr1 and cand_rec[1] else 40.0
            scored.append((base_score + 0.55 * nsim + 0.45 * asim, eid))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        top_cands = scored[:DEFAULT_TOP_K]
        top1_score = top_cands[0][0]
        
        for rank, (b_score, tgt_id) in enumerate(top_cands, 1):
            _, cand_rec = target_records[tgt_id]
            name2, addr2, nc2, _, ca2, _, num2 = cand_rec
            feat = extract_pairwise_features_fast(
                name1, addr1, nc1, ca1, nums1,
                tgt_id, name2, addr2, nc2, ca2, num2,
                b_score, rank, top1_score
            )
            label = 1 if tgt_id in true_m else 0
            X.append(feat)
            y.append(label)
            
    X = np.array(X)
    y = np.array(y)
    print(f"Training on {len(X):,} candidate pairs (Positives: {np.sum(y):,}, Negatives: {len(y)-np.sum(y):,})...", flush=True)
    
    clf = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        n_estimators=450,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        min_child_samples=80,
        reg_alpha=0.5,
        reg_lambda=5.0,
        min_split_gain=0.05,
        colsample_bytree=0.80,
        subsample=0.85,
        subsample_freq=1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X, y)
    
    joblib.dump({'model': clf, 'threshold': OPTIMAL_MATCH_THRESHOLD}, model_save_path)
    print(f"Model successfully saved to {model_save_path} in {time.time()-t0:.2f}s total.", flush=True)
    return clf

def run_test_inference_fast(test_dir: str, output_dir: str, model_save_path: str = "model.joblib"):
    """High-speed partitioned streaming inference across all 1.73M test records."""
    print("\n[2/4] Starting Full Test Inference Engine...", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    
    clf_data = joblib.load(model_save_path)
    clf = clf_data['model']
    threshold = clf_data.get('threshold', OPTIMAL_MATCH_THRESHOLD)
    print(f"Loaded classifier (Decision Threshold = {threshold:.2f}).", flush=True)
    
    matching_out = os.path.join(output_dir, "matching_results.tsv")
    candidate_out = os.path.join(output_dir, "candidate_pairs.tsv")
    
    f_match = open(matching_out, "w", encoding="utf-8", newline="")
    f_cand = open(candidate_out, "w", encoding="utf-8", newline="")
    
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    
    t_start = time.time()
    
    # Process Country by Country to keep memory under 1.5GB
    countries = ["France", "US", "India"]
    total_all_queries = 0
    
    for country in countries:
        print(f"\n==========================================", flush=True)
        print(f"Processing Country Namespace: [{country}]", flush=True)
        print(f"==========================================", flush=True)
        t_c = time.time()
        
        # 1. Load S1 queries for this country
        s1_queries = []
        with open(os.path.join(test_dir, "test_source1.tsv"), "r", encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3] == country:
                    s1_queries.append((parts[0], extract_record_fast(parts[1], parts[2])))
                    
        total_country_queries = len(s1_queries)
        total_all_queries += total_country_queries
        print(f"  Loaded {total_country_queries:,} S1 queries in {time.time()-t_c:.2f}s", flush=True)
        if total_country_queries == 0:
            continue
            
        # 2. Load Targets for this country
        t_tgt = time.time()
        targets = {}
        for fname in ["test_source2.tsv", "test_source3.tsv"]:
            with open(os.path.join(test_dir, fname), "r", encoding="utf-8", errors="replace") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 4 and parts[3] == country:
                        targets[parts[0]] = extract_record_fast(parts[1], parts[2])
                        
        print(f"  Loaded {len(targets):,} target records in {time.time()-t_tgt:.2f}s", flush=True)
        
        # 3. Build Inverted Indices
        t_idx = time.time()
        idx_exact = defaultdict(list)
        idx_token = defaultdict(list)
        idx_prefix = defaultdict(list)
        idx_num_street = defaultdict(list)
        
        for eid, rec in targets.items():
            _, _, nc, core_n, _, core_a, nums = rec
            if nc:
                idx_exact[nc].append(eid)
            for t in core_n:
                if len(t) >= 3:
                    idx_token[t].append(eid)
                    if len(t) >= 4:
                        idx_prefix[t[:4]].append(eid)
            for num in nums:
                for at in core_a:
                    if len(at) >= 3:
                        idx_num_street[(num, at)].append(eid)
                        
        print(f"  Inverted indices built in {time.time()-t_idx:.2f}s", flush=True)
        
        # 4. Stream Queries in Batches
        BATCH_SIZE = 20000
        t_query_start = time.time()
        
        for b_idx in range(0, total_country_queries, BATCH_SIZE):
            batch_s1 = s1_queries[b_idx : b_idx + BATCH_SIZE]
            batch_feats = []
            batch_meta = []
            
            for s1_id, s1_rec in batch_s1:
                name1, addr1, nc1, core_n, ca1, core_a, num1 = s1_rec
                
                cand_scores = defaultdict(float)
                if nc1:
                    for eid in idx_exact.get(nc1, ()):
                        cand_scores[eid] += 60.0
                for t in core_n:
                    if len(t) >= 3:
                        hits = idx_token.get(t, ())
                        if len(hits) <= 300:
                            w = 20.0 / (1.0 + 0.05 * len(hits))
                            for eid in hits:
                                cand_scores[eid] += w
                for num in num1:
                    for at in core_a:
                        if len(at) >= 3:
                            hits = idx_num_street.get((num, at), ())
                            if len(hits) <= 150:
                                w = 30.0 / (1.0 + 0.05 * len(hits))
                                for eid in hits:
                                    cand_scores[eid] += w
                if len(cand_scores) < 3:
                    for t in core_n:
                        if len(t) >= 4:
                            hits = idx_prefix.get(t[:4], ())
                            if len(hits) <= 100:
                                for eid in hits:
                                    cand_scores[eid] += 8.0
                                    
                if not cand_scores:
                    batch_meta.append((s1_id, [], 0, 0))
                    continue
                    
                if len(cand_scores) > 30:
                    top_items = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:30]
                else:
                    top_items = list(cand_scores.items())
                    
                scored = []
                for eid, base_score in top_items:
                    cand_rec = targets[eid]
                    nsim = fuzz.token_set_ratio(name1, cand_rec[0])
                    asim = fuzz.token_set_ratio(addr1, cand_rec[1]) if addr1 and cand_rec[1] else 40.0
                    scored.append((base_score + 0.55 * nsim + 0.45 * asim, eid))
                    
                scored.sort(key=lambda x: x[0], reverse=True)
                top_cands = scored[:DEFAULT_TOP_K]
                top1_score = top_cands[0][0]
                cand_ids = [tgt_id for _, tgt_id in top_cands]
                
                start_i = len(batch_feats)
                for rank, (b_score, tgt_id) in enumerate(top_cands, 1):
                    cand_rec = targets[tgt_id]
                    name2, addr2, nc2, _, ca2, _, num2 = cand_rec
                    feat = extract_pairwise_features_fast(
                        name1, addr1, nc1, ca1, num1,
                        tgt_id, name2, addr2, nc2, ca2, num2,
                        b_score, rank, top1_score
                    )
                    batch_feats.append(feat)
                end_i = len(batch_feats)
                batch_meta.append((s1_id, cand_ids, start_i, end_i))
                
            # Batch ML Inference
            if batch_feats:
                probs = clf.predict_proba(np.array(batch_feats))[:, 1]
            else:
                probs = np.array([])
                
            # Stream Write TSVs
            for s1_id, cand_ids, start_i, end_i in batch_meta:
                if not cand_ids:
                    f_cand.write(f"{s1_id}\t\n")
                    f_match.write(f"{s1_id}\t\n")
                else:
                    cand_str = ",".join(cand_ids)
                    f_cand.write(f"{s1_id}\t{cand_str}\n")
                    
                    sub_probs = probs[start_i:end_i]
                    matched = [cand_ids[i] for i, p in enumerate(sub_probs) if p >= threshold]
                    match_str = ",".join(matched) if matched else ""
                    f_match.write(f"{s1_id}\t{match_str}\n")
                    
            done = min(b_idx + BATCH_SIZE, total_country_queries)
            q_speed = done / (time.time() - t_query_start + 1e-5)
            print(f"  [{country}] Progress: {done:,}/{total_country_queries:,} ({done/total_country_queries*100:.1f}%) - {q_speed:.0f} queries/s", flush=True)
            
        print(f"[{country}] Completed in {time.time()-t_c:.2f}s", flush=True)
        
        # Clean up memory completely between countries
        del targets
        del idx_exact
        del idx_token
        del idx_prefix
        del idx_num_street
        del s1_queries
        gc.collect()
        
    f_match.close()
    f_cand.close()
    print(f"\n[3/4] All {total_all_queries:,} entities processed and written in {time.time()-t_start:.2f}s total!", flush=True)

def main():
    parser = argparse.ArgumentParser(description="High-Speed Business Entity Resolution Pipeline")
    parser.add_argument("--train-dir", type=str, default="dataset/train")
    parser.add_argument("--test-dir", type=str, default="dataset/test")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--model-path", type=str, default="model.joblib")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.train or not os.path.exists(args.model_path):
        train_model_quick(args.train_dir, num_samples=20000, model_save_path=args.model_path)

    run_test_inference_fast(args.test_dir, args.output_dir, model_save_path=args.model_path)
    print("\n[4/4] Pipeline Complete!", flush=True)

if __name__ == "__main__":
    main()
