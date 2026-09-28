import os
import sys
import time
import math
import string
import unicodedata
import argparse
from collections import defaultdict
import numpy as np
import lightgbm as lgb
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

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

HANDCRAFTED_FEATURE_NAMES = [
    'n_ratio', 'n_part_ratio', 'n_tok_sort', 'n_tok_set',
    'nc_tok_sort', 'nc_tok_set', 'exact_name', 'tok_jaccard', 'char3_jaccard',
    'len_diff', 'len_ratio',
    'a_tok_sort', 'a_tok_set', 'ac_tok_sort', 'ac_tok_set', 'addr_empty',
    'num_cnt_s1', 'num_cnt_tgt', 'num_overlap', 'num_prec', 'num_rec',
    'comb_tok_set', 'is_s2', 'block_rank', 'block_score', 'score_diff',
    'exact_addr', 'prefix_match', 'num_exact', 'tok_diff'
]

E5_FEATURE_NAMES = [
    'name_e5_cosine', 'addr_e5_cosine', 'comb_e5_cosine', 'name_e5_l2', 'comb_e5_l2'
]

ALL_HYBRID_FEATURE_NAMES = HANDCRAFTED_FEATURE_NAMES + E5_FEATURE_NAMES

def extract_handcrafted_features(name1, addr1, nc1, ca1, num1,
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

def compute_f05(prec, rec):
    if prec <= 0.0 or rec <= 0.0:
        return 0.0
    beta_sq = 0.25
    return (1.0 + beta_sq) * (prec * rec) / (beta_sq * prec + rec)

def evaluate_thresholds(y_true, y_probs, pair_meta, ground_truth, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.40, 0.95, 29)
        
    best_f05 = -1.0
    best_thresh = 0.50
    best_prec = 0.0
    best_rec = 0.0
    best_cm = None
    best_pair_metrics = None
    
    total_true_links = sum(len(m) for m in ground_truth.values())
    
    for tau in thresholds:
        s1_predictions = defaultdict(set)
        tp = fp = fn = tn = 0
        
        for idx, (prob, (s1_id, tgt_id)) in enumerate(zip(y_probs, pair_meta)):
            actual = y_true[idx]
            pred = 1 if prob >= tau else 0
            
            if pred == 1:
                s1_predictions[s1_id].add(tgt_id)
                if actual == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if actual == 1:
                    fn += 1
                else:
                    tn += 1
                    
        pred_links = sum(len(m) for m in s1_predictions.values())
        correct_links = sum(len(s1_predictions[s1] & ground_truth[s1]) for s1 in ground_truth if s1 in s1_predictions)
        
        prec = correct_links / pred_links if pred_links > 0 else 1.0
        rec = correct_links / total_true_links if total_true_links > 0 else 0.0
        f05 = compute_f05(prec, rec)
        
        pair_acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        pair_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        pair_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = tau
            best_prec = prec
            best_rec = rec
            best_cm = (tn, fp, fn, tp)
            best_pair_metrics = (pair_acc, pair_prec, pair_rec)
            
    return {
        'best_threshold': best_thresh,
        'precision': best_prec,
        'recall': best_rec,
        'f05': best_f05,
        'cm': best_cm,
        'pair_metrics': best_pair_metrics
    }

def print_confusion_matrix(cm, title="Validation Set Confusion Matrix"):
    tn, fp, fn, tp = cm
    total = tn + fp + fn + tp
    print(f"\n--- {title} ---")
    print(f"Format: [[TN, FP], [FN, TP]]")
    print(f"[[{tn:7,d} (TN), {fp:7,d} (FP)],")
    print(f" [{fn:7,d} (FN), {tp:7,d} (TP)]]")
    print(f"Total evaluated pairs: {total:,}")
    print(f"True Positives  (TP): {tp:7,d} ({tp/total*100:5.2f}%)")
    print(f"False Positives (FP): {fp:7,d} ({fp/total*100:5.2f}%)")
    print(f"False Negatives (FN): {fn:7,d} ({fn/total*100:5.2f}%)")
    print(f"True Negatives  (TN): {tn:7,d} ({tn/total*100:5.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Hybrid Entity Resolution Ablation Suite")
    parser.add_argument("--train-dir", type=str, default="dataset/train", help="Path to training data directory")
    parser.add_argument("--num-train", type=int, default=15000, help="Number of training S1 entities")
    parser.add_argument("--num-val", type=int, default=5000, help="Number of validation S1 entities")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding inference")
    parser.add_argument("--fine-tune", action="store_true", help="Run Bi-Encoder Fine-Tuning experiment (Exp G)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("HYBRID ENTITY RESOLUTION: SYSTEMATIC ABLATION & BENCHMARK SUITE")
    print("=" * 80)
    print(f"Configuration: Train S1 = {args.num_train:,} | Val S1 = {args.num_val:,} | Embedding Batch Size = {args.batch_size}")
    
    # 1. Load Ground Truth
    gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")
    all_gt = {}
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                s1_id = parts[0]
                matches = set(parts[1].split(",")) if parts[1].strip() else set()
                all_gt[s1_id] = matches
            if len(all_gt) >= (args.num_train + args.num_val):
                break
                
    all_s1_keys = list(all_gt.keys())
    train_s1_keys = set(all_s1_keys[:args.num_train])
    val_s1_keys = set(all_s1_keys[args.num_train:args.num_train + args.num_val])
    
    gt_train = {k: all_gt[k] for k in train_s1_keys}
    gt_val = {k: all_gt[k] for k in val_s1_keys}
    
    all_true_targets = {m for matches in all_gt.values() for m in matches}
    print(f"Ground Truth loaded: {len(gt_train):,} Train entities ({sum(len(v) for v in gt_train.values()):,} true links), {len(gt_val):,} Val entities ({sum(len(v) for v in gt_val.values()):,} true links)")
    
    # 2. Load S1 Records
    s1_records = {}
    with open(os.path.join(args.train_dir, "train_source1.tsv"), "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[0] in all_gt:
                s1_records[parts[0]] = (parts[3], extract_record_fast(parts[1], parts[2]))
                
    # 3. Load Target Records
    target_records = {}
    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        count = 0
        with open(os.path.join(args.train_dir, fname), "r", encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    if parts[0] in all_true_targets or count < 180000:
                        target_records[parts[0]] = (parts[3], extract_record_fast(parts[1], parts[2]))
                        count += 1
                        
    print(f"Records loaded: {len(s1_records):,} S1 entities, {len(target_records):,} Target entities")
    
    # 4. Build Fast Blocking Inverted Index
    t_idx = time.time()
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
                    
    print(f"Candidate blocking index built in {time.time()-t_idx:.2f}s")
    
    # Helper to generate candidate pairs for a set of S1 entities
    def generate_candidate_pairs(s1_subset, gt_subset):
        pairs = [] # (s1_id, tgt_id, base_score, rank, top1_score, label)
        for s1_id in s1_subset:
            if s1_id not in s1_records:
                continue
            country, s1_rec = s1_records[s1_id]
            name1, addr1, nc1, core_n, ca1, core_a, nums1 = s1_rec
            true_m = gt_subset[s1_id]
            
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
                if eid not in target_records:
                    continue
                _, cand_rec = target_records[eid]
                nsim = fuzz.token_set_ratio(name1, cand_rec[0])
                asim = fuzz.token_set_ratio(addr1, cand_rec[1]) if addr1 and cand_rec[1] else 40.0
                scored.append((base_score + 0.55 * nsim + 0.45 * asim, eid))
                
            scored.sort(key=lambda x: x[0], reverse=True)
            top_cands = scored[:10]
            top1_score = top_cands[0][0]
            
            for rank, (b_score, tgt_id) in enumerate(top_cands, 1):
                label = 1 if tgt_id in true_m else 0
                pairs.append((s1_id, tgt_id, b_score, rank, top1_score, label))
        return pairs

    print("\nGenerating Candidate Pairs via Blocking...", flush=True)
    t_pairs = time.time()
    train_pairs_raw = generate_candidate_pairs(train_s1_keys, gt_train)
    val_pairs_raw = generate_candidate_pairs(val_s1_keys, gt_val)
    print(f"Generated {len(train_pairs_raw):,} Train pairs (Positives: {sum(p[5] for p in train_pairs_raw):,})")
    print(f"Generated {len(val_pairs_raw):,} Val pairs (Positives: {sum(p[5] for p in val_pairs_raw):,}) in {time.time()-t_pairs:.2f}s")
    
    # 5. Extract Handcrafted Features
    print("\nComputing 30 Handcrafted Lexical/Structural Features...", flush=True)
    t_hc = time.time()
    
    X_hc_train = []
    y_train = []
    meta_train = []
    for s1_id, tgt_id, b_score, rank, top1_score, label in train_pairs_raw:
        _, s1_rec = s1_records[s1_id]
        _, cand_rec = target_records[tgt_id]
        feat = extract_handcrafted_features(
            s1_rec[0], s1_rec[1], s1_rec[2], s1_rec[4], s1_rec[6],
            tgt_id, cand_rec[0], cand_rec[1], cand_rec[2], cand_rec[4], cand_rec[6],
            b_score, rank, top1_score
        )
        X_hc_train.append(feat)
        y_train.append(label)
        meta_train.append((s1_id, tgt_id))
        
    X_hc_val = []
    y_val = []
    meta_val = []
    for s1_id, tgt_id, b_score, rank, top1_score, label in val_pairs_raw:
        _, s1_rec = s1_records[s1_id]
        _, cand_rec = target_records[tgt_id]
        feat = extract_handcrafted_features(
            s1_rec[0], s1_rec[1], s1_rec[2], s1_rec[4], s1_rec[6],
            tgt_id, cand_rec[0], cand_rec[1], cand_rec[2], cand_rec[4], cand_rec[6],
            b_score, rank, top1_score
        )
        X_hc_val.append(feat)
        y_val.append(label)
        meta_val.append((s1_id, tgt_id))
        
    X_hc_train = np.array(X_hc_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_hc_val = np.array(X_hc_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)
    print(f"Handcrafted features extracted in {time.time()-t_hc:.2f}s. Shape: Train {X_hc_train.shape}, Val {X_hc_val.shape}")

    # 6. Precompute E5 Neural Embeddings (Cached per unique entity)
    model_snapshot = r'C:\Users\Siva Santhiya\.cache\huggingface\hub\models--intfloat--multilingual-e5-small\snapshots\614241f622f53c4eeff9890bdc4f31cfecc418b3'
    print(f"\nLoading Multilingual E5 Model from local snapshot...", flush=True)
    t_model = time.time()
    embedder = SentenceTransformer(model_snapshot)
    print(f"E5 Model loaded in {time.time()-t_model:.2f}s")
    
    # Collect all needed texts for S1 and Target
    used_s1 = set()
    used_tgt = set()
    for s1_id, tgt_id, _, _, _, _ in train_pairs_raw + val_pairs_raw:
        used_s1.add(s1_id)
        used_tgt.add(tgt_id)
        
    print(f"Extracting & Caching E5 Embeddings for {len(used_s1):,} S1 entities and {len(used_tgt):,} Target entities...", flush=True)
    t_emb = time.time()
    
    # S1 Queries: Prompt = "query: "
    s1_id_list = list(used_s1)
    s1_name_texts = [f"query: {s1_records[sid][1][2]}" for sid in s1_id_list]
    s1_addr_texts = [f"query: {s1_records[sid][1][4]}" for sid in s1_id_list]
    s1_comb_texts = [f"query: {s1_records[sid][1][2]} {s1_records[sid][1][4]}".strip() for sid in s1_id_list]
    
    s1_name_embs = embedder.encode(s1_name_texts, batch_size=args.batch_size, show_progress_bar=False, normalize_embeddings=True)
    s1_addr_embs = embedder.encode(s1_addr_texts, batch_size=args.batch_size, show_progress_bar=False, normalize_embeddings=True)
    s1_comb_embs = embedder.encode(s1_comb_texts, batch_size=args.batch_size, show_progress_bar=False, normalize_embeddings=True)
    
    s1_emb_dict = {}
    for i, sid in enumerate(s1_id_list):
        s1_emb_dict[sid] = (s1_name_embs[i], s1_addr_embs[i], s1_comb_embs[i])
        
    # Target Passages: Prompt = "passage: "
    tgt_id_list = list(used_tgt)
    tgt_name_texts = [f"passage: {target_records[tid][1][2]}" for tid in tgt_id_list]
    tgt_addr_texts = [f"passage: {target_records[tid][1][4]}" for tid in tgt_id_list]
    tgt_comb_texts = [f"passage: {target_records[tid][1][2]} {target_records[tid][1][4]}".strip() for tid in tgt_id_list]
    
    tgt_name_embs = embedder.encode(tgt_name_texts, batch_size=args.batch_size, show_progress_bar=False, normalize_embeddings=True)
    tgt_addr_embs = embedder.encode(tgt_addr_texts, batch_size=args.batch_size, show_progress_bar=False, normalize_embeddings=True)
    tgt_comb_embs = embedder.encode(tgt_comb_texts, batch_size=args.batch_size, show_progress_bar=False, normalize_embeddings=True)
    
    tgt_emb_dict = {}
    for i, tid in enumerate(tgt_id_list):
        tgt_emb_dict[tid] = (tgt_name_embs[i], tgt_addr_embs[i], tgt_comb_embs[i])
        
    print(f"Cached all E5 embeddings in {time.time()-t_emb:.2f}s")
    
    # 7. Compute E5 Features for Train and Val
    def compute_e5_features(meta_list):
        feats = []
        for s1_id, tgt_id in meta_list:
            s1_n, s1_a, s1_c = s1_emb_dict[s1_id]
            tgt_n, tgt_a, tgt_c = tgt_emb_dict[tgt_id]
            
            # Cosine similarities (vectors are L2-normalized, so dot product = cosine)
            name_cos = float(np.dot(s1_n, tgt_n))
            addr_cos = float(np.dot(s1_a, tgt_a))
            comb_cos = float(np.dot(s1_c, tgt_c))
            
            # L2 distances
            name_l2 = float(np.linalg.norm(s1_n - tgt_n))
            comb_l2 = float(np.linalg.norm(s1_c - tgt_c))
            
            feats.append([name_cos, addr_cos, comb_cos, name_l2, comb_l2])
        return np.array(feats, dtype=np.float32)

    X_e5_train = compute_e5_features(meta_train)
    X_e5_val = compute_e5_features(meta_val)
    
    # Combined Full Hybrid Feature matrices
    X_hybrid_train = np.hstack([X_hc_train, X_e5_train])
    X_hybrid_val = np.hstack([X_hc_val, X_e5_val])
    
    print(f"Full Hybrid matrices ready: Train {X_hybrid_train.shape}, Val {X_hybrid_val.shape}")

    # =========================================================================
    # SYSTEMATIC ABLATIONS (EXP A - EXP F)
    # =========================================================================
    experiments = {
        'Exp A: Handcrafted Only (Baseline)': {
            'features': list(range(0, 30)),
            'feature_names': HANDCRAFTED_FEATURE_NAMES,
            'desc': '30 Lexical / Structural / Numerical Handcrafted Features'
        },
        'Exp B: E5 Embeddings Only': {
            'features': list(range(30, 35)),
            'feature_names': E5_FEATURE_NAMES,
            'desc': '5 Semantic E5 Features (Name/Addr/Comb Cosine & L2)'
        },
        'Exp C: Handcrafted + Name E5': {
            'features': list(range(0, 30)) + [30, 33], # HC + name_cos + name_l2
            'feature_names': HANDCRAFTED_FEATURE_NAMES + ['name_e5_cosine', 'name_e5_l2'],
            'desc': '30 Handcrafted + Name E5 Cosine & L2 (32 features)'
        },
        'Exp D: Handcrafted + Address E5': {
            'features': list(range(0, 30)) + [31], # HC + addr_cos
            'feature_names': HANDCRAFTED_FEATURE_NAMES + ['addr_e5_cosine'],
            'desc': '30 Handcrafted + Address E5 Cosine (31 features)'
        },
        'Exp E: Handcrafted + Name + Address E5': {
            'features': list(range(0, 30)) + [30, 31, 33], # HC + name_cos + addr_cos + name_l2
            'feature_names': HANDCRAFTED_FEATURE_NAMES + ['name_e5_cosine', 'addr_e5_cosine', 'name_e5_l2'],
            'desc': '30 Handcrafted + Name & Address E5 (33 features)'
        },
        'Exp F: Full Hybrid (HC + Name + Addr + Comb E5)': {
            'features': list(range(0, 35)),
            'feature_names': ALL_HYBRID_FEATURE_NAMES,
            'desc': 'Full 35D Hybrid: 30 Handcrafted + All 5 E5 Embeddings'
        }
    }
    
    results = {}
    
    print("\n" + "=" * 80)
    print("RUNNING CONTROLLED ABLATIONS (Exp A through Exp F)")
    print("=" * 80)
    
    for exp_id, exp_cfg in experiments.items():
        feat_indices = exp_cfg['features']
        feat_names = exp_cfg['feature_names']
        
        X_tr = X_hybrid_train[:, feat_indices]
        X_va = X_hybrid_val[:, feat_indices]
        
        # Train LightGBM model with identical parameters
        clf = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            n_estimators=450,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=8,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        t0 = time.time()
        clf.fit(X_tr, y_train)
        train_time = time.time() - t0
        
        # Train evaluation
        probs_train = clf.predict_proba(X_tr)[:, 1]
        eval_train = evaluate_thresholds(y_train, probs_train, meta_train, gt_train)
        
        # Val evaluation
        probs_val = clf.predict_proba(X_va)[:, 1]
        eval_val = evaluate_thresholds(y_val, probs_val, meta_val, gt_val)
        
        gap = abs(eval_train['f05'] - eval_val['f05'])
        
        results[exp_id] = {
            'dim': len(feat_indices),
            'opt_thresh': eval_val['best_threshold'],
            'val_prec': eval_val['precision'],
            'val_rec': eval_val['recall'],
            'val_f05': eval_val['f05'],
            'train_f05': eval_train['f05'],
            'gap': gap,
            'cm': eval_val['cm'],
            'pair_metrics': eval_val['pair_metrics'],
            'train_time': train_time,
            'model': clf
        }
        
        print(f"\n[{exp_id}] ({len(feat_indices)} feats | Fit: {train_time:.2f}s)")
        print(f"  -> Optimal Threshold: {eval_val['best_threshold']:.2f}")
        print(f"  -> Validation Match Precision: {eval_val['precision']*100:.2f}% | Recall: {eval_val['recall']*100:.2f}% | Macro-F0.5: {eval_val['f05']:.5f}")
        print(f"  -> Train F0.5: {eval_train['f05']:.5f} | Generalization Gap (|Train-Val|): {gap:.5f}")
        print_confusion_matrix(eval_val['cm'], title=f"Confusion Matrix: {exp_id}")

    # =========================================================================
    # SUMMARY COMPARISON TABLE
    # =========================================================================
    print("\n" + "=" * 95)
    print("ABLATION STUDY SUMMARY COMPARISON TABLE")
    print("=" * 95)
    print(f"{'Experiment':<46} | {'Dim':<4} | {'Thresh':<6} | {'Precision':<9} | {'Recall':<8} | {'Val F0.5':<10} | {'Gap':<7}")
    print("-" * 95)
    
    baseline_f05 = results['Exp A: Handcrafted Only (Baseline)']['val_f05']
    for exp_id, res in results.items():
        delta = res['val_f05'] - baseline_f05
        delta_str = f"({'+' if delta >= 0 else ''}{delta:.5f})" if exp_id != 'Exp A: Handcrafted Only (Baseline)' else "(Baseline)"
        print(f"{exp_id:<46} | {res['dim']:<4} | {res['opt_thresh']:<6.2f} | {res['val_prec']*100:6.2f}%  | {res['val_rec']*100:5.2f}% | {res['val_f05']:<8.5f} {delta_str:<10} | {res['gap']:.5f}")
    print("=" * 95)

    # =========================================================================
    # DETAILED FEATURE IMPORTANCES FOR FULL HYBRID MODEL
    # =========================================================================
    best_clf = results['Exp F: Full Hybrid (HC + Name + Addr + Comb E5)']['model']
    importances_gain = best_clf.booster_.feature_importance(importance_type='gain')
    importances_split = best_clf.booster_.feature_importance(importance_type='split')
    
    sorted_idx = np.argsort(importances_gain)[::-1]
    
    print("\n" + "=" * 80)
    print("FULL HYBRID MODEL: TOP FEATURE IMPORTANCES (BY GAIN & SPLIT)")
    print("=" * 80)
    print(f"{'Rank':<5} | {'Feature Name':<25} | {'Gain Importance':<18} | {'Split Count':<12} | {'Category'}")
    print("-" * 80)
    for rank, idx in enumerate(sorted_idx[:20], 1):
        fname = ALL_HYBRID_FEATURE_NAMES[idx]
        cat = "E5 Semantic Embedding" if fname in E5_FEATURE_NAMES else "Handcrafted Lexical"
        print(f"{rank:<5} | {fname:<25} | {importances_gain[idx]:<18.2f} | {importances_split[idx]:<12} | {cat}")
    print("=" * 80)

if __name__ == "__main__":
    main()
