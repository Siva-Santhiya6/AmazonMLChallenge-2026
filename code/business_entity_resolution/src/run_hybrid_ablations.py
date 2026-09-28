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
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# Define Stop words & Normalization
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

def compute_f05_score(prec, rec):
    if prec <= 0.0 or rec <= 0.0:
        return 0.0
    beta_sq = 0.25
    return (1.0 + beta_sq) * (prec * rec) / (beta_sq * prec + rec)

def evaluate_thresholds(y_true, y_probs, pair_meta, ground_truth, thresholds=None):
    """
    Evaluates entity-level Precision, Recall, and Macro-F0.5 across thresholds.
    Also produces confusion matrix at best threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.40, 0.95, 29)
        
    best_f05 = -1.0
    best_thresh = 0.50
    best_prec = 0.0
    best_rec = 0.0
    best_cm = None
    
    total_true_links = sum(len(m) for m in ground_truth.values())
    
    for tau in thresholds:
        s1_predictions = defaultdict(set)
        
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        
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
                    
        # Entity-level metrics
        pred_links = sum(len(m) for m in s1_predictions.values())
        correct_links = sum(len(s1_predictions[s1] & ground_truth[s1]) for s1 in ground_truth if s1 in s1_predictions)
        
        prec = correct_links / pred_links if pred_links > 0 else 1.0
        rec = correct_links / total_true_links if total_true_links > 0 else 0.0
        f05 = compute_f05_score(prec, rec)
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = tau
            best_prec = prec
            best_rec = rec
            best_cm = (tn, fp, fn, tp)
            
    return {
        'best_threshold': best_thresh,
        'precision': best_prec,
        'recall': best_rec,
        'f05': best_f05,
        'cm': best_cm
    }

print("Hybrid Experiment Module loaded successfully.")
