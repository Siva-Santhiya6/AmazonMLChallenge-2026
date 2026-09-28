import time
import os
import gc
from collections import defaultdict
import string
import unicodedata
from rapidfuzz import fuzz
import joblib
import numpy as np

PUNCT_TRANS = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

def remove_accents(text):
    if not text:
        return ""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def clean_fast(text):
    if not text:
        return ""
    text = remove_accents(text)
    text = text.lower().translate(PUNCT_TRANS)
    return " ".join(text.split())

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
    'court', 'ct', 'boulevard', 'bd', 'way', 'circle', 'cir', 'place', 'pl',
    'highway', 'hwy', 'parkway', 'pkwy', 'suite', 'ste', 'unit', 'apt', 'apartment',
    'floor', 'fl', 'block', 'blk', 'building', 'bldg', 'near', 'behind', 'opp',
    'opposite', 'plot', 'shop', 'no', 'number', 'door', 'h', 'house', 'sec', 'sector',
    'phase', 'east', 'west', 'north', 'south', 'e', 'w', 'n', 's', 'us', 'usa',
    'india', 'france', 'de', 'du', 'la', 'le', 'rue', 'r', 'chem', 'chemin', 'route', 'rte',
    'cedex', 'delhi', 'new', 'hq', 'region'
}

def extract_record(eid, name, addr):
    cn = clean_fast(name)
    ca = clean_fast(addr)
    n_toks = cn.split()
    core_n = tuple(t for t in n_toks if t not in STOP_WORDS_NAME and len(t) > 1) or tuple(t for t in n_toks if len(t) > 1) or tuple(n_toks)
    a_toks = ca.split()
    core_a = tuple(t for t in a_toks if t not in STOP_WORDS_ADDR and len(t) > 2)
    nums = tuple(t for t in a_toks if t.isdigit())
    return (name, addr, cn, core_n, ca, core_a, nums)

print("Testing France Target Loading...", flush=True)
t0 = time.time()
targets = {}
for fname in ["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"]:
    with open(fname, "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[3] == "France":
                targets[parts[0]] = extract_record(parts[0], parts[1], parts[2])

print(f"Loaded {len(targets):,} France targets in {time.time()-t0:.2f}s", flush=True)

# Build index
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

print(f"Inverted indices built in {time.time()-t_idx:.2f}s", flush=True)

# Test 1000 S1 queries
s1_queries = []
with open("dataset/test/test_source1.tsv", "r", encoding="utf-8", errors="replace") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4 and parts[3] == "France":
            s1_queries.append((parts[0], extract_record(parts[0], parts[1], parts[2])))
            if len(s1_queries) >= 1000:
                break

t_q = time.time()
total_cands = 0
for s1_id, s1_rec in s1_queries:
    name1, addr1, nc1, core_n, ca1, core_a, nums = s1_rec
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
    for num in nums:
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
    
    if cand_scores:
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
        top_cands = scored[:10]
        total_cands += len(top_cands)

q_time = time.time() - t_q
print(f"Processed 1,000 France queries in {q_time:.2f}s ({1000/q_time:.0f} queries/sec). Total cands: {total_cands:,}", flush=True)
