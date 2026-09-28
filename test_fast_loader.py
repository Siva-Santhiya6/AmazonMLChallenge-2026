import sys
import os
import time
import unicodedata
import string
from collections import defaultdict
import numpy as np
import lightgbm as lgb
import joblib

sys.stdout.reconfigure(encoding='utf-8')

print("="*70, flush=True)
print("TESTING ULTRA-FAST MULTIPROCESSING INFERENCE ENGINE", flush=True)
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
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def clean_fast(text):
    if not text:
        return ""
    text = remove_accents(text)
    text = text.lower().translate(PUNCT_TRANS)
    return " ".join(text.split())

def extract_record_fast(eid, name, addr, country):
    cn = clean_fast(name)
    ca = clean_fast(addr)
    
    n_toks = cn.split()
    core_n = [t for t in n_toks if t not in STOP_WORDS_NAME and len(t) > 1]
    if not core_n:
        core_n = [t for t in n_toks if len(t) > 1] or n_toks
        
    a_toks = ca.split()
    core_a = [t for t in a_toks if t not in STOP_WORDS_ADDR and len(t) > 2]
    nums = {t for t in a_toks if t.isdigit()}
    
    char3 = {cn[i:i+3] for i in range(len(cn)-2)} if len(cn) >= 3 else set()
    
    return (eid, name, addr, country, cn, tuple(core_n), ca, tuple(core_a), nums, char3)

# Benchmark loading France from test files
t0 = time.time()
print("Loading France records from test set...", flush=True)

s1_france = []
with open("dataset/test/test_source1.tsv", "r", encoding="utf-8", errors="replace") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4 and parts[3] == "France":
            s1_france.append(extract_record_fast(parts[0], parts[1], parts[2], parts[3]))

print(f"Loaded {len(s1_france):,} France S1 queries in {time.time()-t0:.2f}s", flush=True)

t1 = time.time()
targets_france = {}
for name in ["test_source2.tsv", "test_source3.tsv"]:
    with open(f"dataset/test/{name}", "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[3] == "France":
                targets_france[parts[0]] = extract_record_fast(parts[0], parts[1], parts[2], parts[3])

print(f"Loaded {len(targets_france):,} France Target records in {time.time()-t1:.2f}s", flush=True)

# Build Inverted Index for France
t2 = time.time()
idx_exact = defaultdict(list)
idx_token = defaultdict(list)
idx_num_street = defaultdict(list)
idx_num = defaultdict(list)
idx_prefix = defaultdict(list)

for rec in targets_france.values():
    eid, name, addr, country, nc, core_n, ca, core_a, nums, char3 = rec
    if nc:
        idx_exact[nc].append(eid)
    for t in core_n:
        if len(t) >= 3:
            idx_token[t].append(eid)
            if len(t) >= 4:
                idx_prefix[t[:4]].append(eid)
    for num in nums:
        idx_num[num].append(eid)
        for at in core_a:
            if len(at) >= 3:
                idx_num_street[(num, at)].append(eid)

print(f"Inverted Index built for France in {time.time()-t2:.2f}s", flush=True)

