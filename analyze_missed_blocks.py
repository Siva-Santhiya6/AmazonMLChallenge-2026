import csv
import sys
import re
import unicodedata
from collections import defaultdict
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

print("="*70, flush=True)
print("ANALYZING MISSED MATCHES IN BLOCKING", flush=True)
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
    'इंडस्ट्रीज': 'industries', 'एलएलपी': 'llp'
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

# Load 2,000 S1 records
val_gt = {}
with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader, None)
    for row in reader:
        if len(row) >= 2:
            s1_id = row[0]
            matches = set(row[1].strip().split(",")) if row[1].strip() else set()
            val_gt[s1_id] = matches
        if len(val_gt) >= 2000:
            break

val_s1_ids = set(val_gt.keys())
val_s1_records = {}
with open("dataset/train/train_source1.tsv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader, None)
    for row in reader:
        if row[0] in val_s1_ids:
            val_s1_records[row[0]] = {
                'id': row[0], 'name': row[1], 'addr': row[2], 'country': row[3],
                'name_clean': clean_name(row[1]),
                'core_tokens': get_core_tokens(clean_name(row[1])),
                'addr_clean': clean_address(row[2]),
                'numbers': extract_numbers(row[2])
            }

all_true_target_ids = {m for matches in val_gt.values() for m in matches}
target_records = {}
def load_targets(filename):
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            eid = row[0]
            if eid in all_true_target_ids:
                target_records[eid] = {
                    'id': eid, 'name': row[1], 'addr': row[2], 'country': row[3],
                    'name_clean': clean_name(row[1]),
                    'core_tokens': get_core_tokens(clean_name(row[1])),
                    'addr_clean': clean_address(row[2]),
                    'numbers': extract_numbers(row[2])
                }

load_targets("dataset/train/train_source2.tsv")
load_targets("dataset/train/train_source3.tsv")

# Build indices
index_exact_name = defaultdict(list)
index_token = defaultdict(list)
index_num_token = defaultdict(list)
index_prefix = defaultdict(list)

for eid, rec in target_records.items():
    country = rec['country']
    n_clean = rec['name_clean']
    core = rec['core_tokens']
    numbers = rec['numbers']
    if n_clean:
        index_exact_name[(country, n_clean)].append(eid)
    for t in core:
        if len(t) >= 3:
            index_token[(country, t)].append(eid)
            if len(t) >= 4:
                index_prefix[(country, t[:4])].append(eid)
    for num in numbers:
        for t in core:
            if len(t) >= 3:
                index_num_token[(country, num, t[:3])].append(eid)

# Find missed pairs
missed_samples = []
for s1_id, s1_rec in val_s1_records.items():
    country = s1_rec['country']
    n_clean = s1_rec['name_clean']
    core = s1_rec['core_tokens']
    numbers = s1_rec['numbers']
    
    cand_scores = set()
    if n_clean:
        cand_scores.update(index_exact_name.get((country, n_clean), []))
    for num in numbers:
        for t in core:
            if len(t) >= 3:
                cand_scores.update(index_num_token.get((country, num, t[:3]), []))
    for t in core:
        if len(t) >= 3:
            cand_scores.update(index_token.get((country, t), []))
    for t in core:
        if len(t) >= 4:
            cand_scores.update(index_prefix.get((country, t[:4]), []))
    
    true_m = val_gt.get(s1_id, set())
    missed = true_m - cand_scores
    for m in missed:
        if m in target_records:
            missed_samples.append((s1_rec, target_records[m]))

print(f"Total Missed Matches out of {sum(len(m) for m in val_gt.values())}: {len(missed_samples)}", flush=True)
print("\nSample Missed Matches Analysis:", flush=True)
for i, (s1, tgt) in enumerate(missed_samples[:20], 1):
    print(f"\n--- Missed Match #{i} [{s1['country']}] ---", flush=True)
    print(f"  S1  : Name='{s1['name']}', Clean='{s1['name_clean']}', Core={s1['core_tokens']}, Addr='{s1['addr']}'", flush=True)
    print(f"  Tgt : ID={tgt['id']}, Name='{tgt['name']}', Clean='{tgt['name_clean']}', Core={tgt['core_tokens']}, Addr='{tgt['addr']}'", flush=True)
    print(f"  Name Fuzz Token Set: {fuzz.token_set_ratio(s1['name'], tgt['name']):.1f} | Addr Fuzz: {fuzz.token_set_ratio(s1['addr'], tgt['addr']):.1f}", flush=True)

