import csv
import sys
import re
import unicodedata
from collections import defaultdict, Counter
import time
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

print("="*70, flush=True)
print("TESTING TEXT NORMALIZATION & BLOCKING STRATEGIES", flush=True)
print("="*70, flush=True)

# Common legal suffixes across US, India, France
STOP_WORDS_NAME = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'services', 'service', 'solutions',
    'technologies', 'technology', 'enterprises', 'enterprise', 'group',
    'holdings', 'holding', 'industries', 'industry', 'sa', 'sarl', 'sas',
    'gmbh', 'dba', 'the', 'and', 'llp', 'pllc', 'center', 'centre',
    'associates', 'partners', 'international', 'consulting', 'consultants',
    'management', 'global', 'systems', 'system', 'india', 'usa', 'france',
    'societe', 'et', 'cie', 'de', 'du', 'la', 'le', 'les', 'des'
}

# Hindi to Roman basic phonetic mappings for common corporate words
INDIC_TRANSLIT = {
    'प्राइवेट': 'private',
    'लिमिटेड': 'limited',
    'एंटरप्राइजेज': 'enterprises',
    'इंटरप्राइजेज': 'enterprises',
    'सर्विसेज': 'services',
    'सॉल्यूशंस': 'solutions',
    'टेक्नोलॉजीज': 'technologies',
    'कंपनी': 'company',
    'प्रॉपर्टीज': 'properties',
    'एसोसिएट्स': 'associates',
    'उद्योग': 'udyog',
    'इन्फ्राटेक': 'infratech',
    'मार्केटिंग': 'marketing',
    'फाइनेंस': 'finance',
    'ग्रुप': 'group',
    'इंटरनेशनल': 'international',
    'कंसल्टेंट्स': 'consultants',
    'कार्पोरेशन': 'corporation',
    'कारपोरेशन': 'corporation',
    'वेंचर्स': 'ventures',
    'होटल': 'hotel',
    'इंडस्ट्रीज': 'industries',
    'एलएलपी': 'llp',
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
    # Map Indic words first
    name = normalize_indic(name)
    # Remove accents
    name = remove_accents(name)
    # Lowercase
    name = name.lower()
    # Remove special patterns like <<, >>, [[, ]], d.b.a., etc.
    name = re.sub(r'\[+.*?\]+|\(+.*?\)+|<+.*?>+', ' ', name)
    name = re.sub(r'\b(d\.?b\.?a\.?|doing business as)\b', ' ', name)
    # Replace non-alphanumeric with space
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    # Tokenize and filter out single letters or empty tokens
    tokens = [t for t in name.split() if len(t) > 0]
    return " ".join(tokens)

def get_core_name_tokens(cleaned_name):
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

# Test cleaning functions on sample noisy data
test_samples = [
    "PAYNE-ENRTPRMISES",
    "Korbrixx D.B.A. Obsidian, LLC",
    "Payne Énterprises",
    "रेड वेंचर्स प्राइवेट लिमिटेड",
    "Payne Enterprises  LLC",
    "SCI Ptit Àmicale",
    "<< Team Ecole",
    "Chordia + Pagnters - 7306204978"
]

print("Testing Name Normalizer on Samples:")
for s in test_samples:
    cn = clean_name(s)
    core = get_core_name_tokens(cn)
    print(f"  Orig: {s:35} -> Cleaned: {cn:30} -> Core: {core}")

