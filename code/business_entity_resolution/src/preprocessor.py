import re
import unicodedata
from .config import STOP_WORDS_NAME, STOP_WORDS_ADDR, INDIC_TRANSLIT

def remove_accents(text: str) -> str:
    """Strip accents and diacritics from unicode text."""
    if not text:
        return ""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def normalize_indic(text: str) -> str:
    """Map common Indic Devanagari legal/business terms to standard English equivalents."""
    if not text:
        return ""
    words = text.split()
    mapped = [INDIC_TRANSLIT.get(w, w) for w in words]
    return " ".join(mapped)

def clean_text(text: str) -> str:
    """Standardize business name or address text."""
    if not text:
        return ""
    text = normalize_indic(text)
    text = remove_accents(text)
    text = text.lower()
    text = re.sub(r'\[+.*?\]+|\(+.*?\)+|<+.*?>+', ' ', text)
    text = re.sub(r'\b(d\.?b\.?a\.?|doing business as)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 0]
    return " ".join(tokens)

def get_core_name_tokens(cleaned_name: str) -> list:
    """Extract significant corporate name tokens excluding legal stopwords."""
    tokens = cleaned_name.split()
    core = [t for t in tokens if t not in STOP_WORDS_NAME and len(t) > 1]
    if not core:
        core = [t for t in tokens if len(t) > 1]
    if not core:
        core = tokens
    return core

def get_address_tokens(cleaned_addr: str) -> list:
    """Extract significant address tokens excluding generic road/direction stopwords."""
    tokens = cleaned_addr.split()
    core = [t for t in tokens if t not in STOP_WORDS_ADDR and len(t) > 2]
    return core

def extract_numbers(text: str) -> set:
    """Extract street numbers, unit numbers, and PIN codes."""
    if not text:
        return set()
    return set(re.findall(r'\b\d+\b', text))

def get_char_ngrams(text: str, n: int = 3) -> set:
    """Generate character n-grams for fast typo-tolerant similarity."""
    if not text or len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text)-n+1)}

def preprocess_record(eid: str, name: str, addr: str, country: str) -> dict:
    """Create a structured preprocessed entity dictionary."""
    cn = clean_text(name)
    ca = clean_text(addr)
    return {
        'id': eid,
        'name': name or "",
        'addr': addr or "",
        'country': country or "",
        'name_clean': cn,
        'core_name': get_core_name_tokens(cn),
        'addr_clean': ca,
        'core_addr': get_address_tokens(ca),
        'numbers': extract_numbers(addr or ""),
        'char3': get_char_ngrams(cn, 3)
    }
