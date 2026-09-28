from rapidfuzz import fuzz
from .config import FEATURE_NAMES

def extract_pairwise_features(s1_rec: dict, cand_rec: dict, block_score: float, block_rank: int, top1_score: float) -> list:
    """Extract a comprehensive 26-dimensional similarity feature vector for a pair."""
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
    
    # Length statistics
    l1, l2 = len(name1), len(name2)
    len_diff = abs(l1 - l2)
    len_ratio = min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 1.0
    
    # 2. Address Similarities
    a_tok_sort = fuzz.token_sort_ratio(addr1, addr2) if addr1 and addr2 else 0.0
    a_tok_set = fuzz.token_set_ratio(addr1, addr2) if addr1 and addr2 else 0.0
    ac_tok_sort = fuzz.token_sort_ratio(ca1, ca2) if ca1 and ca2 else 0.0
    ac_tok_set = fuzz.token_set_ratio(ca1, ca2) if ca1 and ca2 else 0.0
    addr_empty = 1.0 if not addr2.strip() else 0.0
    
    # Numeric Overlap (House numbers, suite numbers, PIN codes)
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
