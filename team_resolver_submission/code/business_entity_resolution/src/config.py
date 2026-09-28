# Configuration and constants for Business Entity Resolution Pipeline

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

FEATURE_NAMES = [
    'n_ratio', 'n_part_ratio', 'n_tok_sort', 'n_tok_set',
    'nc_tok_sort', 'nc_tok_set', 'exact_name', 'tok_jaccard', 'char3_jaccard',
    'len_diff', 'len_ratio',
    'a_tok_sort', 'a_tok_set', 'ac_tok_sort', 'ac_tok_set', 'addr_empty',
    'num_cnt_s1', 'num_cnt_tgt', 'num_overlap', 'num_prec', 'num_rec',
    'comb_tok_set', 'is_s2', 'block_rank', 'block_score', 'score_diff',
    'exact_addr', 'prefix_match', 'num_exact', 'tok_diff'
]

DEFAULT_TOP_K = 10
OPTIMAL_MATCH_THRESHOLD = 0.70
