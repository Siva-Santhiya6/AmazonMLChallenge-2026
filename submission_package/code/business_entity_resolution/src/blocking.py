from collections import defaultdict
from rapidfuzz import fuzz

class MultiPassBlockingIndex:
    """High-performance in-memory inverted blocking index with multi-pass candidate generation."""
    
    def __init__(self):
        self.idx_exact_name = defaultdict(list)
        self.idx_name_token = defaultdict(list)
        self.idx_name_prefix = defaultdict(list)
        self.idx_addr_num_street = defaultdict(list)
        self.idx_addr_num = defaultdict(list)
        self.idx_addr_token = defaultdict(list)
        self.target_records = {}

    def add_target_records(self, records_dict: dict):
        """Index a dictionary of preprocessed target records."""
        for eid, rec in records_dict.items():
            self.target_records[eid] = rec
            country = rec['country']
            nc = rec['name_clean']
            core_n = rec['core_name']
            core_a = rec['core_addr']
            nums = rec['numbers']
            
            # 1. Exact Name Index
            if nc:
                self.idx_exact_name[(country, nc)].append(eid)
            
            # 2. Name Token Index & Prefix Index
            for t in core_n:
                if len(t) >= 3:
                    self.idx_name_token[(country, t)].append(eid)
                    if len(t) >= 4:
                        self.idx_name_prefix[(country, t[:4])].append(eid)
            
            # 3. Address Number + Street Index
            for num in nums:
                self.idx_addr_num[(country, num)].append(eid)
                for at in core_a:
                    if len(at) >= 3:
                        self.idx_addr_num_street[(country, num, at)].append(eid)
            
            # 4. Address Token Index
            for at in core_a:
                if len(at) >= 4:
                    self.idx_addr_token[(country, at)].append(eid)

    def query_candidates(self, s1_rec: dict, top_k: int = 10) -> list:
        """Query top candidate pairs for an S1 entity using weighted multi-pass scoring."""
        country = s1_rec['country']
        nc = s1_rec['name_clean']
        core_n = s1_rec['core_name']
        core_a = s1_rec['core_addr']
        nums = s1_rec['numbers']
        name_orig = s1_rec['name']
        addr_orig = s1_rec['addr']
        
        cand_scores = defaultdict(float)
        
        # Pass 1: Exact Name match
        if nc:
            for eid in self.idx_exact_name.get((country, nc), []):
                cand_scores[eid] += 60.0
        
        # Pass 2: Name Token match
        for t in core_n:
            if len(t) >= 3:
                hits = self.idx_name_token.get((country, t), [])
                if len(hits) <= 400:
                    w = 20.0 / (1.0 + 0.05 * len(hits))
                    for eid in hits:
                        cand_scores[eid] += w
        
        # Pass 3: Address Number + Street match
        for num in nums:
            for at in core_a:
                if len(at) >= 3:
                    hits = self.idx_addr_num_street.get((country, num, at), [])
                    if len(hits) <= 200:
                        w = 30.0 / (1.0 + 0.05 * len(hits))
                        for eid in hits:
                            cand_scores[eid] += w
        
        # Pass 4: Address Number match (for numbers with length >= 4)
        for num in nums:
            if len(num) >= 4:
                hits = self.idx_addr_num.get((country, num), [])
                if len(hits) <= 100:
                    for eid in hits:
                        cand_scores[eid] += 15.0
        
        # Pass 5: Prefix fallback if few candidates
        if len(cand_scores) < 4:
            for t in core_n:
                if len(t) >= 4:
                    hits = self.idx_name_prefix.get((country, t[:4]), [])
                    if len(hits) <= 150:
                        for eid in hits:
                            cand_scores[eid] += 8.0
        
        # Pass 6: Address Token fallback
        if len(cand_scores) < 4:
            for at in core_a:
                if len(at) >= 5:
                    hits = self.idx_addr_token.get((country, at), [])
                    if len(hits) <= 50:
                        for eid in hits:
                            cand_scores[eid] += 10.0
        
        if not cand_scores:
            return []
        
        # Fast composite re-ranking
        scored = []
        for eid, base_score in cand_scores.items():
            cand_rec = self.target_records[eid]
            nsim = fuzz.token_set_ratio(name_orig, cand_rec['name'])
            asim = fuzz.token_set_ratio(addr_orig, cand_rec['addr']) if addr_orig and cand_rec['addr'] else 40.0
            total_sim = base_score + (0.55 * nsim + 0.45 * asim)
            scored.append((total_sim, eid))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]
