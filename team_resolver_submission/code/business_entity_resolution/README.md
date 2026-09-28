# Business Entity Resolution Pipeline

High-performance Machine Learning solution for large-scale multi-source Business Entity Resolution across noisy tabular datasets (US, India, France).

---

## 1. Overview & Architecture

The pipeline consists of two tightly integrated, scalable stages:

1. **Multi-Pass Inverted Blocking & Candidate Generation:**
   - Country-partitioned search space pruning ($O(N \cdot M) \rightarrow O(K)$).
   - Multi-index inverted table combining:
     - Exact normalized name hashes
     - Core name token index (excluding legal stop-words)
     - Character 3-gram prefix indices
     - Address street number & landmark anchor inverted index
     - Indic transliteration normalization (Devanagari $\rightarrow$ Roman)
     - Accented unicode normalization (French/Latin $\rightarrow$ NFKD ASCII)
   - Weighted composite re-ranking producing top-$K$ ($K \le 10$) high-quality candidate pairs per Source 1 entity.

2. **Gradient Boosted Pairwise Matching Model (LightGBM):**
   - 26-dimensional handcrafted feature vector capturing lexical, token set/sort, substring, char 3-gram Jaccard, digit overlap precision/recall, and blocking contextual metrics.
   - Decision threshold calibrated specifically for **Macro-$F_{0.5}$** (precision-heavy metric penalizing false merges $2\times$ more than false negatives).
   - Dynamic singleton filtering (emitting empty lists for singletons).

---

## 2. Requirements & Setup

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Supported Python versions: Python 3.8 to 3.14+.

---

## 3. Reproduction & Execution

To train the model and generate both `output/matching_results.tsv` and `output/candidate_pairs.tsv` end-to-end:

```bash
python -m src.pipeline --train-dir dataset/train --test-dir dataset/test --output-dir output --train
```

To run inference using an already trained model checkpoint (`model.joblib`):

```bash
python -m src.pipeline --test-dir dataset/test --output-dir output --model-path model.joblib
```

---

## 4. Submission Validation

Validate the output files locally using the challenge validator:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 5. Directory Structure

```
business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── config.py           # Hyperparameters, stop words, feature names
│   ├── preprocessor.py     # Text cleaning, unicode normalization, tokenization
│   ├── blocking.py         # Multi-pass inverted index and candidate generation
│   ├── features.py         # 26D pairwise feature extractor
│   ├── model.py            # LightGBM classifier with F0.5 thresholding
│   └── pipeline.py         # End-to-end training and streaming inference engine
├── requirements.txt        # Pinned dependencies
└── README.md               # Reproduction guide
```
