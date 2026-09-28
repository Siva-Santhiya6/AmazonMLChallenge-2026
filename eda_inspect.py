import os
import sys
import importlib.util
import subprocess

print("Python version:", sys.version, flush=True)
print("CPU count:", os.cpu_count(), flush=True)

pkgs = [
    'pandas', 'numpy', 'scipy', 'sklearn', 'lightgbm', 'xgboost', 
    'catboost', 'torch', 'transformers', 'rapidfuzz', 'polars', 
    'pyarrow', 'tqdm', 'joblib', 'nltk', 'Levenshtein', 'numba', 'scikit-learn'
]

print("\n--- Package Check ---", flush=True)
for p in pkgs:
    try:
        mod = __import__(p)
        ver = getattr(mod, '__version__', 'installed')
        print(f"  [YES] {p}: {ver}", flush=True)
    except Exception as e:
        print(f"  [NO]  {p}: {e}", flush=True)

print("\n--- Pip List ---", flush=True)
subprocess.run([sys.executable, "-m", "pip", "list"])
