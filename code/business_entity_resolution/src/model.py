import numpy as np
import lightgbm as lgb
import joblib
from .config import FEATURE_NAMES

class EntityResolutionClassifier:
    """LightGBM Binary Matcher with F0.5-optimized decision thresholding."""
    
    def __init__(self, n_estimators: int = 450, learning_rate: float = 0.05, num_leaves: int = 63,
                 max_depth: int = 8, min_child_samples: int = 80, reg_alpha: float = 0.5,
                 reg_lambda: float = 5.0, min_split_gain: float = 0.05, colsample_bytree: float = 0.80,
                 subsample: float = 0.85, subsample_freq: int = 1):
        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_split_gain=min_split_gain,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            subsample_freq=subsample_freq,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.threshold = 0.70

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the LightGBM model on extracted pairwise feature vectors."""
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive match probability."""
        if len(X) == 0:
            return np.array([])
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath: str):
        """Serialize model to disk."""
        joblib.dump({'model': self.model, 'threshold': self.threshold}, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Load trained model from disk."""
        instance = cls()
        data = joblib.load(filepath)
        instance.model = data['model']
        instance.threshold = data.get('threshold', 0.70)
        return instance
