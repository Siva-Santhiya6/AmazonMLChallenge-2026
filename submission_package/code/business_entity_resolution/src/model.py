import numpy as np
import lightgbm as lgb
import joblib
from .config import FEATURE_NAMES

class EntityResolutionClassifier:
    """LightGBM Binary Matcher with F0.5-optimized decision thresholding."""
    
    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, num_leaves: int = 31):
        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
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
