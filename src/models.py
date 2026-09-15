## Temel modeller, ensemble yapısı ve sınıf dengesizliği yönetimi.

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from imblearn.combine import SMOTETomek
import joblib

def get_baseline_models():
    """Temel kıyaslama modellerini döner."""
    return {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, eval_metric="logloss"),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", probability=True),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced")
    }

def build_ensemble_model(scale_pos_weight: float = 1.0) -> VotingClassifier:
    """Optimize edilmiş Random Forest ve XGBoost tabanlı Soft Voting modeli kurar."""
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=14,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.08,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )
    return VotingClassifier(estimators=[("rf", rf), ("xgb", xgb_model)], voting="soft")

def balance_dataset(X, y):
    """SMOTE-Tomek uygulayarak sınıfları dengeler."""
    resampler = SMOTETomek(random_state=42)
    return resampler.fit_resample(X, y)

def save_model(model, filepath: str):
    joblib.dump(model, filepath)

def load_model(filepath: str):
    return joblib.load(filepath)