import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from app.config import settings

def map_icd9(val: Any) -> str:
    """Map ICD-9 diagnosis codes into standard clinical disease categories."""
    if val is None or pd.isna(val) or str(val).strip() in ("?", ""):
        return "Other"
    val_str = str(val).strip()
    if val_str.startswith("V") or val_str.startswith("E"):
        return "Other"
    try:
        code = float(val_str)
        if (390 <= code <= 459) or code == 785:
            return "Circulatory"
        elif (460 <= code <= 519) or code == 786:
            return "Respiratory"
        elif (520 <= code <= 579) or code == 787:
            return "Digestive"
        elif 250 <= code < 251:
            return "Diabetes"
        elif 800 <= code <= 999:
            return "Injury"
        elif 710 <= code <= 739:
            return "Musculoskeletal"
        elif (580 <= code <= 629) or code == 788:
            return "Genitourinary"
        elif 140 <= code <= 239:
            return "Neoplasms"
        else:
            return "Other"
    except ValueError:
        v_low = val_str.lower()
        if any(w in v_low for w in ["circulat", "heart", "cardiac", "coronary", "cad", "mi"]):
            return "Circulatory"
        elif any(w in v_low for w in ["respirat", "lung", "copd", "pneumonia", "asthma"]):
            return "Respiratory"
        elif any(w in v_low for w in ["digest", "gi", "gastro"]):
            return "Digestive"
        elif "diabet" in v_low:
            return "Diabetes"
        elif any(w in v_low for w in ["renal", "kidney", "nephro", "urinary"]):
            return "Genitourinary"
        elif any(w in v_low for w in ["cancer", "neoplasm", "tumor"]):
            return "Neoplasms"
        return "Other"


MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

BASE_FEATURE_COLS = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed", "diag_1_group",
] + MEDICATION_COLS

ENGINEERED_FEATURE_COLS = [
    "prior_utilization", "lab_intensity", "med_intensity", "comorbidity_score",
    "is_emergency", "discharged_home", "has_insulin", "active_med_count",
]

FEATURE_COLS = BASE_FEATURE_COLS + ENGINEERED_FEATURE_COLS


class DataPreprocessor:
    @staticmethod
    def load_dataset(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.replace("?", np.nan)
        df = df.dropna(subset=["readmitted"])
        # Standard Kaggle clinical benchmark: exclude deceased/hospice patients
        if "discharge_disposition_id" in df.columns:
            df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])].copy()
        df["target"] = (df["readmitted"] == "<30").astype(int)
        df = DataPreprocessor.add_engineered_features(df)
        return df

    @staticmethod
    def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
        if "diag_1" in df.columns:
            df["diag_1_group"] = df["diag_1"].apply(map_icd9)
        elif "diag_1_group" not in df.columns:
            df["diag_1_group"] = "Other"

        outpatient = df["number_outpatient"].fillna(0).astype(float) if "number_outpatient" in df.columns else 0
        emergency = df["number_emergency"].fillna(0).astype(float) if "number_emergency" in df.columns else 0
        inpatient = df["number_inpatient"].fillna(0).astype(float) if "number_inpatient" in df.columns else 0
        stay = df["time_in_hospital"].fillna(1).astype(float).replace(0, 1) if "time_in_hospital" in df.columns else 1
        labs = df["num_lab_procedures"].fillna(0).astype(float) if "num_lab_procedures" in df.columns else 0
        meds = df["num_medications"].fillna(0).astype(float) if "num_medications" in df.columns else 0
        diagnoses = df["number_diagnoses"].fillna(0).astype(float) if "number_diagnoses" in df.columns else 0

        df["prior_utilization"] = outpatient + emergency + inpatient
        df["lab_intensity"] = labs / stay
        df["med_intensity"] = meds / stay
        df["comorbidity_score"] = diagnoses

        if "admission_type_id" in df.columns:
            df["is_emergency"] = df["admission_type_id"].isin([1, 2]).astype(int)
        else:
            df["is_emergency"] = 0

        if "discharge_disposition_id" in df.columns:
            df["discharged_home"] = (df["discharge_disposition_id"] == 1).astype(int)
        else:
            df["discharged_home"] = 1

        if "insulin" in df.columns:
            df["has_insulin"] = (df["insulin"] != "No").astype(int)
        else:
            df["has_insulin"] = 0

        med_present = [c for c in MEDICATION_COLS if c in df.columns]
        if med_present:
            df["active_med_count"] = (df[med_present] != "No").sum(axis=1)
        else:
            df["active_med_count"] = 0

        return df

    @staticmethod
    def patient_to_features(patient_data: Dict[str, Any]) -> pd.DataFrame:
        row = {}
        for col in BASE_FEATURE_COLS:
            val = patient_data.get(col)
            if val is None or val == "?" or (isinstance(val, float) and pd.isna(val)):
                mapped = {
                    "A1Cresult": "a1cresult",
                    "diabetesMed": "diabetes_med",
                }.get(col)
                if mapped:
                    val = patient_data.get(mapped)

            if val is None or val == "?" or (isinstance(val, float) and pd.isna(val)):
                if col in MEDICATION_COLS:
                    val = "No"
                elif col == "diag_1_group":
                    diag_raw = patient_data.get("diag_1") or patient_data.get("diagnosis")
                    val = map_icd9(diag_raw)
                elif col in ["race", "gender", "age", "max_glu_serum", "A1Cresult", "change", "diabetesMed"]:
                    val = "Unknown"
                else:
                    val = 0
            elif col == "diag_1_group":
                val = map_icd9(val)

            row[col] = str(val) if col in (["race", "gender", "age", "max_glu_serum", "A1Cresult", "change", "diabetesMed", "diag_1_group"] + MEDICATION_COLS) else val

        outpatient = float(row.get("number_outpatient") or 0)
        emergency = float(row.get("number_emergency") or 0)
        inpatient = float(row.get("number_inpatient") or 0)
        stay = max(float(row.get("time_in_hospital") or 1), 1.0)
        labs = float(row.get("num_lab_procedures") or 0)
        meds = float(row.get("num_medications") or 0)
        diagnoses = float(row.get("number_diagnoses") or 0)

        adm_type = row.get("admission_type_id")
        try:
            is_emerg = 1 if int(adm_type) in (1, 2) else 0
        except (TypeError, ValueError):
            is_emerg = 0

        disch_disp = row.get("discharge_disposition_id")
        try:
            disch_home = 1 if int(disch_disp) == 1 else 0
        except (TypeError, ValueError):
            disch_home = 1

        insulin_val = str(row.get("insulin", "No")).lower()
        has_ins = 1 if insulin_val not in ("no", "0", "false", "none") else 0

        active_meds = sum(
            1 for m in MEDICATION_COLS if str(row.get(m, "No")).lower() not in ("no", "0", "false", "none", "unknown")
        )

        row["prior_utilization"] = outpatient + emergency + inpatient
        row["lab_intensity"] = labs / stay
        row["med_intensity"] = meds / stay
        row["comorbidity_score"] = diagnoses
        row["is_emergency"] = is_emerg
        row["discharged_home"] = disch_home
        row["has_insulin"] = has_ins
        row["active_med_count"] = active_meds

        return pd.DataFrame([row])


class ModelTrainer:
    def __init__(self, model_dir: str = settings.ml_model_dir):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, Any] = {}

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if "diag_1_group" not in df.columns or "prior_utilization" not in df.columns:
            df = DataPreprocessor.add_engineered_features(df)
        available = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available].copy()
        y = df["target"]
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].fillna("Unknown").astype(str)
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(0).astype(float)
        return X, y

    def _build_pipeline(
        self,
        model_type: str,
        X: pd.DataFrame,
        scale_pos_weight: float = 1.0,
        class_weight_strategy: str = "balanced",
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Pipeline:
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        transformers = []
        if num_cols:
            transformers.append(("num", StandardScaler(), num_cols))
        if cat_cols:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            )

        preprocessor = ColumnTransformer(transformers=transformers)
        params = hyperparameters or {}

        if model_type == "xgboost":
            weight = scale_pos_weight if class_weight_strategy == "balanced" else 1.0
            n_estimators = int(params.get("n_estimators", 160))
            max_depth = int(params.get("max_depth", 4))
            learning_rate = float(params.get("learning_rate", 0.06))

            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=weight,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
            )
        else:
            cw = "balanced_subsample" if class_weight_strategy == "balanced" else None
            n_estimators = int(params.get("n_estimators", 200))
            max_depth = int(params.get("max_depth", 13))
            min_samples_leaf = int(params.get("min_samples_leaf", 4))

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=cw,
                random_state=42,
                n_jobs=-1,
            )

        return Pipeline([("preprocessor", preprocessor), ("classifier", model)])

    def train(
        self,
        df: pd.DataFrame,
        model_type: str = "random_forest",
        version: Optional[str] = None,
        class_weight_strategy: str = "balanced",
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        X, y = self._prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pos_count = int(y_train.sum())
        neg_count = int(len(y_train) - pos_count)
        scale_pos = neg_count / max(pos_count, 1)

        pipeline = self._build_pipeline(
            model_type,
            X_train,
            scale_pos_weight=scale_pos,
            class_weight_strategy=class_weight_strategy,
            hyperparameters=hyperparameters,
        )
        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_prob))

        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = int(np.argmax(f1_scores))
        optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.48
        # Constrain threshold to prevent extreme recall over-prediction and maintain high precision
        optimal_threshold = float(np.clip(optimal_threshold, 0.44, 0.54))

        y_pred_calibrated = (y_prob >= optimal_threshold).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred_calibrated)),
            "precision": float(precision_score(y_test, y_pred_calibrated, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_calibrated, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred_calibrated, zero_division=0)),
            "roc_auc": roc_auc,
            "optimal_threshold": round(optimal_threshold, 4),
            "model_name": model_type,
            "version": version or "v1.0.0",
            "sample_size": len(df),
            "hyperparameters": hyperparameters or {},
            "class_weight_strategy": class_weight_strategy,
            "trained_at": pd.Timestamp.now().isoformat(),
        }

        active_model_path = self.model_dir / f"{model_type}_model.joblib"
        joblib.dump(pipeline, active_model_path)

        if version:
            versioned_path = self.model_dir / f"{model_type}_{version}.joblib"
            joblib.dump(pipeline, versioned_path)
            metrics["model_path"] = str(versioned_path)
        else:
            metrics["model_path"] = str(active_model_path)

        metrics_path = self.model_dir / f"{model_type}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        self.metrics[model_type] = metrics
        return metrics


class PredictionEngine:
    def __init__(self, model_dir: str = settings.ml_model_dir):
        self.model_dir = Path(model_dir)
        self._models: Dict[str, Pipeline] = {}
        self._thresholds: Dict[str, float] = {}

    def load_model(self, model_type: str = "random_forest", version: Optional[str] = None) -> Pipeline:
        cache_key = f"{model_type}_{version}" if version else model_type
        if cache_key not in self._models:
            if version:
                path = self.model_dir / f"{model_type}_{version}.joblib"
                if not path.exists():
                    path = self.model_dir / f"{model_type}_model.joblib"
            else:
                path = self.model_dir / f"{model_type}_model.joblib"

            if not path.exists():
                raise FileNotFoundError(f"Model not found: {path}. Run training first.")
            self._models[cache_key] = joblib.load(path)
            
            metrics = self.load_metrics(model_type, model_dir=str(self.model_dir))
            threshold = metrics.get("optimal_threshold", 0.48) if metrics else 0.48
            self._thresholds[cache_key] = threshold

        return self._models[cache_key]

    def predict(
        self,
        patient_data: Dict[str, Any],
        model_type: str = "random_forest",
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        t0 = time.time()
        model = self.load_model(model_type, version)
        features = DataPreprocessor.patient_to_features(patient_data)
        prob = float(model.predict_proba(features)[0][1])
        risk_score = round(prob * 100, 2)

        cache_key = f"{model_type}_{version}" if version else model_type
        threshold = self._thresholds.get(cache_key, 0.48)

        # Harmonized, calibrated risk categorization:
        # High Risk: probability clearly exceeds optimal decision threshold (>= threshold * 1.12 or prob >= 0.52)
        # Medium Risk: probability indicates elevated risk near decision boundary (>= threshold * 0.75 or prob >= 0.32)
        # Low Risk: probability is safely below decision threshold (< threshold * 0.75)
        if prob >= max(0.52, threshold * 1.12) or risk_score >= 55:
            category = "High"
        elif prob >= max(0.32, threshold * 0.75) or risk_score >= 32:
            category = "Medium"
        else:
            category = "Low"

        importance = self._get_feature_importance(model, features)
        latency_ms = round((time.time() - t0) * 1000, 2)

        return {
            "risk_score": risk_score,
            "risk_category": category,
            "readmission_probability": round(prob, 4),
            "model_used": model_type,
            "model_version": version or "v1.0.0",
            "optimal_threshold": threshold,
            "feature_importance": importance,
            "latency_ms": latency_ms,
        }

    def _get_feature_importance(self, model: Pipeline, features: pd.DataFrame) -> Dict[str, float]:
        try:
            classifier = model.named_steps["classifier"]
            if hasattr(classifier, "feature_importances_"):
                preprocessor = model.named_steps["preprocessor"]
                names = preprocessor.get_feature_names_out()
                importances = classifier.feature_importances_
                top_indices = np.argsort(importances)[-10:][::-1]
                return {str(names[i]): float(importances[i]) for i in top_indices}
        except Exception:
            pass
        return {}

    @staticmethod
    def get_risk_category(score: float) -> str:
        if score >= 55:
            return "High"
        if score >= 32:
            return "Medium"
        return "Low"

    @staticmethod
    def load_metrics(model_type: str = "random_forest", model_dir: Optional[str] = None) -> Optional[Dict]:
        m_dir = Path(model_dir) if model_dir else Path(settings.ml_model_dir)
        path = m_dir / f"{model_type}_metrics.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
