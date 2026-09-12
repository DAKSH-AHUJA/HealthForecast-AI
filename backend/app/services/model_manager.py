import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.config import settings
from app.ml.predictor import DataPreprocessor, ModelTrainer, PredictionEngine
from app.models.prediction import ModelInferenceLog, ModelVersionRecord
from app.models.user import User


class ModelManagerService:
    def __init__(self, model_dir: str = settings.ml_model_dir):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.engine = PredictionEngine(model_dir=str(self.model_dir))

    def get_version_history(self, db: Session, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        query = db.query(ModelVersionRecord)
        if model_name:
            query = query.filter(ModelVersionRecord.model_name == model_name)
        records = query.order_by(ModelVersionRecord.trained_at.desc()).all()

        results = []
        for r in records:
            results.append({
                "id": r.id,
                "model_name": r.model_name,
                "version": r.version,
                "is_active": r.is_active,
                "accuracy": round(r.accuracy, 4),
                "precision": round(r.precision, 4),
                "recall": round(r.recall, 4),
                "f1_score": round(r.f1_score, 4),
                "roc_auc": round(r.roc_auc, 4),
                "optimal_threshold": round(r.optimal_threshold or 0.45, 4),
                "sample_size": r.training_sample_size,
                "hyperparameters": json.loads(r.hyperparameters) if r.hyperparameters else {},
                "notes": r.notes,
                "trained_by": r.trained_by,
                "trained_at": r.trained_at.isoformat() if r.trained_at else None,
            })
        return results

    def controlled_retrain(
        self,
        db: Session,
        model_type: str,
        sample_size: int = 20000,
        class_weight_strategy: str = "balanced",
        hyperparameters: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        trained_by: str = "system_admin",
    ) -> Dict[str, Any]:
        df = DataPreprocessor.load_dataset(settings.dataset_path)
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        # Generate new version identifier
        existing_count = db.query(ModelVersionRecord).filter(
            ModelVersionRecord.model_name == model_type
        ).count()
        new_version = f"v{existing_count + 1}.0.0"

        trainer = ModelTrainer(model_dir=str(self.model_dir))
        metrics = trainer.train(
            df=df,
            model_type=model_type,
            version=new_version,
            class_weight_strategy=class_weight_strategy,
            hyperparameters=hyperparameters,
        )

        # Deactivate previous active models of this type
        db.query(ModelVersionRecord).filter(
            ModelVersionRecord.model_name == model_type,
            ModelVersionRecord.is_active == True,
        ).update({"is_active": False})

        version_record = ModelVersionRecord(
            model_name=model_type,
            version=new_version,
            is_active=True,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            roc_auc=metrics["roc_auc"],
            optimal_threshold=metrics["optimal_threshold"],
            training_sample_size=len(df),
            hyperparameters=json.dumps(metrics.get("hyperparameters", {})),
            model_path=metrics.get("model_path"),
            notes=notes or f"Controlled retraining with {class_weight_strategy} weighting",
            trained_by=trained_by,
            trained_at=datetime.utcnow(),
        )
        db.add(version_record)
        db.commit()
        db.refresh(version_record)

        return metrics

    def activate_version(self, db: Session, version_id: int) -> Dict[str, Any]:
        record = db.query(ModelVersionRecord).filter(ModelVersionRecord.id == version_id).first()
        if not record:
            raise ValueError("Model version not found")

        # Copy versioned model to active model file
        versioned_path = self.model_dir / f"{record.model_name}_{record.version}.joblib"
        active_path = self.model_dir / f"{record.model_name}_model.joblib"
        if versioned_path.exists():
            shutil.copy2(versioned_path, active_path)

        # Update metrics JSON
        metrics = {
            "model_name": record.model_name,
            "version": record.version,
            "accuracy": record.accuracy,
            "precision": record.precision,
            "recall": record.recall,
            "f1_score": record.f1_score,
            "roc_auc": record.roc_auc,
            "optimal_threshold": record.optimal_threshold,
            "trained_at": record.trained_at.isoformat() if record.trained_at else None,
            "sample_size": record.training_sample_size,
        }
        with open(self.model_dir / f"{record.model_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Update active statuses in DB
        db.query(ModelVersionRecord).filter(
            ModelVersionRecord.model_name == record.model_name
        ).update({"is_active": False})
        record.is_active = True
        db.commit()

        # Invalidate memory cache in PredictionEngine
        self.engine._models.clear()
        self.engine._thresholds.clear()

        return {"message": f"Successfully activated {record.model_name} {record.version}", "version": record.version}

    def rollback_model(self, db: Session, model_type: str) -> Dict[str, Any]:
        current_active = db.query(ModelVersionRecord).filter(
            ModelVersionRecord.model_name == model_type,
            ModelVersionRecord.is_active == True,
        ).first()

        # Find previous version
        query = db.query(ModelVersionRecord).filter(
            ModelVersionRecord.model_name == model_type
        )
        if current_active:
            query = query.filter(ModelVersionRecord.id < current_active.id)
        prev = query.order_by(ModelVersionRecord.id.desc()).first()

        if not prev:
            raise ValueError(f"No previous version available for rollback of {model_type}")

        return self.activate_version(db, prev.id)

    def log_inference(
        self,
        db: Session,
        model_name: str,
        risk_score: float,
        risk_category: str,
        probability: float,
        latency_ms: float = 0.0,
        model_version: Optional[str] = None,
    ):
        log = ModelInferenceLog(
            model_name=model_name,
            model_version=model_version or "active",
            risk_score=risk_score,
            risk_category=risk_category,
            readmission_probability=probability,
            latency_ms=latency_ms,
            created_at=datetime.utcnow(),
        )
        db.add(log)
        db.commit()

    def get_monitoring_dashboard(self, db: Session) -> Dict[str, Any]:
        logs = db.query(ModelInferenceLog).all()
        total_inferences = len(logs)

        high_count = sum(1 for l in logs if l.risk_category == "High")
        med_count = sum(1 for l in logs if l.risk_category == "Medium")
        low_count = sum(1 for l in logs if l.risk_category == "Low")

        avg_prob = sum(l.readmission_probability for l in logs) / total_inferences if total_inferences else 0.0
        avg_latency = sum(l.latency_ms for l in logs) / total_inferences if total_inferences else 0.0

        active_versions = {}
        for m in ["random_forest", "xgboost"]:
            active = db.query(ModelVersionRecord).filter(
                ModelVersionRecord.model_name == m,
                ModelVersionRecord.is_active == True,
            ).first()
            if active:
                active_versions[m] = {
                    "version": active.version,
                    "accuracy": round(active.accuracy, 4),
                    "precision": round(active.precision, 4),
                    "recall": round(active.recall, 4),
                    "f1_score": round(active.f1_score, 4),
                    "roc_auc": round(active.roc_auc, 4),
                    "optimal_threshold": round(active.optimal_threshold or 0.45, 4),
                }
            else:
                metrics = PredictionEngine.load_metrics(m)
                if metrics:
                    active_versions[m] = {
                        "version": metrics.get("version", "v1.0.0"),
                        "accuracy": round(metrics.get("accuracy", 0), 4),
                        "precision": round(metrics.get("precision", 0), 4),
                        "recall": round(metrics.get("recall", 0), 4),
                        "f1_score": round(metrics.get("f1_score", 0), 4),
                        "roc_auc": round(metrics.get("roc_auc", 0), 4),
                        "optimal_threshold": round(metrics.get("optimal_threshold", 0.45), 4),
                    }

        # Distribution drift assessment
        expected_pos_rate = 0.11
        actual_high_rate = (high_count / total_inferences) if total_inferences else 0.0
        drift_detected = abs(actual_high_rate - expected_pos_rate) > 0.15 and total_inferences > 20

        return {
            "total_inferences": total_inferences,
            "risk_distribution": {
                "high": high_count,
                "medium": med_count,
                "low": low_count,
                "high_pct": round(high_count / total_inferences * 100, 1) if total_inferences else 0,
                "medium_pct": round(med_count / total_inferences * 100, 1) if total_inferences else 0,
                "low_pct": round(low_count / total_inferences * 100, 1) if total_inferences else 0,
            },
            "avg_probability": round(avg_prob, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "drift_status": "Drift Alert" if drift_detected else "Normal",
            "active_versions": active_versions,
            "status": "Healthy",
        }
