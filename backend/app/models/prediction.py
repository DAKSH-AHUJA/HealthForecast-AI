from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


class RiskPrediction(Base):
    __tablename__ = "risk_predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_category = Column(String(20), nullable=False)
    readmission_probability = Column(Float, nullable=False)
    model_used = Column(String(50))
    feature_importance = Column(Text)
    clinical_insights = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="risk_predictions")


class ReadmissionForecast(Base):
    __tablename__ = "readmission_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    forecast_period_days = Column(Integer, default=30)
    readmission_probability = Column(Float, nullable=False)
    confidence_score = Column(Float)
    risk_factors = Column(Text)
    recommendations = Column(Text)
    forecast_report = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="readmission_forecasts")


class ModelVersionRecord(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), nullable=False, index=True)  # "random_forest" or "xgboost"
    version = Column(String(50), nullable=False, index=True)      # e.g. "v1.0.0"
    is_active = Column(Boolean, default=False, index=True)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    optimal_threshold = Column(Float, default=0.5)
    training_sample_size = Column(Integer, default=20000)
    hyperparameters = Column(Text, nullable=True)  # JSON string
    model_path = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)
    trained_by = Column(String(100), default="system_admin")
    trained_at = Column(DateTime, default=datetime.utcnow)


class ModelInferenceLog(Base):
    __tablename__ = "model_inference_logs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), nullable=False, index=True)
    model_version = Column(String(50), nullable=True)
    risk_score = Column(Float, nullable=False)
    risk_category = Column(String(20), nullable=False)
    readmission_probability = Column(Float, nullable=False)
    latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

