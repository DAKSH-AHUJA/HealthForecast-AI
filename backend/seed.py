"""Initialize database, seed users, import dataset, and train ML models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.auth.security import get_password_hash
from app.database import Base, SessionLocal, engine, ensure_schema
from app.ml.predictor import DataPreprocessor, ModelTrainer
from app.models.user import User, UserRole
from app.services.prediction_service import DatasetService, backfill_patient_names


def seed_users(db):
    users = [
        ("doctor1", "priya.mehta@healthforecast.com", "Dr. Priya Mehta", UserRole.DOCTOR, "Cardiology", "doctor123"),
        ("admin1", "ananya.krishnan@healthforecast.com", "Ananya Krishnan", UserRole.HOSPITAL_ADMIN, "Administration", "admin123"),
        ("researcher1", "emily.chen@healthforecast.com", "Dr. Emily Chen", UserRole.RESEARCHER, "Clinical Research", "research123"),
        ("sysadmin", "rajesh.iyer@healthforecast.com", "Rajesh Iyer", UserRole.SYSTEM_ADMIN, "IT", "sysadmin123"),
    ]
    for username, email, full_name, role, dept, password in users:
        user = db.query(User).filter(User.username == username).first()
        if user:
            user.full_name = full_name
            user.email = email
            user.department = dept
            user.role = role
        else:
            db.add(User(
                email=email,
                username=username,
                hashed_password=get_password_hash(password),
                full_name=full_name,
                role=role,
                department=dept,
            ))
    db.commit()
    print("[OK] Users seeded")


from datetime import datetime, timedelta
import random

def assign_patients_to_doctor(db):
    doctor = db.query(User).filter(User.username == "doctor1").first()
    if doctor:
        from app.models.patient import Patient
        patients = db.query(Patient).limit(100).all()
        for p in patients:
            p.assigned_doctor_id = doctor.id
        db.commit()
        print(f"[OK] Assigned {len(patients)} patients to Dr. Priya Mehta")


def seed_treatments(db):
    from app.models.patient import Patient, Treatment
    existing_count = db.query(Treatment).count()
    if existing_count > 0:
        print(f"[OK] Treatments already present ({existing_count} records)")
        return

    patients = db.query(Patient).all()
    count = 0
    now = datetime.utcnow()

    med_options = [
        ("Metformin", ["500mg daily", "850mg twice daily", "1000mg twice daily"]),
        ("Insulin", ["10 units subcutaneously", "20 units before meals", "NPH 15 units at bedtime"]),
        ("Glipizide", ["5mg daily", "10mg twice daily"]),
        ("Glyburide", ["2.5mg daily", "5mg twice daily"]),
        ("Pioglitazone", ["15mg daily", "30mg daily"]),
        ("Sitagliptin", ["50mg daily", "100mg daily"]),
        ("Empagliflozin", ["10mg daily", "25mg daily"]),
    ]

    for p in patients:
        # Determine realistic outcome based on readmission and A1C
        if p.readmitted == "<30":
            outcomes = ["Readmitted", "Readmitted", "Stable", "Adverse Reaction"]
        elif p.a1cresult in [">7", ">8"]:
            outcomes = ["Improved", "Stable", "Improved", "Recovered"]
        else:
            outcomes = ["Recovered", "Recovered", "Improved", "Stable"]

        # Add 1 to 2 treatments per patient
        num_tx = 2 if (p.num_medications or 0) > 8 else 1
        for idx in range(num_tx):
            med, doses = random.choice(med_options)
            dosage = random.choice(doses)
            outcome = random.choice(outcomes)
            status = "active" if idx == 0 and outcome in ["Improved", "Stable"] else "completed"
            
            days_ago = random.randint(15, 60)
            start_date = now - timedelta(days=days_ago)
            end_date = (start_date + timedelta(days=random.randint(7, 21))) if status == "completed" else None

            db.add(Treatment(
                patient_id=p.id,
                medication=med,
                dosage=dosage,
                status=status,
                outcome=outcome,
                start_date=start_date,
                end_date=end_date,
            ))
            count += 1

    db.commit()
    print(f"[OK] Seeded {count} realistic treatments across {len(patients)} patients")


def main():
    print("Initializing HealthForecast AI...")
    Base.metadata.create_all(bind=engine)
    ensure_schema()
    db = SessionLocal()

    try:
        seed_users(db)

        dataset_path = Path(__file__).parent / "data" / "diabetic_data.csv"
        if dataset_path.exists():
            result = DatasetService.load_and_import(db, limit=500)
            print(f"[OK] Imported {result['imported']} patients from dataset")
            named = backfill_patient_names(db)
            print(f"[OK] Patient names assigned ({named} updated)")
            assign_patients_to_doctor(db)
            seed_treatments(db)

            from app.services.model_manager import ModelManagerService
            model_mgr = ModelManagerService()

            print("Calibrating & Training ML models (20k sample with balanced weights)...")
            rf = model_mgr.controlled_retrain(
                db=db,
                model_type="random_forest",
                sample_size=20000,
                class_weight_strategy="balanced",
                trained_by="system_admin",
                notes="Initial production model with balanced class weights and threshold calibration",
            )
            xgb = model_mgr.controlled_retrain(
                db=db,
                model_type="xgboost",
                sample_size=20000,
                class_weight_strategy="balanced",
                trained_by="system_admin",
                notes="Initial production model with scale_pos_weight and threshold calibration",
            )
            print(f"[OK] Random Forest v1.0.0 - Accuracy: {rf['accuracy']:.4f}, Recall: {rf['recall']:.4f}, F1: {rf['f1_score']:.4f}, ROC-AUC: {rf['roc_auc']:.4f}, Threshold: {rf['optimal_threshold']}")
            print(f"[OK] XGBoost v1.0.0 - Accuracy: {xgb['accuracy']:.4f}, Recall: {xgb['recall']:.4f}, F1: {xgb['f1_score']:.4f}, ROC-AUC: {xgb['roc_auc']:.4f}, Threshold: {xgb['optimal_threshold']}")
        else:
            print("[WARN] Dataset not found. Run download_dataset.py first.")

        print("\nDefault login credentials:")
        print("  Dr. Priya Mehta (Doctor):           doctor1 / doctor123")
        print("  Ananya Krishnan (Hospital Admin):   admin1 / admin123")
        print("  Dr. Emily Chen (Researcher):        researcher1 / research123")
        print("  Rajesh Iyer (System Admin):         sysadmin / sysadmin123")
    finally:
        db.close()


if __name__ == "__main__":
    main()

