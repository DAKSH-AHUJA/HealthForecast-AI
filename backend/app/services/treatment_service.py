from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models.patient import Admission, Patient, Treatment
from app.models.user import User, UserRole


class TreatmentService:
    @staticmethod
    def get_aggregate_effectiveness(db: Session) -> Dict[str, Any]:
        treatments = db.query(Treatment).all()
        total = len(treatments)

        if total == 0:
            return {
                "total_treatments": 0,
                "active_treatments": 0,
                "completed_treatments": 0,
                "overall_recovery_rate": 0.0,
                "overall_improvement_rate": 0.0,
                "overall_readmission_rate": 0.0,
                "medication_performance": [],
                "outcome_distribution": {
                    "Recovered": 0,
                    "Improved": 0,
                    "Stable": 0,
                    "Readmitted": 0,
                    "Adverse Reaction": 0,
                },
            }

        active_count = sum(1 for t in treatments if t.status == "active")
        completed_count = sum(1 for t in treatments if t.status == "completed")

        outcome_counts = {
            "Recovered": 0,
            "Improved": 0,
            "Stable": 0,
            "Readmitted": 0,
            "Adverse Reaction": 0,
        }

        # Group by medication
        by_med: Dict[str, List[Treatment]] = {}
        for t in treatments:
            med = (t.medication or "Unknown").capitalize()
            by_med.setdefault(med, []).append(t)

            outcome = t.outcome or "Stable"
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        recovered_total = outcome_counts.get("Recovered", 0)
        improved_total = outcome_counts.get("Improved", 0)
        readmitted_total = outcome_counts.get("Readmitted", 0)

        med_performance = []
        for med, items in by_med.items():
            count = len(items)
            rec = sum(1 for i in items if i.outcome == "Recovered")
            imp = sum(1 for i in items if i.outcome == "Improved")
            stable = sum(1 for i in items if i.outcome == "Stable")
            readm = sum(1 for i in items if i.outcome == "Readmitted")
            adv = sum(1 for i in items if i.outcome == "Adverse Reaction")

            durations = []
            for i in items:
                if i.start_date and i.end_date:
                    durations.append((i.end_date - i.start_date).days)
                elif i.start_date:
                    durations.append(max((datetime.utcnow() - i.start_date).days, 1))
            avg_days = round(sum(durations) / len(durations), 1) if durations else 14.0

            recovery_pct = round((rec / count) * 100, 1)
            improvement_pct = round((imp / count) * 100, 1)
            readmission_pct = round((readm / count) * 100, 1)

            # Effectiveness score combines recovery and improvement minus readmission penalty
            effectiveness_score = round(
                min(max((recovery_pct * 1.0 + improvement_pct * 0.7) - (readmission_pct * 0.8), 0), 100), 1
            )

            med_performance.append({
                "medication": med,
                "total_treated": count,
                "recovered": rec,
                "improved": imp,
                "stable": stable,
                "readmitted": readm,
                "adverse_reaction": adv,
                "recovery_rate": recovery_pct,
                "improvement_rate": improvement_pct,
                "readmission_rate": readmission_pct,
                "avg_duration_days": avg_days,
                "effectiveness_score": effectiveness_score,
            })

        # Sort by total treated descending
        med_performance.sort(key=lambda x: x["total_treated"], reverse=True)

        return {
            "total_treatments": total,
            "active_treatments": active_count,
            "completed_treatments": completed_count,
            "overall_recovery_rate": round((recovered_total / total) * 100, 1),
            "overall_improvement_rate": round((improved_total / total) * 100, 1),
            "overall_readmission_rate": round((readmitted_total / total) * 100, 1),
            "medication_performance": med_performance,
            "outcome_distribution": outcome_counts,
        }

    @staticmethod
    def get_patient_recovery_analysis(db: Session, patient: Patient) -> Dict[str, Any]:
        treatments = (
            db.query(Treatment)
            .filter(Treatment.patient_id == patient.id)
            .order_by(Treatment.start_date.desc())
            .all()
        )

        active_treatments = [t for t in treatments if t.status == "active"]
        completed_treatments = [t for t in treatments if t.status != "active"]

        # Calculate recovery indicators
        adverse_events = [t for t in treatments if t.outcome == "Adverse Reaction"]
        recovered_treatments = [t for t in treatments if t.outcome == "Recovered"]
        readmitted_treatments = [t for t in treatments if t.outcome == "Readmitted"]

        # Clinical response evaluation
        has_high_a1c = patient.a1cresult in [">7", ">8"]
        has_high_glu = patient.max_glu_serum in [">200", ">300"]
        polypharmacy = (patient.num_medications or 0) > 10

        if recovered_treatments and not readmitted_treatments and not has_high_a1c:
            response_rating = "Favorable Recovery"
            status_color = "green"
        elif readmitted_treatments or (has_high_a1c and has_high_glu):
            response_rating = "Suboptimal Response / High Risk"
            status_color = "red"
        else:
            response_rating = "Moderate Progress"
            status_color = "amber"

        # Treatment recommendations based on outcomes and patient indicators
        recommendations = []
        if patient.diabetes_med == "No":
            recommendations.append("Initiate first-line glycemic therapy (Metformin 500mg daily titration).")
        elif has_high_a1c:
            recommendations.append("A1C elevated despite current medication: consider adding basal insulin or GLP-1 receptor agonist.")

        if polypharmacy:
            recommendations.append(f"Patient prescribed {patient.num_medications} medications: conduct structured deprescribing audit.")

        if adverse_events:
            recommendations.append("Prior adverse reaction noted: re-evaluate tolerance and switch medication class.")

        if patient.readmitted == "<30":
            recommendations.append("Patient has history of 30-day readmission: schedule mandatory 7-day post-discharge pharmacology reconciliation.")

        if not recommendations:
            recommendations.append("Maintain current therapy and monitor blood glucose log at next regular follow-up.")

        # Serialize treatments
        serialized_treatments = []
        for t in treatments:
            duration_days = None
            if t.start_date and t.end_date:
                duration_days = (t.end_date - t.start_date).days
            elif t.start_date:
                duration_days = (datetime.utcnow() - t.start_date).days

            serialized_treatments.append({
                "id": t.id,
                "medication": t.medication,
                "dosage": t.dosage,
                "status": t.status,
                "outcome": t.outcome or "Pending",
                "start_date": t.start_date.isoformat() if t.start_date else None,
                "end_date": t.end_date.isoformat() if t.end_date else None,
                "duration_days": duration_days,
            })

        return {
            "patient_id": patient.id,
            "patient_code": patient.patient_id,
            "patient_name": patient.full_name,
            "total_treatments": len(treatments),
            "active_count": len(active_treatments),
            "completed_count": len(completed_treatments),
            "response_rating": response_rating,
            "status_color": status_color,
            "recommendations": recommendations,
            "treatments": serialized_treatments,
        }
