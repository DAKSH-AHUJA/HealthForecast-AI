from typing import Any, Dict, List, Optional

from app.models.patient import Patient


class ClinicalInsightsEngine:
    @staticmethod
    def generate_insights(
        patient: Patient,
        risk_score: float,
        risk_category: str,
        readmission_probability: float,
        feature_importance: Dict[str, float],
    ) -> Dict[str, Any]:
        risk_factors = ClinicalInsightsEngine._identify_risk_factors(patient, feature_importance)
        pillars = ClinicalInsightsEngine._structured_clinical_pillars(patient, risk_category, readmission_probability)
        
        # Flattened recommendations for backward compatibility
        care_recommendations = (
            pillars["immediate_interventions"]
            + pillars["transition_care"]
            + pillars["medication_safety"]
            + pillars["patient_education"]
        )

        follow_up = ClinicalInsightsEngine._follow_up_plan(patient, risk_category, readmission_probability)
        discharge = ClinicalInsightsEngine._discharge_support(patient, risk_category)

        # Dynamic clinical summary
        insights_text = [
            f"Patient {patient.full_name or patient.patient_id} ({patient.patient_id}) classified as {risk_category} risk ({risk_score}%).",
            f"Calibrated 30-day readmission probability: {readmission_probability * 100:.1f}%.",
        ]
        if patient.time_in_hospital and patient.time_in_hospital >= 7:
            insights_text.append(f"Extended stay of {patient.time_in_hospital} days increases hospital-acquired deconditioning risk.")
        if patient.num_medications and patient.num_medications >= 10:
            insights_text.append(f"Polypharmacy alert: {patient.num_medications} medications on profile. Perform drug interaction screen.")
        if patient.number_inpatient and patient.number_inpatient >= 2:
            insights_text.append(f"Frequent prior hospitalizations ({patient.number_inpatient} visits) strongly signal chronic instability.")
        if patient.a1cresult in [">7", ">8"]:
            insights_text.append(f"Suboptimal glycemic control (HbA1c {patient.a1cresult}) elevates acute complication and infection risks.")

        return {
            "key_risk_factors": risk_factors,
            "care_recommendations": care_recommendations,
            "clinical_pillars": pillars,
            "follow_up_plan": follow_up,
            "discharge_support": discharge,
            "clinical_insights": insights_text,
        }

    @staticmethod
    def _is_cardiovascular(code: Optional[str]) -> bool:
        if not code:
            return False
        try:
            val = float(code)
            return 390 <= val <= 459 or val == 785
        except ValueError:
            return code.startswith("4") or "heart" in code.lower()

    @staticmethod
    def _is_respiratory(code: Optional[str]) -> bool:
        if not code:
            return False
        try:
            val = float(code)
            return 460 <= val <= 519 or val == 786
        except ValueError:
            return "respiratory" in code.lower() or "copd" in code.lower()

    @staticmethod
    def _is_renal(code: Optional[str]) -> bool:
        if not code:
            return False
        try:
            val = float(code)
            return 580 <= val <= 589
        except ValueError:
            return "renal" in code.lower() or "kidney" in code.lower()

    @staticmethod
    def _identify_risk_factors(patient: Patient, importance: Dict[str, float]) -> List[str]:
        factors = []
        # Direct clinical data triggers
        if patient.a1cresult in [">7", ">8"]:
            factors.append(f"Severe Glycemic Dysregulation (HbA1c {patient.a1cresult}) [Triggered by Lab Result]")
        if patient.max_glu_serum in [">200", ">300"]:
            factors.append(f"Acute Hyperglycemia (Serum Glucose {patient.max_glu_serum}) [Triggered by Blood Glucose]")
        if patient.number_inpatient and patient.number_inpatient >= 2:
            factors.append(f"Recurrent Hospitalization Pattern ({patient.number_inpatient} prior admissions in past year) [Triggered by Utilization]")
        if patient.number_emergency and patient.number_emergency >= 2:
            factors.append(f"Frequent Emergency Department Utilization ({patient.number_emergency} visits) [Triggered by Acute Care Visits]")
        if patient.num_medications and patient.num_medications >= 10:
            factors.append(f"Complex Polypharmacy ({patient.num_medications} active medications) [Triggered by Prescription Burden]")
        if patient.time_in_hospital and patient.time_in_hospital >= 7:
            factors.append(f"Protracted Hospital Stay ({patient.time_in_hospital} days) [Triggered by Inpatient Length of Stay]")
        if patient.number_diagnoses and patient.number_diagnoses >= 7:
            factors.append(f"High Comorbidity Burden ({patient.number_diagnoses} concurrent diagnoses) [Triggered by Diagnostic Count]")
        if patient.change == "Ch":
            factors.append("Active Diabetes Medication Regimen Adjustment during Admission [Triggered by Inpatient Rx Change]")
        if patient.diabetes_med == "No":
            factors.append("Absence of Prescribed Diabetes Pharmacotherapy despite Diagnosis [Triggered by Rx Gap]")

        # Clinical diagnoses flags
        for diag in [patient.diag_1, patient.diag_2, patient.diag_3]:
            if ClinicalInsightsEngine._is_cardiovascular(diag):
                factors.append("Underlying Cardiovascular Disease / Congestive Heart Failure Comorbidity [Triggered by ICD Diagnosis]")
                break
            elif ClinicalInsightsEngine._is_renal(diag):
                factors.append("Renal Impairment / Chronic Kidney Disease Comorbidity [Triggered by ICD Diagnosis]")
                break

        # Top model predictive factor highlights
        for feat, _ in list(importance.items())[:3]:
            clean = feat.replace("cat__", "").replace("num__", "")
            if clean not in str(factors):
                factors.append(f"Predictive ML Factor: {clean}")

        return factors[:8] if factors else ["Standard clinical risk profile — routine surveillance advised"]

    @staticmethod
    def _structured_clinical_pillars(patient: Patient, category: str, probability: float) -> Dict[str, List[str]]:
        immediate = []
        transition = []
        med_safety = []
        education = []

        # 1. Immediate Inpatient / Clinical Interventions
        if patient.max_glu_serum in [">200", ">300"]:
            immediate.append("Order pre-meal and bedtime blood glucose checks; titrate basal-bolus insulin sliding scale.")
        if patient.a1cresult in [">7", ">8"]:
            immediate.append("Request in-hospital Endocrinology consult for comprehensive glycemic regimen redesign.")
        if ClinicalInsightsEngine._is_cardiovascular(patient.diag_1) or ClinicalInsightsEngine._is_cardiovascular(patient.diag_2):
            immediate.append("Institute strict daily weights, telemetry monitoring, and fluid restriction (<2L/day) if volume overloaded.")
        if ClinicalInsightsEngine._is_renal(patient.diag_1) or ClinicalInsightsEngine._is_renal(patient.diag_2):
            immediate.append("Order repeat basic metabolic panel (BMP) 24 hours prior to discharge to confirm stable eGFR and potassium.")
        if not immediate:
            immediate.append("Conduct vital sign stability check and discharge criteria assessment 24 hours prior to release.")

        # 2. Transition of Care & Discharge Planning
        if category == "High" or probability >= 0.50:
            transition.append("Assign Dedicated Transitional Care Coordinator (TCM) for 30-day post-discharge care bridge.")
            transition.append("Schedule in-person Primary Care Physician (PCP) appointment within 7 days of discharge.")
            transition.append("Mandate 48-hour post-discharge telephone wellness check by clinic care manager.")
        elif category == "Medium":
            transition.append("Schedule outpatient follow-up visit within 10 to 14 days of discharge.")
            transition.append("Conduct post-discharge phone check-in at Day 5 to assess symptom stability.")
        else:
            transition.append("Schedule routine follow-up appointment within 30 days.")
            transition.append("Provide electronic patient portal summary and emergency contact directives.")

        if patient.time_in_hospital and patient.time_in_hospital >= 7:
            transition.append("Complete physical/occupational therapy discharge evaluation for home safety and assistive devices.")

        # 3. Medication Safety & Deprescribing
        if patient.num_medications and patient.num_medications >= 10:
            med_safety.append(f"Conduct comprehensive clinical pharmacist medication reconciliation for {patient.num_medications} drugs.")
            med_safety.append("Screen for anticholinergic cognitive burden and apply Beers Criteria for high-risk geriatric medications.")
        if patient.change == "Ch":
            med_safety.append("Provide visual medication calendar detailing newly adjusted dosage and discontinuation instructions.")
        if patient.diabetes_med == "No":
            med_safety.append("Evaluate initiation of guideline-directed metformin or GLP-1 receptor agonist prior to discharge.")
        if ClinicalInsightsEngine._is_renal(patient.diag_1):
            med_safety.append("Audit all discharge prescriptions for renal clearance adjustments; strictly avoid OTC NSAIDs.")
        if not med_safety:
            med_safety.append("Perform discharge prescription reconciliation with outpatient retail pharmacy.")

        # 4. Patient Self-Management & Disease Education
        if patient.a1cresult in [">7", ">8"] or patient.max_glu_serum in [">200", ">300"]:
            education.append("Deliver certified diabetes self-management education (DSME) on recognizing hypoglycemia vs hyperglycemia.")
            education.append("Provide continuous glucose monitoring (CGM) or home glucometer with target logging chart.")
        if ClinicalInsightsEngine._is_cardiovascular(patient.diag_1):
            education.append("Educate patient on the 'Rule of 2s': notify clinic if weight increases by 2 lbs overnight or 5 lbs in a week.")
            education.append("Provide dietary counseling on low-sodium intake (< 2,000 mg/day) and symptom red flags (worsening dyspnea, edema).")
        education.append("Supply written 'Red Flag Warning Signs' sheet with direct 24/7 clinical triage telephone line.")

        return {
            "immediate_interventions": immediate,
            "transition_care": transition,
            "medication_safety": med_safety,
            "patient_education": education,
        }

    @staticmethod
    def _follow_up_plan(patient: Patient, category: str, probability: float) -> List[str]:
        if category == "High" or probability >= 0.50:
            return [
                "Day 2 (48 hrs post-discharge): Structured telephonic outreach by registered nurse to verify medication access.",
                "Day 7: Comprehensive clinic visit with primary physician and clinical pharmacist medication reconciliation.",
                "Day 14: Mid-month lab follow-up (BMP, fasting glucose, electrolyte panel as indicated).",
                "Day 21: Care coordinator check-in for symptom monitoring and treatment adherence evaluation.",
                "Day 30: Formal 30-day transition closure assessment and long-term care management plan.",
            ]
        if category == "Medium":
            return [
                "Day 3-5: Routine nurse phone check-in to confirm prescription fill and address recovery questions.",
                "Day 14: In-person or telehealth physician evaluation and symptom review.",
                "Day 30: Routine outcome check-up and preventive screening interval planning.",
            ]
        return [
            "Day 7: Automated patient portal wellness check-in.",
            "Day 30: Standard post-discharge follow-up visit with primary care provider.",
        ]

    @staticmethod
    def _discharge_support(patient: Patient, category: str) -> List[str]:
        support = [
            "Provide written discharge instructions in patient's preferred language at 6th-grade reading level.",
            "Confirm prescription medications are electronically sent and filled prior to hospital departure.",
        ]
        if category in ["High", "Medium"]:
            support.extend([
                "Coordinate home health nursing visits for vital monitoring and medication management.",
                "Provide direct 24/7 on-call triage hotline number to bypass emergency department for non-critical concerns.",
                "Facilitate medical transportation assistance for upcoming 7-day follow-up appointment.",
            ])
        if patient.age in ["[70-80)", "[80-90)", "[90-100)"]:
            support.append("Involve family caregiver in bedside discharge teach-back and home equipment setup.")
        if patient.number_emergency and patient.number_emergency >= 2:
            support.append("Establish urgent care alternative care pathway to avert avoidable emergency room visits.")
        return support

    @staticmethod
    def generate_forecast_report(
        patient: Patient,
        probability: float,
        risk_factors: List[str],
        recommendations: List[str],
        period_days: int,
    ) -> str:
        name_line = f"Patient: {patient.full_name} ({patient.patient_id})\n" if patient.full_name else f"Patient ID: {patient.patient_id}\n"
        category = "High" if probability >= 0.55 else "Medium" if probability >= 0.35 else "Low"
        return (
            f"CLINICAL READMISSION FORECAST & DECISION SUPPORT REPORT\n"
            f"{'=' * 60}\n"
            f"{name_line}"
            f"Forecast Evaluation Window: {period_days} days\n"
            f"Calibrated Readmission Probability: {probability * 100:.1f}%\n"
            f"Stratified Risk Category: {category}\n"
            f"Clinical Status: {'Requires Immediate Transitional Care Management' if category == 'High' else 'Standard Clinical Surveillance'}\n\n"
            f"KEY IDENTIFIED CLINICAL RISK FACTORS:\n"
            + "\n".join(f"  • {f}" for f in risk_factors)
            + f"\n\nACTIONABLE CARE RECOMMENDATIONS:\n"
            + "\n".join(f"  • {r}" for r in recommendations)
            + f"\n\nGenerated by HealthForecast AI Clinical Decision Support System"
        )

