import { useEffect, useState } from 'react';
import { patientsAPI, predictionsAPI } from '../services/api';
import {
  Brain,
  Heart,
  Calendar,
  Home,
  Shield,
  Stethoscope,
  Pill,
  BookOpen,
  ArrowRight,
  AlertCircle,
} from 'lucide-react';
import { patientLabel } from '../utils/patients';

export default function ClinicalInsights() {
  const [patients, setPatients] = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    patientsAPI.list(0, 100).then((res) => setPatients(res.data)).catch(console.error);
  }, []);

  const loadInsights = async () => {
    if (!selectedId) return;
    setLoading(true);
    try {
      const res = await predictionsAPI.clinicalInsights(parseInt(selectedId));
      setInsights(res.data);
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to load insights');
    } finally {
      setLoading(false);
    }
  };

  const categoryClass = (cat) => {
    if (cat === 'High') return 'risk-high';
    if (cat === 'Medium') return 'risk-medium';
    return 'risk-low';
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold">Clinical Decision Support Engine</h1>
        <p className="text-gray-500">
          Multi-pillar, evidence-based recommendations and care pathways connected directly to patient clinical data and risk factors
        </p>
      </div>

      <div className="card mb-8">
        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="block text-sm font-medium mb-1">Select Patient</label>
            <select
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              className="w-full px-3 py-2 border rounded-lg text-sm bg-white"
            >
              <option value="">Choose patient...</option>
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {patientLabel(p)} — {p.age}, {p.gender}
                </option>
              ))}
            </select>
          </div>
          <button
            onClick={loadInsights}
            disabled={loading || !selectedId}
            className="btn-primary disabled:opacity-50 text-sm"
          >
            {loading ? 'Analyzing Clinical Profile...' : 'Generate Decision Support'}
          </button>
        </div>
      </div>

      {insights && (
        <>
          {/* Patient Overview Header Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div className="card text-center">
              <p className="text-xs text-gray-500 uppercase tracking-wider font-semibold">Patient Profile</p>
              <p className="text-lg font-bold text-gray-800 mt-1">
                {insights.patient_name || insights.patient_code}
              </p>
              {insights.patient_name && (
                <p className="text-xs text-gray-500">ID: {insights.patient_code}</p>
              )}
            </div>
            <div className="card text-center">
              <p className="text-xs text-gray-500 uppercase tracking-wider font-semibold">Risk Category</p>
              <span
                className={`inline-block mt-2 px-3 py-1 rounded-full text-xs font-bold ${categoryClass(
                  insights.risk_category
                )}`}
              >
                {insights.risk_category} Risk
              </span>
            </div>
            <div className="card text-center">
              <p className="text-xs text-gray-500 uppercase tracking-wider font-semibold">Risk Score</p>
              <p className="text-2xl font-bold text-gray-800 mt-1">{insights.risk_score}%</p>
            </div>
            <div className="card text-center">
              <p className="text-xs text-gray-500 uppercase tracking-wider font-semibold">Calibrated Probability</p>
              <p className="text-2xl font-bold text-primary-600 mt-1">
                {(insights.readmission_probability * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          {/* Key Clinical Risk Factors with Triggers */}
          <div className="card mb-8">
            <h3 className="font-semibold mb-3 flex items-center gap-2 text-red-900">
              <Shield className="w-5 h-5 text-red-600" /> Patient Clinical Risk Factors & Triggers
            </h3>
            <p className="text-xs text-gray-500 mb-4">
              Identified from admission history, laboratory results, medications, and ICD diagnostic patterns
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {insights.key_risk_factors.map((f, i) => {
                const parts = f.split('[');
                const desc = parts[0];
                const trigger = parts.length > 1 ? parts[1].replace(']', '') : null;

                return (
                  <div key={i} className="p-3 bg-red-50/70 border border-red-100 rounded-lg flex items-start gap-2">
                    <span className="text-red-600 font-bold text-xs mt-0.5">{i + 1}.</span>
                    <div className="flex-1">
                      <p className="text-xs font-semibold text-gray-800">{desc}</p>
                      {trigger && (
                        <span className="inline-block mt-1 px-2 py-0.5 rounded text-[10px] font-medium bg-red-200 text-red-900">
                          {trigger}
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Four Clinical Pillars of Decision Support */}
          <div className="mb-8">
            <h3 className="text-lg font-bold mb-4 text-gray-800 flex items-center gap-2">
              <Brain className="w-6 h-6 text-primary-600" /> Comprehensive Clinical Action Pillars
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Pillar 1 */}
              <div className="card border-l-4 border-l-rose-500">
                <h4 className="font-bold text-sm text-gray-900 mb-3 flex items-center gap-2">
                  <Stethoscope className="w-4 h-4 text-rose-600" /> Pillar 1: Immediate Clinical & Inpatient Interventions
                </h4>
                <ul className="space-y-2">
                  {(insights.clinical_pillars?.immediate_interventions || []).map((item, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-xs text-gray-700 bg-rose-50/50 p-2 rounded">
                      <ArrowRight className="w-3.5 h-3.5 text-rose-500 shrink-0 mt-0.5" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Pillar 2 */}
              <div className="card border-l-4 border-l-blue-500">
                <h4 className="font-bold text-sm text-gray-900 mb-3 flex items-center gap-2">
                  <Home className="w-4 h-4 text-blue-600" /> Pillar 2: Transition of Care & Post-Discharge Planning
                </h4>
                <ul className="space-y-2">
                  {(insights.clinical_pillars?.transition_care || []).map((item, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-xs text-gray-700 bg-blue-50/50 p-2 rounded">
                      <ArrowRight className="w-3.5 h-3.5 text-blue-500 shrink-0 mt-0.5" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Pillar 3 */}
              <div className="card border-l-4 border-l-purple-500">
                <h4 className="font-bold text-sm text-gray-900 mb-3 flex items-center gap-2">
                  <Pill className="w-4 h-4 text-purple-600" /> Pillar 3: Medication Safety & Deprescribing Protocols
                </h4>
                <ul className="space-y-2">
                  {(insights.clinical_pillars?.medication_safety || []).map((item, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-xs text-gray-700 bg-purple-50/50 p-2 rounded">
                      <ArrowRight className="w-3.5 h-3.5 text-purple-500 shrink-0 mt-0.5" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Pillar 4 */}
              <div className="card border-l-4 border-l-emerald-500">
                <h4 className="font-bold text-sm text-gray-900 mb-3 flex items-center gap-2">
                  <BookOpen className="w-4 h-4 text-emerald-600" /> Pillar 4: Patient Self-Management & Disease Education
                </h4>
                <ul className="space-y-2">
                  {(insights.clinical_pillars?.patient_education || []).map((item, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-xs text-gray-700 bg-emerald-50/50 p-2 rounded">
                      <ArrowRight className="w-3.5 h-3.5 text-emerald-500 shrink-0 mt-0.5" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          {/* Follow-up Plan and Discharge Support */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="card">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Calendar className="w-5 h-5 text-blue-600" /> Structured Follow-Up Schedule
              </h3>
              <ul className="space-y-2">
                {insights.follow_up_plan.map((f, i) => (
                  <li key={i} className="flex items-start gap-2 p-2.5 bg-blue-50/60 rounded-lg text-xs text-gray-700">
                    <span className="text-blue-600 font-bold">{i + 1}.</span> {f}
                  </li>
                ))}
              </ul>
            </div>

            <div className="card">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Heart className="w-5 h-5 text-green-600" /> Tailored Discharge Support Directives
              </h3>
              <ul className="space-y-2">
                {insights.discharge_support.map((d, i) => (
                  <li key={i} className="flex items-start gap-2 p-2.5 bg-green-50/60 rounded-lg text-xs text-gray-700">
                    <span className="text-green-600 font-bold">✓</span> {d}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </>
      )}

      {!insights && !loading && (
        <div className="card text-center py-16">
          <Brain className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <p className="text-gray-500 font-medium">Select a patient above to generate comprehensive clinical decision support</p>
          <p className="text-xs text-gray-400 mt-1">Evaluates clinical factors, comorbidities, medication burden, and produces 4-pillar recommendations</p>
        </div>
      )}
    </div>
  );
}
