import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { patientsAPI, predictionsAPI, authAPI } from '../services/api';
import { useAuth } from '../context/AuthContext';
import { Search, AlertTriangle, Pill, UserCheck, Shield } from 'lucide-react';

export default function Patients() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [patients, setPatients] = useState([]);
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [predicting, setPredicting] = useState(false);
  const [assigningPatient, setAssigningPatient] = useState(null);
  const [selectedDoctorId, setSelectedDoctorId] = useState('');

  const isAdmin = user?.role === 'system_admin' || user?.role === 'hospital_admin';
  const isDoctor = user?.role === 'doctor';
  const isResearcher = user?.role === 'researcher';

  useEffect(() => {
    loadPatients();
    if (isAdmin) {
      authAPI.listUsers()
        .then((res) => {
          const docList = res.data.filter((u) => u.role === 'doctor');
          setDoctors(docList);
          if (docList.length > 0) setSelectedDoctorId(docList[0].id.toString());
        })
        .catch(console.error);
    }
  }, []);

  const loadPatients = () => {
    setLoading(true);
    patientsAPI.list(0, 100)
      .then((res) => setPatients(res.data))
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  const filtered = patients.filter((p) => {
    const haystack = `${p.full_name || ''} ${p.patient_id || ''} ${p.gender || ''} ${p.age || ''}`.toLowerCase();
    return haystack.includes(search.toLowerCase());
  });

  const runPrediction = async (patientId) => {
    setPredicting(true);
    try {
      await predictionsAPI.predictRisk(patientId);
      alert('Risk prediction completed successfully!');
    } catch (err) {
      alert(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setPredicting(false);
    }
  };

  const handleAssignDoctor = async (e) => {
    e.preventDefault();
    if (!assigningPatient || !selectedDoctorId) return;
    try {
      await patientsAPI.assignDoctor(assigningPatient.id, parseInt(selectedDoctorId));
      setAssigningPatient(null);
      loadPatients();
    } catch (err) {
      alert(err.response?.data?.detail || 'Assignment failed');
    }
  };

  const riskBadge = (readmitted) => {
    if (readmitted === '<30') return <span className="risk-high px-2 py-1 rounded-full text-xs font-semibold">Readmitted</span>;
    if (readmitted === '>30') return <span className="risk-medium px-2 py-1 rounded-full text-xs font-semibold">Late Readmit</span>;
    return <span className="risk-low px-2 py-1 rounded-full text-xs font-semibold">Not Readmitted</span>;
  };

  if (loading) return <div className="text-gray-500 p-8">Loading patient directory...</div>;

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold">Patient Management</h1>
          <p className="text-gray-500">
            {isResearcher
              ? 'Anonymized patient health records (De-identified research view)'
              : isDoctor
              ? 'Patients under your direct clinical care and assignment'
              : 'Hospital clinical directory — patient records, doctor assignments, and risk tracking'}
          </p>
        </div>
        {isDoctor && (
          <span className="px-3 py-1 bg-blue-50 border border-blue-200 text-blue-800 rounded-full text-xs font-semibold flex items-center gap-1">
            <UserCheck className="w-3.5 h-3.5" /> Showing your assigned patients only
          </span>
        )}
      </div>

      <div className="card mb-6">
        <div className="relative">
          <Search className="absolute left-3 top-2.5 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search by patient name, ID, gender, or age..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg text-sm"
          />
        </div>
      </div>

      <div className="card overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-gray-500">
              <th className="pb-3 pr-4">Patient</th>
              <th className="pb-3 pr-4">Patient ID</th>
              <th className="pb-3 pr-4">Age</th>
              <th className="pb-3 pr-4">Gender</th>
              <th className="pb-3 pr-4">Stay (days)</th>
              <th className="pb-3 pr-4">Medications</th>
              <th className="pb-3 pr-4">Diagnoses</th>
              <th className="pb-3 pr-4">Readmission</th>
              {!isResearcher && <th className="pb-3 pr-4">Assigned Doctor</th>}
              {!isResearcher && <th className="pb-3">Actions</th>}
            </tr>
          </thead>
          <tbody>
            {filtered.map((p) => (
              <tr key={p.id} className="border-b border-gray-50 hover:bg-gray-50">
                <td className="py-3 pr-4 font-semibold text-gray-800">
                  {isResearcher ? 'Anonymized' : (p.full_name || '—')}
                </td>
                <td className="py-3 pr-4 text-gray-500 font-mono text-xs">{p.patient_id}</td>
                <td className="py-3 pr-4">{p.age}</td>
                <td className="py-3 pr-4">{p.gender}</td>
                <td className="py-3 pr-4">{p.time_in_hospital}</td>
                <td className="py-3 pr-4">{p.num_medications}</td>
                <td className="py-3 pr-4">{p.number_diagnoses}</td>
                <td className="py-3 pr-4">{riskBadge(p.readmitted)}</td>
                {!isResearcher && (
                  <td className="py-3 pr-4">
                    {p.assigned_doctor_id ? (
                      <span className="text-xs font-medium text-blue-700 bg-blue-50 px-2 py-0.5 rounded-full">
                        Assigned
                      </span>
                    ) : (
                      <span className="text-xs text-gray-400">Unassigned</span>
                    )}
                  </td>
                )}
                {!isResearcher && (
                  <td className="py-3">
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => runPrediction(p.id)}
                        disabled={predicting}
                        className="flex items-center gap-1 text-xs btn-primary py-1 px-2.5"
                      >
                        <AlertTriangle className="w-3 h-3" /> Risk
                      </button>
                      <button
                        onClick={() => navigate('/treatments')}
                        className="flex items-center gap-1 text-xs btn-secondary py-1 px-2.5"
                      >
                        <Pill className="w-3 h-3" /> Treatments
                      </button>
                      {isAdmin && (
                        <button
                          onClick={() => setAssigningPatient(p)}
                          className="text-xs text-primary-600 hover:text-primary-800 font-medium px-1"
                        >
                          Assign
                        </button>
                      )}
                    </div>
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <p className="text-center text-gray-500 py-8 text-sm">
            No patients found matching your query or assignment.
          </p>
        )}
      </div>

      {/* Assign Doctor Modal */}
      {assigningPatient && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl max-w-sm w-full p-6 shadow-xl">
            <h3 className="text-base font-bold mb-2">Assign Doctor to Patient</h3>
            <p className="text-xs text-gray-500 mb-4">
              Patient: <span className="font-semibold text-gray-700">{assigningPatient.full_name} ({assigningPatient.patient_id})</span>
            </p>
            <form onSubmit={handleAssignDoctor} className="space-y-4">
              <div>
                <label className="block text-xs font-medium mb-1">Select Attending Physician</label>
                <select
                  value={selectedDoctorId}
                  onChange={(e) => setSelectedDoctorId(e.target.value)}
                  className="w-full px-3 py-2 border rounded-lg text-sm bg-white"
                  required
                >
                  {doctors.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.full_name} ({d.department || 'General Medicine'})
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex justify-end gap-2 pt-2 border-t">
                <button
                  type="button"
                  onClick={() => setAssigningPatient(null)}
                  className="btn-secondary text-xs"
                >
                  Cancel
                </button>
                <button type="submit" className="btn-primary text-xs">
                  Confirm Assignment
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
