import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Layout from './components/Layout';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Patients from './pages/Patients';
import RiskPrediction from './pages/RiskPrediction';
import Forecasting from './pages/Forecasting';
import ClinicalInsights from './pages/ClinicalInsights';
import ModelManagement from './pages/ModelManagement';
import TreatmentEffectiveness from './pages/TreatmentEffectiveness';

function ProtectedRoute({ children, requiredRole }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="flex items-center justify-center h-screen">Loading...</div>;
  if (!user) return <Navigate to="/login" />;
  if (requiredRole && user.role !== requiredRole) {
    return <Navigate to="/" replace />;
  }
  return children;
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
        <Route index element={<Dashboard />} />
        <Route path="patients" element={<Patients />} />
        <Route path="treatments" element={<TreatmentEffectiveness />} />
        <Route path="risk-prediction" element={<RiskPrediction />} />
        <Route path="forecasting" element={<Forecasting />} />
        <Route path="clinical-insights" element={<ClinicalInsights />} />
        <Route
          path="models"
          element={
            <ProtectedRoute requiredRole="system_admin">
              <ModelManagement />
            </ProtectedRoute>
          }
        />
      </Route>
    </Routes>
  );
}


export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AuthProvider>
  );
}
