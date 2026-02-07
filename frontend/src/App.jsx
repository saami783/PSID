/**
 * Composant principal de l'application
 * Point d'entrée qui orchestre les différentes features et le routing
 */
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import InferencePage from './pages/Inference';
import { AnalyticsDashboard } from './features/analytics/AnalyticsDashboard';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/analytics" element={<AnalyticsDashboard />} />
        <Route path="/inference" element={<InferencePage />} />
      </Routes>
    </Router>
  );
}

export default App;
