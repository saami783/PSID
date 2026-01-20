/**
 * Dashboard principal pour les analytics
 * Utilise les hooks et composants pour séparer la logique métier du JSX
 */
import { useAnalytics } from './hooks/useAnalytics';
import { Card } from '../../components/ui/Card';
import { SimpleTestChart } from '../../components/charts/SimpleTestChart';
import { SimpleLineChart } from '../../components/charts/SimpleLineChart';
import { SimplePieChart } from '../../components/charts/SimplePieChart';

export function AnalyticsDashboard() {
  const { stats, demographics, pathologies, loading, error } = useAnalytics();
  
  // Ne pas utiliser correlation pour éviter les erreurs

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Chargement des données...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <strong className="font-bold">Erreur: </strong>
          <span className="block sm:inline">{error}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 mb-8">
          PSID - Medical Analytics Dashboard
        </h1>

        {/* Statistiques générales */}
        {stats && (
          <Card title="Informations générales" className="mb-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-gray-600">Total de lignes</p>
                <p className="text-2xl font-bold">{stats.total_rows?.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Total de colonnes</p>
                <p className="text-2xl font-bold">{stats.total_columns}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Pathologies</p>
                <p className="text-2xl font-bold">{stats.pathology_columns?.length}</p>
              </div>
            </div>
          </Card>
        )}

        {/* Statistiques démographiques */}
        {demographics && (
          <Card title="Statistiques démographiques" className="mb-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-gray-600">Patients uniques</p>
                <p className="text-2xl font-bold">{demographics.total_patients?.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Âge moyen</p>
                <p className="text-2xl font-bold">{demographics.average_age?.toFixed(1)} ans</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Images par patient</p>
                <p className="text-2xl font-bold">{demographics.images_per_patient?.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Répartition H/F</p>
                <p className="text-sm">
                  H: {demographics.sex_distribution?.Male?.toLocaleString() || 0} | 
                  F: {demographics.sex_distribution?.Female?.toLocaleString() || 0}
                </p>
              </div>
            </div>
          </Card>
        )}

        {/* Graphiques de test simples */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <Card title="Test Graphique 1 - Barres" className="mb-6">
            <SimpleTestChart />
          </Card>
          <Card title="Test Graphique 2 - Ligne" className="mb-6">
            <SimpleLineChart />
          </Card>
        </div>

        <Card title="Test Graphique 3 - Camembert" className="mb-6">
          <SimplePieChart />
        </Card>
      </div>
    </div>
  );
}
