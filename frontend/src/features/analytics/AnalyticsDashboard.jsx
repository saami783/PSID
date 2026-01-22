/**
 * Dashboard principal pour les analytics
 * 10 visualisations interactives organisées en 4 axes thématiques
 */
import { useAnalytics } from './hooks/useAnalytics';
import { FilterProvider } from '../../context/FilterContext';
import { Card } from '../../components/ui/Card';
import { FilterBar } from '../../components/ui/FilterBar';
import { ErrorBoundary } from '../../components/ui/ErrorBoundary';

// AXE 1 : Démographie & Biais
import { AgePyramid } from '../../components/charts/demographics/AgePyramid';
import { AgeBoxplot } from '../../components/charts/demographics/AgeBoxplot';
import { PatientFrequencyHistogram } from '../../components/charts/demographics/PatientFrequencyHistogram';

// AXE 2 : Panorama Clinique
import { PrevalenceBarChart } from '../../components/charts/clinical/PrevalenceBarChart';
import { CooccurrenceHeatmap } from '../../components/charts/clinical/CooccurrenceHeatmap';

// AXE 3 : Fiabilité & Bruit
import { UncertaintyStackedBar } from '../../components/charts/uncertainty/UncertaintyStackedBar';
import { UncertaintyTreemap } from '../../components/charts/uncertainty/UncertaintyTreemap';
import { UncertaintyByViewType } from '../../components/charts/uncertainty/UncertaintyByViewType';

// AXE 4 : Métadonnées Techniques
import { ViewTypeDonut } from '../../components/charts/metadata/ViewTypeDonut';
import { PathologyVsDevices } from '../../components/charts/metadata/PathologyVsDevices';

/**
 * Composant interne du dashboard (doit être dans le FilterProvider)
 */
function DashboardContent() {
  const { 
    stats, 
    demographics, 
    // AXE 1
    agePyramid,
    ageBoxplot,
    patientFrequency,
    // AXE 2
    prevalence,
    cooccurrence,
    // AXE 3
    uncertaintyStacked,
    uncertaintyTreemap,
    uncertaintyByView,
    // AXE 4
    viewDistribution,
    pathologyDevices,
    // États
    loading, 
    error 
  } = useAnalytics();

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Chargement des données...</p>
          <p className="mt-2 text-sm text-gray-400">Calcul des 10 visualisations en cours...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded max-w-lg">
          <strong className="font-bold">Erreur: </strong>
          <span className="block sm:inline">{error}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-[1800px] mx-auto">
        {/* En-tête */}
        <header className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900">
            PSID - Medical Analytics Dashboard
          </h1>
          <p className="text-gray-600 mt-1">
            Analyse du dataset CheXpert - {stats?.total_rows?.toLocaleString() || '...'} radiographies
          </p>
        </header>

        {/* Barre de filtres globaux */}
        <FilterBar />

        {/* Statistiques résumées */}
        {(stats || demographics) && (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-xs text-gray-500 uppercase">Total Images</p>
              <p className="text-2xl font-bold text-gray-800">
                {stats?.total_rows?.toLocaleString() || '-'}
              </p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-xs text-gray-500 uppercase">Patients</p>
              <p className="text-2xl font-bold text-gray-800">
                {demographics?.total_patients?.toLocaleString() || '-'}
              </p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-xs text-gray-500 uppercase">Âge Moyen</p>
              <p className="text-2xl font-bold text-gray-800">
                {demographics?.average_age?.toFixed(1) || '-'} ans
              </p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-xs text-gray-500 uppercase">Hommes</p>
              <p className="text-2xl font-bold text-blue-600">
                {demographics?.sex_distribution?.Male?.toLocaleString() || '-'}
              </p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-xs text-gray-500 uppercase">Femmes</p>
              <p className="text-2xl font-bold text-pink-600">
                {demographics?.sex_distribution?.Female?.toLocaleString() || '-'}
              </p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-xs text-gray-500 uppercase">Pathologies</p>
              <p className="text-2xl font-bold text-gray-800">
                {stats?.pathology_columns?.length || 14}
              </p>
            </div>
          </div>
        )}

        {/* ============================================================= */}
        {/* AXE 1 : Démographie & Biais (Équité de l'IA) */}
        {/* ============================================================= */}
        <section className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-1 h-8 bg-blue-600 rounded"></div>
            <div>
              <h2 className="text-xl font-bold text-gray-800">
                AXE 1 : Démographie & Biais
              </h2>
              <p className="text-sm text-gray-500">Équité de l'IA - Analyse de la structure de population</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            <Card title="1. Pyramide des Âges" className="xl:col-span-1">
              <AgePyramid data={agePyramid} />
            </Card>
            <Card title="2. Boxplot Âge par Pathologie" className="xl:col-span-1">
              <AgeBoxplot data={ageBoxplot} />
            </Card>
            <Card title="3. Distribution Fréquence Patient" className="xl:col-span-1">
              <PatientFrequencyHistogram data={patientFrequency} />
            </Card>
          </div>
        </section>

        {/* ============================================================= */}
        {/* AXE 2 : Panorama Clinique (Biologie & Corrélations) */}
        {/* ============================================================= */}
        <section className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-1 h-8 bg-green-600 rounded"></div>
            <div>
              <h2 className="text-xl font-bold text-gray-800">
                AXE 2 : Panorama Clinique
              </h2>
              <p className="text-sm text-gray-500">Biologie & Corrélations entre pathologies</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <Card title="4. Prévalence des Pathologies">
              <PrevalenceBarChart data={prevalence} />
            </Card>
            <Card title="5. Heatmap de Co-occurrence">
              <CooccurrenceHeatmap data={cooccurrence} />
            </Card>
          </div>
        </section>

        {/* ============================================================= */}
        {/* AXE 3 : Fiabilité & Bruit (Analyse de l'Incertitude) */}
        {/* ============================================================= */}
        <section className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-1 h-8 bg-yellow-500 rounded"></div>
            <div>
              <h2 className="text-xl font-bold text-gray-800">
                AXE 3 : Fiabilité & Bruit
              </h2>
              <p className="text-sm text-gray-500">Analyse de l'incertitude dans les labels (-1)</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            <Card title="6. Positifs vs Incertains" className="xl:col-span-1">
              <UncertaintyStackedBar data={uncertaintyStacked} />
            </Card>
            <Card title="7. Treemap de l'Incertitude" className="xl:col-span-1">
              <UncertaintyTreemap data={uncertaintyTreemap} />
            </Card>
            <Card title="8. Incertitude par Type de Vue" className="xl:col-span-1">
              <UncertaintyByViewType data={uncertaintyByView} />
            </Card>
          </div>
        </section>

        {/* ============================================================= */}
        {/* AXE 4 : Métadonnées Techniques (Contexte de Capture) */}
        {/* ============================================================= */}
        <section className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-1 h-8 bg-purple-600 rounded"></div>
            <div>
              <h2 className="text-xl font-bold text-gray-800">
                AXE 4 : Métadonnées Techniques
              </h2>
              <p className="text-sm text-gray-500">Contexte de capture des radiographies</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <Card title="9. Distribution des Types de Vues">
              <ViewTypeDonut data={viewDistribution} />
            </Card>
            <Card title="10. Pathologies vs Support Devices">
              <PathologyVsDevices data={pathologyDevices} />
            </Card>
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center text-sm text-gray-500 py-6 border-t border-gray-200">
          <p>Dashboard Analytics - Dataset CheXpert</p>
          <p className="mt-1">Les filtres globaux s'appliquent à l'ensemble des 10 visualisations</p>
        </footer>
      </div>
    </div>
  );
}

/**
 * Dashboard principal avec FilterProvider et ErrorBoundary
 */
export function AnalyticsDashboard() {
  return (
    <ErrorBoundary>
      <FilterProvider>
        <DashboardContent />
      </FilterProvider>
    </ErrorBoundary>
  );
}
