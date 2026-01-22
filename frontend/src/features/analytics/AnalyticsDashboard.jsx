/**
 * Dashboard principal pour les analytics
 * 12 visualisations interactives organisées en 4 axes thématiques
 */
import { Link } from 'react-router-dom';
import { useAnalytics } from './hooks/useAnalytics';
import { FilterProvider } from '../../context/FilterContext';
import { Card } from '../../components/ui/Card';
import { FilterBar } from '../../components/ui/FilterBar';
import { ErrorBoundary } from '../../components/ui/ErrorBoundary';
import { GraphAnalysis } from '../../components/analytics/GraphAnalysis';

// AXE 1 : Démographie & Biais
import { AgePyramid } from '../../components/charts/demographics/AgePyramid';
import { AgeBoxplot } from '../../components/charts/demographics/AgeBoxplot';
import { PatientFrequencyHistogram } from '../../components/charts/demographics/PatientFrequencyHistogram';

// AXE 2 : Panorama Clinique
import { PrevalenceBarChart } from '../../components/charts/clinical/PrevalenceBarChart';
import { CooccurrenceHeatmap } from '../../components/charts/clinical/CooccurrenceHeatmap';
import { MultiPathologiesHistogram } from '../../components/charts/clinical/MultiPathologiesHistogram';
import { ConditionalProbabilitiesChart } from '../../components/charts/clinical/ConditionalProbabilitiesChart';

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
    multiPathologies,
    conditionalProbs,
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
          <p className="mt-2 text-sm text-gray-400">Calcul des 12 visualisations en cours...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded max-w-lg">
          <strong className="font-bold">Erreur : </strong>
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
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <Link 
                  to="/" 
                  className="flex items-center gap-2 text-blue-600 hover:text-blue-700 transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                  </svg>
                  <span className="text-sm font-medium">Retour à l'accueil</span>
                </Link>
              </div>
              <h1 className="text-3xl font-bold text-gray-900">
                DeepChex - Dashboard d'analyse médicale
              </h1>
              <p className="text-gray-600 mt-1">
                Analyse du dataset CheXpert - {stats?.total_rows?.toLocaleString() || '...'} radiographies | 12 visualisations
              </p>
            </div>
          </div>
        </header>

        {/* Barre de filtres globaux */}
        <FilterBar />

        {/* Statistiques résumées */}
        {(stats || demographics) && (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-xs text-gray-500">Total images</p>
              <p className="text-2xl font-bold text-gray-800">
                {stats?.total_rows?.toLocaleString() || '-'}
              </p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-xs text-gray-500">Patients</p>
              <p className="text-2xl font-bold text-gray-800">
                {demographics?.total_patients?.toLocaleString() || '-'}
              </p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-xs text-gray-500">Âge moyen</p>
              <p className="text-2xl font-bold text-gray-800">
                {demographics?.average_age?.toFixed(1) || '-'} ans
              </p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-xs text-gray-500">Hommes</p>
              <p className="text-2xl font-bold text-blue-600">
                {demographics?.sex_distribution?.Male?.toLocaleString() || '-'}
              </p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-xs text-gray-500">Femmes</p>
              <p className="text-2xl font-bold text-pink-600">
                {demographics?.sex_distribution?.Female?.toLocaleString() || '-'}
              </p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-xs text-gray-500">Pathologies</p>
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
                Axe 1 : Démographie & Biais
              </h2>
              <p className="text-sm text-gray-600 leading-relaxed">
                Cet axe étudie la structure de la population par âge et par sexe pour vérifier que le dataset est équilibré et ne contient pas de déséquilibres cachés qui pourraient fausser les prédictions futures.
              </p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            <Card title="1. Pyramide des âges" className="xl:col-span-1">
              <AgePyramid data={agePyramid} />
              <GraphAnalysis>
                On observe une répartition très inégale avec une concentration massive de patients âgés de 50 à 80 ans, tandis que les jeunes de moins de 30 ans sont presque absents des données. Cette caractéristique montre que la base de données est spécialisée en gériatrie, ce qui obligera l'IA à être particulièrement testée si elle doit être utilisée sur des patients plus jeunes dont l'anatomie pulmonaire est différente.
              </GraphAnalysis>
            </Card>
            <Card title="2. Boxplot âge par pathologie" className="xl:col-span-1">
              <AgeBoxplot data={ageBoxplot} />
              <GraphAnalysis>
                L'analyse des données révèle que les patients déclarés sains sont nettement plus jeunes que ceux présentant des maladies, avec un écart d'environ dix ans sur la moyenne d'âge. Ce décalage statistique est un point de vigilance critique car le modèle pourrait apprendre à associer la jeunesse des tissus à la santé au lieu de réellement chercher des signes cliniques, créant ainsi un biais de prédiction lié à l'âge.
              </GraphAnalysis>
            </Card>
            <Card title="3. Distribution fréquence patient" className="xl:col-span-1">
              <PatientFrequencyHistogram data={patientFrequency} />
              <GraphAnalysis>
                Bien que la majorité des patients n'ait qu'une seule image, une petite partie de la population possède un nombre très élevé de clichés allant jusqu'à 92 par personne. Ces patients chroniques pèsent lourdement dans le dataset et pourraient pousser l'IA à mémoriser leurs traits physiques particuliers au lieu d'apprendre des caractéristiques générales de maladies, ce qui nécessite une séparation rigoureuse des données par individu lors de l'entraînement.
              </GraphAnalysis>
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
                Axe 2 : Panorama Clinique
              </h2>
              <p className="text-sm text-gray-600 leading-relaxed">
                Cet axe cartographie la réalité médicale du dataset en analysant la fréquence des diagnostics et la manière dont les différentes maladies coexistent sur une même image.
              </p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <Card title="4. Prévalence des pathologies">
              <PrevalenceBarChart data={prevalence} />
              <GraphAnalysis>
                La distribution des diagnostics montre une domination des dispositifs médicaux et des opacités pulmonaires, alors que des maladies comme la pneumonie ne représentent que 2,7 % des cas totaux. Ce déséquilibre montre que nous travaillons sur une population hospitalière lourde et impose au futur modèle d'IA des techniques spécifiques pour ne pas ignorer les pathologies les moins fréquentes mais tout aussi critiques.
              </GraphAnalysis>
            </Card>
            <Card title="5. Heatmap de co-occurrence">
              <CooccurrenceHeatmap data={cooccurrence} />
              <GraphAnalysis>
                Les données mettent en évidence des liens statistiques très forts entre certaines observations, comme l'épanchement pleural qui apparaît très souvent aux côtés du matériel médical et des opacités. Le risque pour l'algorithme est de diagnostiquer une maladie par simple association statistique plutôt que par une analyse visuelle réelle des tissus, par exemple en prédisant un épanchement dès qu'il détecte un drain.
              </GraphAnalysis>
            </Card>
            <Card title="6. Multi-pathologies (sévérité)" className="xl:col-span-1">
              <MultiPathologiesHistogram data={multiPathologies} />
              <GraphAnalysis>
                La plupart des examens révèlent des cas complexes avec une moyenne de 2,2 pathologies par patient et des situations cumulant jusqu'à 8 diagnostics différents. Cette densité clinique signifie que l'IA devra être capable de démêler plusieurs signaux visuels qui se mélangent sur une seule radio, rendant la détection précise de chaque maladie beaucoup plus difficile qu'un diagnostic unique.
              </GraphAnalysis>
            </Card>
            <Card 
              title="7. Probabilités conditionnelles (comorbidités)" 
              className="xl:col-span-1"
            >
              <ConditionalProbabilitiesChart data={conditionalProbs} />
              <GraphAnalysis>
                L'analyse des probabilités conditionnelles révèle des associations médicales significatives entre certaines pathologies, où la présence d'une maladie augmente considérablement la probabilité d'en observer une autre. Ces patterns de comorbidités reflètent la réalité clinique complexe des patients hospitalisés et montrent que l'IA devra être entraînée à reconnaître ces associations naturelles tout en évitant de créer des liens artificiels qui ne seraient basés que sur des corrélations statistiques sans fondement médical réel.
              </GraphAnalysis>
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
                Axe 3 : Fiabilité & Bruit
              </h2>
              <p className="text-sm text-gray-600 leading-relaxed">
                Cet axe évalue le niveau de certitude des étiquettes de données pour identifier les zones de doute où l'interprétation médicale est difficile.
              </p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            <Card title="8. Positifs vs incertains" className="xl:col-span-1">
              <UncertaintyStackedBar data={uncertaintyStacked} />
              <GraphAnalysis>
                On constate qu'environ 21 % des diagnostics marqués comme pathologiques comportent une part d'incertitude, particulièrement pour des classes comme l'atélectasie ou la consolidation. Ce niveau de bruit dans les données d'origine est un défi majeur car si l'IA apprend sur des cas où même les experts hésitent, elle risque de produire des prédictions floues ou de manquer de confiance dans ses résultats.
              </GraphAnalysis>
            </Card>
            <Card title="9. Treemap de l'incertitude" className="xl:col-span-1">
              <UncertaintyTreemap data={uncertaintyTreemap} />
              <GraphAnalysis>
                Le doute médical se concentre massivement sur un petit groupe de pathologies qui partagent des caractéristiques visuelles très proches sur une radiographie. Cette observation montre une limite naturelle des données : certaines maladies sont physiquement difficiles à distinguer sans examens complémentaires, et l'IA rencontrera logiquement la même difficulté d'interprétation sur ces cas précis.
              </GraphAnalysis>
            </Card>
            <Card title="10. Incertitude par type de vue" className="xl:col-span-1">
              <UncertaintyByViewType data={uncertaintyByView} />
              <GraphAnalysis>
                Les statistiques montrent que le taux d'incertitude reste presque identique, autour de 16 %, que l'image soit prise de face ou de profil. Cela prouve que le doute est lié à la complexité de l'interprétation médicale elle-même et non à un simple manque de visibilité technique, ce qui signifie que l'ajout d'angles de vue différents ne suffira pas à éliminer totalement l'incertitude du modèle.
              </GraphAnalysis>
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
                Axe 4 : Métadonnées Techniques
              </h2>
              <p className="text-sm text-gray-600 leading-relaxed">
                Cet axe examine les conditions techniques de prise de vue et l'influence du matériel médical visible sur la qualité et le contexte des images.
              </p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <Card title="11. Distribution des types de vues">
              <ViewTypeDonut data={viewDistribution} />
              <GraphAnalysis>
                La très grande majorité des images (85 %) est prise en position AP, ce qui correspond à des patients souvent allongés ou en incapacité de se tenir debout. Cette prédominance technique oriente le modèle vers un usage en soins intensifs ou aux urgences, mais il devra faire l'objet d'une attention particulière s'il est utilisé pour des examens de routine sur des patients debout.
              </GraphAnalysis>
            </Card>
            <Card title="12. Pathologies vs support devices">
              <PathologyVsDevices data={pathologyDevices} />
              <GraphAnalysis>
                Les graphiques montrent une corrélation directe entre la présence d'objets métalliques comme des tubes ou des électrodes et la détection de maladies graves. Il existe un risque réel que l'IA apprenne à reconnaître ces objets (plus faciles à voir) pour poser son diagnostic au lieu d'analyser l'état réel des poumons, ce qui rendrait le modèle inefficace pour le dépistage précoce. On mesure que la présence de dispositifs médicaux augmente statistiquement de 14,6 % la fréquence de certaines pathologies comme l'épanchement pleural dans les rapports. Cette donnée confirme que le matériel médical agit comme un indice contextuel fort dans ce dataset, et l'entraînement devra s'assurer que l'IA reste capable de diagnostiquer une pathologie même sur un patient qui n'est pas encore appareillé.
              </GraphAnalysis>
            </Card>
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center text-sm text-gray-500 py-6 border-t border-gray-200">
          <p>Dashboard analytics - Dataset CheXpert</p>
          <p className="mt-1">Les filtres globaux s'appliquent à l'ensemble des 12 visualisations</p>
          <p className="mt-1 text-xs italic">Architecture UX déterministe - pas d'interaction de filtrage sur les graphiques</p>
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
