/**
 * GlobalFilterBar - Barre de contrôles unique (Architecture UX Stricte)
 * Permet de filtrer tous les graphiques par sexe, type de vue et pathologie cible
 * Aucune interaction de filtrage ne doit se faire en cliquant sur les graphiques
 */
import { useFilters } from '../../context/FilterContext';

// Liste des pathologies disponibles pour l'analyse conditionnelle
const PATHOLOGY_OPTIONS = [
  'No Finding',
  'Enlarged Cardiomediastinum',
  'Cardiomegaly',
  'Lung Opacity',
  'Lung Lesion',
  'Edema',
  'Consolidation',
  'Pneumonia',
  'Atelectasis',
  'Pneumothorax',
  'Pleural Effusion',
  'Pleural Other',
  'Fracture',
  'Support Devices'
];

export function FilterBar() {
  const { filters, updateFilter, resetFilters } = useFilters();

  const hasActiveFilters = filters.sex || filters.viewType || filters.targetPathology;

  return (
    <div className="sticky top-0 z-50 bg-white rounded-lg shadow-lg p-4 mb-6 border-2 border-indigo-200">
      <div className="flex flex-wrap items-center gap-4">
        {/* Titre */}
        <div className="flex items-center gap-2">
          <svg 
            className="w-5 h-5 text-indigo-600" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" 
            />
          </svg>
          <span className="font-bold text-gray-800">Contrôles globaux</span>
          <span className="text-xs text-gray-500 italic">(interface déterministe)</span>
        </div>

        {/* Séparateur */}
        <div className="hidden md:block w-px h-8 bg-gray-300"></div>

        {/* Filtre par Sexe */}
        <div className="flex items-center gap-2">
          <label htmlFor="sex-filter" className="text-sm font-medium text-gray-700">
            Sexe:
          </label>
          <select
            id="sex-filter"
            value={filters.sex || 'all'}
            onChange={(e) => updateFilter('sex', e.target.value)}
            className="border-2 border-gray-300 rounded-md px-3 py-2 text-sm bg-white 
                       focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500
                       hover:border-gray-400 transition-colors cursor-pointer"
          >
            <option value="all">Tous</option>
            <option value="Male">Homme</option>
            <option value="Female">Femme</option>
          </select>
        </div>

        {/* Filtre par Type de Vue */}
        <div className="flex items-center gap-2">
          <label htmlFor="view-filter" className="text-sm font-medium text-gray-700">
            Vue:
          </label>
          <select
            id="view-filter"
            value={filters.viewType || 'all'}
            onChange={(e) => updateFilter('viewType', e.target.value)}
            className="border-2 border-gray-300 rounded-md px-3 py-2 text-sm bg-white 
                       focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500
                       hover:border-gray-400 transition-colors cursor-pointer"
          >
            <option value="all">Toutes</option>
            <optgroup label="Orientation">
              <option value="Frontal">Frontal</option>
              <option value="Lateral">Latéral</option>
            </optgroup>
            <optgroup label="Position">
              <option value="AP">AP</option>
              <option value="PA">PA</option>
            </optgroup>
          </select>
        </div>

        {/* Filtre Focus Pathologie */}
        <div className="flex items-center gap-2 border-l-2 border-green-300 pl-4">
          <label htmlFor="pathology-filter" className="text-sm font-medium text-green-700">
            🔍 Focus pathologie :
          </label>
          <select
            id="pathology-filter"
            value={filters.targetPathology || 'all'}
            onChange={(e) => updateFilter('targetPathology', e.target.value)}
            className="border-2 border-green-300 rounded-md px-3 py-2 text-sm bg-green-50 
                       focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500
                       hover:border-green-400 transition-colors cursor-pointer font-medium"
            title="Pilote l'analyse des comorbidités conditionnelles"
          >
            <option value="all">-- Sélectionner une pathologie --</option>
            {PATHOLOGY_OPTIONS.map((pathology) => (
              <option key={pathology} value={pathology}>
                {pathology}
              </option>
            ))}
          </select>
        </div>

        {/* Bouton Réinitialiser */}
        {hasActiveFilters && (
          <button
            onClick={resetFilters}
            className="flex items-center gap-1 px-3 py-2 text-sm font-medium text-red-600 
                       hover:text-red-700 hover:bg-red-50 rounded-md transition-colors
                       border border-red-300 hover:border-red-400"
          >
            <svg 
              className="w-4 h-4" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M6 18L18 6M6 6l12 12" 
              />
            </svg>
            Réinitialiser
          </button>
        )}

        {/* Indicateur de filtres actifs */}
        {hasActiveFilters && (
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <span className="text-xs text-gray-500 font-medium">Filtres actifs:</span>
            {filters.sex && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs 
                             font-semibold bg-blue-100 text-blue-800 border border-blue-300">
                {filters.sex === 'Male' ? '👨 Homme' : '👩 Femme'}
              </span>
            )}
            {filters.viewType && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs 
                             font-semibold bg-purple-100 text-purple-800 border border-purple-300">
                📸 {filters.viewType}
              </span>
            )}
            {filters.targetPathology && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs 
                             font-semibold bg-green-100 text-green-800 border border-green-300">
                🎯 {filters.targetPathology}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Note explicative */}
      <div className="mt-2 pt-2 border-t border-gray-200">
        <p className="text-xs text-gray-500 italic">
          ℹ️ <strong>Philosophie UX :</strong> Tous les graphiques écoutent ces contrôles. 
          Aucune interaction de filtrage ne se fait en cliquant sur les graphiques (lecture seule).
        </p>
      </div>
    </div>
  );
}
