/**
 * Barre de filtres globaux pour le dashboard analytics
 * Permet de filtrer tous les graphiques par sexe et type de vue
 */
import { useFilters } from '../../context/FilterContext';

export function FilterBar() {
  const { filters, updateFilter, resetFilters } = useFilters();

  const hasActiveFilters = filters.sex || filters.viewType;

  return (
    <div className="bg-white rounded-lg shadow-md p-4 mb-6">
      <div className="flex flex-wrap items-center gap-4">
        {/* Titre */}
        <div className="flex items-center gap-2">
          <svg 
            className="w-5 h-5 text-gray-500" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" 
            />
          </svg>
          <span className="font-semibold text-gray-700">Filtres Globaux</span>
        </div>

        {/* Séparateur */}
        <div className="hidden sm:block w-px h-8 bg-gray-300"></div>

        {/* Filtre par Sexe */}
        <div className="flex items-center gap-2">
          <label htmlFor="sex-filter" className="text-sm text-gray-600">
            Sexe:
          </label>
          <select
            id="sex-filter"
            value={filters.sex || 'all'}
            onChange={(e) => updateFilter('sex', e.target.value)}
            className="border border-gray-300 rounded-md px-3 py-1.5 text-sm bg-white 
                       focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500
                       hover:border-gray-400 transition-colors"
          >
            <option value="all">Tous</option>
            <option value="Male">Homme</option>
            <option value="Female">Femme</option>
          </select>
        </div>

        {/* Filtre par Type de Vue */}
        <div className="flex items-center gap-2">
          <label htmlFor="view-filter" className="text-sm text-gray-600">
            Type de Vue:
          </label>
          <select
            id="view-filter"
            value={filters.viewType || 'all'}
            onChange={(e) => updateFilter('viewType', e.target.value)}
            className="border border-gray-300 rounded-md px-3 py-1.5 text-sm bg-white 
                       focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500
                       hover:border-gray-400 transition-colors"
          >
            <option value="all">Toutes</option>
            <optgroup label="Orientation">
              <option value="Frontal">Frontal</option>
              <option value="Lateral">Latéral</option>
            </optgroup>
            <optgroup label="Position">
              <option value="AP">AP (Antéro-Postérieur)</option>
              <option value="PA">PA (Postéro-Antérieur)</option>
            </optgroup>
          </select>
        </div>

        {/* Bouton Réinitialiser */}
        {hasActiveFilters && (
          <button
            onClick={resetFilters}
            className="flex items-center gap-1 px-3 py-1.5 text-sm text-red-600 
                       hover:text-red-700 hover:bg-red-50 rounded-md transition-colors"
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
          <div className="ml-auto flex items-center gap-2">
            <span className="text-xs text-gray-500">Filtres actifs:</span>
            {filters.sex && (
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs 
                             font-medium bg-blue-100 text-blue-800">
                {filters.sex === 'Male' ? 'Homme' : 'Femme'}
              </span>
            )}
            {filters.viewType && (
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs 
                             font-medium bg-green-100 text-green-800">
                {filters.viewType}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
