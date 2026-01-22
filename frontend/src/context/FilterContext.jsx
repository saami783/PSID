/**
 * Contexte global pour les filtres du dashboard analytics
 * Permet de synchroniser les filtres entre tous les graphiques
 */
import { createContext, useContext, useState, useCallback } from 'react';

// Création du contexte
const FilterContext = createContext(null);

/**
 * Provider pour les filtres globaux
 */
export function FilterProvider({ children }) {
  const [filters, setFilters] = useState({
    sex: null,        // 'Male' | 'Female' | null (tous)
    viewType: null    // 'Frontal' | 'Lateral' | 'AP' | 'PA' | null (tous)
  });

  /**
   * Met à jour un filtre spécifique
   */
  const updateFilter = useCallback((filterName, value) => {
    setFilters(prev => ({
      ...prev,
      [filterName]: value === 'all' ? null : value
    }));
  }, []);

  /**
   * Réinitialise tous les filtres
   */
  const resetFilters = useCallback(() => {
    setFilters({
      sex: null,
      viewType: null
    });
  }, []);

  /**
   * Construit les query params pour les appels API
   */
  const getQueryParams = useCallback(() => {
    const params = new URLSearchParams();
    if (filters.sex) params.append('sex', filters.sex);
    if (filters.viewType) params.append('view_type', filters.viewType);
    return params.toString();
  }, [filters]);

  const value = {
    filters,
    setFilters,
    updateFilter,
    resetFilters,
    getQueryParams
  };

  return (
    <FilterContext.Provider value={value}>
      {children}
    </FilterContext.Provider>
  );
}

/**
 * Hook personnalisé pour accéder aux filtres
 */
export function useFilters() {
  const context = useContext(FilterContext);
  if (!context) {
    throw new Error('useFilters doit être utilisé dans un FilterProvider');
  }
  return context;
}

export default FilterContext;
