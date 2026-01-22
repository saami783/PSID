/**
 * Service API centralisé pour les appels au backend
 * Découple la logique réseau de la logique métier des composants
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

/**
 * Fonction utilitaire pour gérer les réponses API
 */
async function handleResponse(response) {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Erreur réseau' }));
    throw new Error(error.message || `HTTP error! status: ${response.status}`);
  }
  return response.json();
}

/**
 * Construit l'URL avec les query params de filtre
 */
function buildUrl(endpoint, queryParams = '') {
  const url = `${API_BASE_URL}/api/analytics/${endpoint}`;
  return queryParams ? `${url}?${queryParams}` : url;
}

/**
 * API pour les analytics du dataset CheXpert
 */
export const analyticsAPI = {
  /**
   * Récupère les statistiques générales du dataset
   */
  getStats: async () => {
    const response = await fetch(`${API_BASE_URL}/api/analytics/stats`);
    return handleResponse(response);
  },

  /**
   * Récupère les statistiques démographiques
   */
  getDemographics: async () => {
    const response = await fetch(`${API_BASE_URL}/api/analytics/demographics`);
    return handleResponse(response);
  },

  /**
   * Récupère les statistiques par pathologie
   */
  getPathologies: async () => {
    const response = await fetch(`${API_BASE_URL}/api/analytics/pathologies`);
    return handleResponse(response);
  },

  /**
   * Récupère la matrice de corrélation entre pathologies
   */
  getCorrelation: async () => {
    const response = await fetch(`${API_BASE_URL}/api/analytics/correlation`);
    return handleResponse(response);
  },

  /**
   * Vérifie l'état de santé de l'API
   */
  healthCheck: async () => {
    const response = await fetch(`${API_BASE_URL}/api/analytics/health`);
    return handleResponse(response);
  },

  // =========================================================================
  // AXE 1 : Démographie & Biais
  // =========================================================================

  /**
   * Récupère les données pour la pyramide des âges
   * @param {string} queryParams - Query params de filtre
   */
  getAgePyramid: async (queryParams = '') => {
    const response = await fetch(buildUrl('age-pyramid', queryParams));
    return handleResponse(response);
  },

  /**
   * Récupère les boxplots d'âge par pathologie
   * @param {string} queryParams - Query params de filtre
   */
  getAgeBoxplot: async (queryParams = '') => {
    const response = await fetch(buildUrl('age-boxplot', queryParams));
    return handleResponse(response);
  },

  /**
   * Récupère l'histogramme de fréquence patient
   * @param {string} queryParams - Query params de filtre
   */
  getPatientFrequency: async (queryParams = '') => {
    const response = await fetch(buildUrl('patient-frequency', queryParams));
    return handleResponse(response);
  },

  // =========================================================================
  // AXE 2 : Panorama Clinique
  // =========================================================================

  /**
   * Récupère le bar chart de prévalence des pathologies
   * @param {string} queryParams - Query params de filtre
   */
  getPrevalence: async (queryParams = '') => {
    const response = await fetch(buildUrl('prevalence', queryParams));
    return handleResponse(response);
  },

  /**
   * Récupère la heatmap de co-occurrence
   * @param {string} queryParams - Query params de filtre
   */
  getCooccurrence: async (queryParams = '') => {
    const response = await fetch(buildUrl('cooccurrence', queryParams));
    return handleResponse(response);
  },

  // =========================================================================
  // AXE 3 : Fiabilité & Bruit (Analyse de l'Incertitude)
  // =========================================================================

  /**
   * Récupère le stacked bar chart d'incertitude
   * @param {string} queryParams - Query params de filtre
   */
  getUncertaintyStacked: async (queryParams = '') => {
    const response = await fetch(buildUrl('uncertainty-stacked', queryParams));
    return handleResponse(response);
  },

  /**
   * Récupère les données du treemap d'incertitude
   * @param {string} queryParams - Query params de filtre
   */
  getUncertaintyTreemap: async (queryParams = '') => {
    const response = await fetch(buildUrl('uncertainty-treemap', queryParams));
    return handleResponse(response);
  },

  /**
   * Récupère l'analyse d'incertitude par type de vue
   * @param {string} queryParams - Query params de filtre
   */
  getUncertaintyByView: async (queryParams = '') => {
    const response = await fetch(buildUrl('uncertainty-by-view', queryParams));
    return handleResponse(response);
  },

  // =========================================================================
  // AXE 4 : Métadonnées Techniques
  // =========================================================================

  /**
   * Récupère la distribution des types de vues
   * @param {string} queryParams - Query params de filtre
   */
  getViewDistribution: async (queryParams = '') => {
    const response = await fetch(buildUrl('view-distribution', queryParams));
    return handleResponse(response);
  },

  /**
   * Récupère l'analyse pathologies vs support devices
   * @param {string} queryParams - Query params de filtre
   */
  getPathologyDevices: async (queryParams = '') => {
    const response = await fetch(buildUrl('pathology-devices', queryParams));
    return handleResponse(response);
  }
};
