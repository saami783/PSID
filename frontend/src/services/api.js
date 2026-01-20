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
  }
};
