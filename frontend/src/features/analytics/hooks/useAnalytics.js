/**
 * Hook personnalisé pour gérer la logique métier des analytics
 * Séparation de la logique métier du JSX
 */
import { useState, useEffect } from 'react';
import { analyticsAPI } from '../../../services/api';

export function useAnalytics() {
  const [stats, setStats] = useState(null);
  const [demographics, setDemographics] = useState(null);
  const [pathologies, setPathologies] = useState(null);
  const [correlation, setCorrelation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Charger les données avec gestion d'erreur robuste
        // Utiliser Promise.allSettled pour continuer même si une API échoue
        const results = await Promise.allSettled([
          analyticsAPI.getStats(),
          analyticsAPI.getDemographics(),
          analyticsAPI.getPathologies()
        ]);

        // Traiter chaque résultat
        if (results[0].status === 'fulfilled') {
          setStats(results[0].value.data);
        } else {
          console.error('Erreur lors du chargement des stats:', results[0].reason);
        }

        if (results[1].status === 'fulfilled') {
          setDemographics(results[1].value.data);
        } else {
          console.error('Erreur lors du chargement des demographics:', results[1].reason);
        }

        if (results[2].status === 'fulfilled') {
          setPathologies(results[2].value.data);
        } else {
          console.error('Erreur lors du chargement des pathologies:', results[2].reason);
        }

        // Ne pas charger la corrélation pour éviter les erreurs
        setCorrelation(null);
      } catch (err) {
        // Ne définir l'erreur que si toutes les APIs ont échoué
        const hasAnyData = stats || demographics || pathologies;
        if (!hasAnyData) {
          setError(err.message || 'Erreur lors du chargement des données');
        }
        console.error('Erreur lors du chargement des données:', err);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  return {
    stats,
    demographics,
    pathologies,
    correlation,
    loading,
    error
  };
}
