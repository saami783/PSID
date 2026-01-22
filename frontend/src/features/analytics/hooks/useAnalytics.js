/**
 * Hook personnalisé pour gérer la logique métier des analytics
 * Séparation de la logique métier du JSX
 * Intègre les filtres globaux pour les 12 visualisations
 */
import { useState, useEffect, useCallback } from 'react';
import { analyticsAPI } from '../../../services/api';
import { useFilters } from '../../../context/FilterContext';

export function useAnalytics() {
  // États pour les données de base
  const [stats, setStats] = useState(null);
  const [demographics, setDemographics] = useState(null);
  const [pathologies, setPathologies] = useState(null);
  const [correlation, setCorrelation] = useState(null);
  
  // États pour les 12 visualisations (10 + 2 nouvelles pour Axe 2)
  const [agePyramid, setAgePyramid] = useState(null);
  const [ageBoxplot, setAgeBoxplot] = useState(null);
  const [patientFrequency, setPatientFrequency] = useState(null);
  const [prevalence, setPrevalence] = useState(null);
  const [cooccurrence, setCooccurrence] = useState(null);
    const [multiPathologies, setMultiPathologies] = useState(null);
    const [conditionalProbs, setConditionalProbs] = useState(null);
  const [uncertaintyStacked, setUncertaintyStacked] = useState(null);
  const [uncertaintyTreemap, setUncertaintyTreemap] = useState(null);
  const [uncertaintyByView, setUncertaintyByView] = useState(null);
  const [viewDistribution, setViewDistribution] = useState(null);
  const [pathologyDevices, setPathologyDevices] = useState(null);
  
  // États de chargement et erreur
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Récupérer les filtres du contexte
  const { filters, getQueryParams, getConditionalQueryParams, getAxe4QueryParams } = useFilters();

  /**
   * Charge les données de base (stats, demographics)
   */
  const loadBaseData = useCallback(async () => {
    const results = await Promise.allSettled([
      analyticsAPI.getStats(),
      analyticsAPI.getDemographics(),
      analyticsAPI.getPathologies()
    ]);

    if (results[0].status === 'fulfilled') {
      setStats(results[0].value.data);
    } else {
      console.error('Erreur stats:', results[0].reason);
    }

    if (results[1].status === 'fulfilled') {
      setDemographics(results[1].value.data);
    } else {
      console.error('Erreur demographics:', results[1].reason);
    }

    if (results[2].status === 'fulfilled') {
      setPathologies(results[2].value.data);
    } else {
      console.error('Erreur pathologies:', results[2].reason);
    }
  }, []);

  /**
   * Charge les données des 12 visualisations avec les filtres
   */
  const loadChartData = useCallback(async (queryParams, conditionalQueryParams, axe4QueryParams) => {
    const results = await Promise.allSettled([
      // AXE 1 : Démographie
      analyticsAPI.getAgePyramid(queryParams),
      analyticsAPI.getAgeBoxplot(queryParams),
      analyticsAPI.getPatientFrequency(queryParams),
      // AXE 2 : Clinique
      analyticsAPI.getPrevalence(queryParams),
      analyticsAPI.getCooccurrence(queryParams),
      analyticsAPI.getMultiPathologies(queryParams),
      analyticsAPI.getConditionalProbabilities(conditionalQueryParams),
      // AXE 3 : Incertitude
      analyticsAPI.getUncertaintyStacked(queryParams),
      analyticsAPI.getUncertaintyTreemap(queryParams),
      analyticsAPI.getUncertaintyByView(axe4QueryParams),
      // AXE 4 : Métadonnées
      analyticsAPI.getViewDistribution(axe4QueryParams),
      analyticsAPI.getPathologyDevices(queryParams)
    ]);

    // AXE 1
    if (results[0].status === 'fulfilled') {
      setAgePyramid(results[0].value.data);
    } else {
      console.error('Erreur agePyramid:', results[0].reason);
    }

    if (results[1].status === 'fulfilled') {
      setAgeBoxplot(results[1].value.data);
    } else {
      console.error('Erreur ageBoxplot:', results[1].reason);
    }

    if (results[2].status === 'fulfilled') {
      setPatientFrequency(results[2].value.data);
    } else {
      console.error('Erreur patientFrequency:', results[2].reason);
    }

    // AXE 2
    if (results[3].status === 'fulfilled') {
      setPrevalence(results[3].value.data);
    } else {
      console.error('Erreur prevalence:', results[3].reason);
    }

    if (results[4].status === 'fulfilled') {
      setCooccurrence(results[4].value.data);
    } else {
      console.error('Erreur cooccurrence:', results[4].reason);
    }

    if (results[5].status === 'fulfilled') {
      setMultiPathologies(results[5].value.data);
    } else {
      console.error('Erreur multiPathologies:', results[5].reason);
    }

    if (results[6].status === 'fulfilled') {
      setConditionalProbs(results[6].value.data);
    } else {
      console.error('Erreur conditionalProbs:', results[6].reason);
      // Ne pas afficher d'erreur si aucune pathologie n'est sélectionnée
      if (conditionalQueryParams && conditionalQueryParams.includes('target_disease')) {
        console.warn('Aucune pathologie cible sélectionnée pour les probabilités conditionnelles');
      }
    }

    // AXE 3
    if (results[7].status === 'fulfilled') {
      setUncertaintyStacked(results[7].value.data);
    } else {
      console.error('Erreur uncertaintyStacked:', results[7].reason);
    }

    if (results[8].status === 'fulfilled') {
      setUncertaintyTreemap(results[8].value.data);
    } else {
      console.error('Erreur uncertaintyTreemap:', results[8].reason);
    }

    if (results[9].status === 'fulfilled') {
      setUncertaintyByView(results[9].value.data);
    } else {
      console.error('Erreur uncertaintyByView:', results[9].reason);
    }

    // AXE 4
    if (results[10].status === 'fulfilled') {
      setViewDistribution(results[10].value.data);
    } else {
      console.error('Erreur viewDistribution:', results[10].reason);
    }

    if (results[11].status === 'fulfilled') {
      setPathologyDevices(results[11].value.data);
    } else {
      console.error('Erreur pathologyDevices:', results[11].reason);
    }
  }, []);

  // Effet pour charger les données initiales et lors du changement de filtres
  useEffect(() => {
    let isCancelled = false;
    
    const loadAllData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Construire les query params directement ici
        const params = new URLSearchParams();
        if (filters.sex) params.append('sex', filters.sex);
        if (filters.viewType) params.append('view_type', filters.viewType);
        const queryParams = params.toString();

        // Construire les query params pour les probabilités conditionnelles
        const conditionalParams = new URLSearchParams();
        if (filters.sex) conditionalParams.append('sex', filters.sex);
        if (filters.viewType) conditionalParams.append('view_type', filters.viewType);
        if (filters.targetPathology) conditionalParams.append('target_disease', filters.targetPathology);
        const conditionalQueryParams = conditionalParams.toString();

        // Construire les query params pour l'Axe 4 (inclut targetPathology)
        const axe4QueryParams = getAxe4QueryParams();

        // Charger les données en parallèle
        await Promise.all([
          loadBaseData(),
          loadChartData(queryParams, conditionalQueryParams, axe4QueryParams)
        ]);

        // Ne pas mettre à jour si le composant est démonté ou si les filtres ont changé
        if (!isCancelled) {
          setLoading(false);
        }
      } catch (err) {
        if (!isCancelled) {
          setError(err.message || 'Erreur lors du chargement des données');
          console.error('Erreur lors du chargement des données:', err);
          setLoading(false);
        }
      }
    };

    loadAllData();
    
    // Cleanup function pour éviter les mises à jour d'état sur composant démonté
    return () => {
      isCancelled = true;
    };
  }, [filters.sex, filters.viewType, filters.targetPathology, loadBaseData, loadChartData]);

  return {
    // Données de base
    stats,
    demographics,
    pathologies,
    correlation,
    
    // AXE 1 : Démographie & Biais
    agePyramid,
    ageBoxplot,
    patientFrequency,
    
    // AXE 2 : Panorama Clinique
    prevalence,
    cooccurrence,
    multiPathologies,
    conditionalProbs,
    
    // AXE 3 : Fiabilité & Bruit
    uncertaintyStacked,
    uncertaintyTreemap,
    uncertaintyByView,
    
    // AXE 4 : Métadonnées Techniques
    viewDistribution,
    pathologyDevices,
    
    // États
    loading,
    error
  };
}
