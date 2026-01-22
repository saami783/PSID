/**
 * Composant d'analyse pour les graphiques
 * Affiche une bulle d'analyse stylisée sous chaque graphique
 * Style discret avec fond gris clair et bordure bleue
 */
export function GraphAnalysis({ children }) {
  return (
    <div className="mt-4 p-4 bg-gray-50 border-l-4 border-blue-400 rounded-r">
      <p className="text-xs text-gray-700 leading-relaxed">
        {children}
      </p>
    </div>
  );
}
