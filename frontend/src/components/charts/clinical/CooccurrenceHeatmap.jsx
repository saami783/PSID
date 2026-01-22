/**
 * Heatmap de Co-occurrence des pathologies
 * Matrice montrant les liens entre les pathologies
 */
import { useState } from 'react';

export function CooccurrenceHeatmap({ data }) {
  const [hoveredCell, setHoveredCell] = useState(null);

  if (!data || !Array.isArray(data.matrix) || !Array.isArray(data.pathologies) || data.pathologies.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Aucune donnée disponible
      </div>
    );
  }

  const { matrix, pathologies, max_value } = data;
  
  // Protection supplémentaire
  if (matrix.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Aucune donnée disponible
      </div>
    );
  }

  // Fonction pour obtenir la couleur basée sur la valeur
  const getColor = (value) => {
    if (value === 0) return '#f3f4f6';
    
    const ratio = value / max_value;
    
    // Palette de bleu-violet
    if (ratio < 0.1) return '#dbeafe';
    if (ratio < 0.2) return '#bfdbfe';
    if (ratio < 0.3) return '#93c5fd';
    if (ratio < 0.4) return '#60a5fa';
    if (ratio < 0.5) return '#3b82f6';
    if (ratio < 0.6) return '#2563eb';
    if (ratio < 0.7) return '#1d4ed8';
    if (ratio < 0.8) return '#1e40af';
    if (ratio < 0.9) return '#7c3aed';
    return '#5b21b6';
  };

  // Raccourcir les noms de pathologies pour l'affichage
  const shortenName = (name) => {
    const shortNames = {
      'No Finding': 'No Find',
      'Enlarged Cardiomediastinum': 'Enl Card',
      'Cardiomegaly': 'Cardiom',
      'Lung Opacity': 'Lung Op',
      'Lung Lesion': 'Lung Les',
      'Edema': 'Edema',
      'Consolidation': 'Consol',
      'Pneumonia': 'Pneum',
      'Atelectasis': 'Atelect',
      'Pneumothorax': 'Pneumot',
      'Pleural Effusion': 'Pl Eff',
      'Pleural Other': 'Pl Other',
      'Fracture': 'Fract',
      'Support Devices': 'Supp Dev'
    };
    return shortNames[name] || name.substring(0, 8);
  };

  const cellSize = 35;
  const labelWidth = 70;

  return (
    <div className="w-full overflow-x-auto">
      {/* Tooltip */}
      {hoveredCell && (
        <div 
          className="fixed z-50 bg-white p-3 border rounded shadow-lg pointer-events-none"
          style={{
            left: hoveredCell.x + 10,
            top: hoveredCell.y + 10,
          }}
        >
          <p className="font-semibold text-gray-800 text-sm">
            {pathologies[hoveredCell.row]} + {pathologies[hoveredCell.col]}
          </p>
          <p className="text-gray-600">
            Co-occurrence: <strong>{hoveredCell.value.toLocaleString()}</strong>
          </p>
        </div>
      )}

      <div className="inline-block">
        {/* En-tête avec les noms des colonnes */}
        <div className="flex" style={{ marginLeft: labelWidth }}>
          {pathologies.map((name, index) => (
            <div 
              key={`header-${index}`}
              className="text-xs text-gray-600 font-medium text-center overflow-hidden"
              style={{ 
                width: cellSize, 
                transform: 'rotate(-45deg)',
                transformOrigin: 'left bottom',
                height: 60,
                whiteSpace: 'nowrap'
              }}
              title={name}
            >
              {shortenName(name)}
            </div>
          ))}
        </div>

        {/* Matrice */}
        <div className="mt-8">
          {matrix.map((row, rowIndex) => (
            <div key={`row-${rowIndex}`} className="flex items-center">
              {/* Label de la ligne */}
              <div 
                className="text-xs text-gray-600 font-medium text-right pr-2 truncate"
                style={{ width: labelWidth }}
                title={pathologies[rowIndex]}
              >
                {shortenName(pathologies[rowIndex])}
              </div>
              
              {/* Cellules */}
              {row.map((value, colIndex) => (
                <div
                  key={`cell-${rowIndex}-${colIndex}`}
                  className="border border-white cursor-pointer transition-transform hover:scale-110 hover:z-10"
                  style={{
                    width: cellSize,
                    height: cellSize,
                    backgroundColor: getColor(value),
                  }}
                  onMouseEnter={(e) => setHoveredCell({
                    row: rowIndex,
                    col: colIndex,
                    value: value,
                    x: e.clientX,
                    y: e.clientY
                  })}
                  onMouseLeave={() => setHoveredCell(null)}
                >
                  {/* Afficher la valeur si la cellule est assez grande */}
                  {rowIndex === colIndex && (
                    <div className="w-full h-full flex items-center justify-center text-[8px] font-bold text-white">
                      {value >= 1000 ? `${(value/1000).toFixed(0)}k` : value}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Légende */}
      <div className="mt-6 flex items-center justify-center gap-2">
        <span className="text-xs text-gray-600">Faible</span>
        <div className="flex">
          {['#f3f4f6', '#dbeafe', '#93c5fd', '#3b82f6', '#1d4ed8', '#5b21b6'].map((color, i) => (
            <div 
              key={i}
              className="w-6 h-4"
              style={{ backgroundColor: color }}
            />
          ))}
        </div>
        <span className="text-xs text-gray-600">Élevée</span>
      </div>

      {/* Info */}
      <p className="mt-2 text-xs text-gray-500 text-center">
        La diagonale représente le nombre de cas positifs pour chaque pathologie.
        Les autres cellules montrent le nombre de cas où les deux pathologies sont présentes simultanément.
      </p>
    </div>
  );
}
