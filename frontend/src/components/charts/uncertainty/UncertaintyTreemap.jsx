/**
 * Treemap de l'Incertitude
 * Visualisation de la répartition du volume de "bruit" par pathologie
 */
import { Treemap, ResponsiveContainer, Tooltip } from 'recharts';

// Composant personnalisé pour le contenu des cellules du treemap
const CustomizedContent = (props) => {
  const { x, y, width, height, name, value, percent, colors, index } = props;
  
  // Protection contre les valeurs undefined
  if (x === undefined || y === undefined || width === undefined || height === undefined) {
    return null;
  }
  
  // Valeurs par défaut sécurisées
  const safeName = name || '';
  const safeValue = value || 0;
  const safePercent = percent || 0;
  const safeColors = colors || ['#6b7280'];
  const safeIndex = index || 0;
  
  // Ne pas afficher de texte si la cellule est trop petite
  const showText = width > 60 && height > 40;
  const showPercent = width > 40 && height > 25;
  
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        style={{
          fill: safeColors[safeIndex % safeColors.length],
          stroke: '#fff',
          strokeWidth: 2,
          strokeOpacity: 1,
        }}
      />
      {showText && safeName && (
        <>
          <text
            x={x + width / 2}
            y={y + height / 2 - 8}
            textAnchor="middle"
            fill="#fff"
            fontSize={11}
            fontWeight="bold"
          >
            {safeName.length > 12 ? safeName.substring(0, 10) + '...' : safeName}
          </text>
          <text
            x={x + width / 2}
            y={y + height / 2 + 8}
            textAnchor="middle"
            fill="#fff"
            fontSize={10}
          >
            {safeValue >= 1000 ? `${(safeValue/1000).toFixed(1)}k` : safeValue}
          </text>
        </>
      )}
      {!showText && showPercent && (
        <text
          x={x + width / 2}
          y={y + height / 2 + 4}
          textAnchor="middle"
          fill="#fff"
          fontSize={9}
          fontWeight="bold"
        >
          {safePercent}%
        </text>
      )}
    </g>
  );
};

export function UncertaintyTreemap({ data }) {
  if (!data || !Array.isArray(data) || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Aucune donnée disponible
      </div>
    );
  }

  // Palette de couleurs (du jaune-orange au rouge-violet pour représenter le "bruit")
  const COLORS = [
    '#f59e0b', '#f97316', '#ef4444', '#dc2626', '#b91c1c',
    '#9333ea', '#7c3aed', '#6366f1', '#4f46e5', '#4338ca',
    '#3730a3', '#312e81', '#1e1b4b', '#0f172a'
  ];

  // Préparer les données pour le treemap
  const treemapData = data
    .filter(item => item.value > 0)
    .map((item, index) => ({
      name: item.name,
      size: item.value,
      value: item.value,
      percent: item.percent,
      colors: COLORS,
      index: index
    }));

  const totalUncertain = data.reduce((sum, item) => sum + item.value, 0);

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="bg-white p-3 border rounded shadow-lg">
          <p className="font-semibold text-gray-800 mb-2">{item.name}</p>
          <p className="text-yellow-600">
            Labels incertains : <strong>{item.value.toLocaleString()}</strong>
          </p>
          <p className="text-gray-600">
            Pourcentage du bruit : <strong>{item.percent}%</strong>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={400}>
        <Treemap
          data={treemapData}
          dataKey="size"
          aspectRatio={4 / 3}
          stroke="#fff"
          fill="#8884d8"
          content={<CustomizedContent colors={COLORS} />}
        >
          <Tooltip content={<CustomTooltip />} />
        </Treemap>
      </ResponsiveContainer>

      {/* Statistiques */}
      <div className="mt-4 bg-yellow-50 p-4 rounded">
        <div className="text-center">
          <p className="text-sm text-yellow-700">Total Labels Incertains (-1)</p>
          <p className="text-3xl font-bold text-yellow-800">
            {totalUncertain.toLocaleString()}
          </p>
        </div>
      </div>

      {/* Liste des pathologies les plus bruitées */}
      <div className="mt-4">
        <h4 className="text-sm font-semibold text-gray-700 mb-2">
          Répartition du bruit
        </h4>
        <div className="grid grid-cols-2 gap-2">
          {data.slice(0, 6).map((item, index) => (
            <div 
              key={item.name}
              className="flex items-center gap-2 p-2 rounded"
              style={{ backgroundColor: `${COLORS[index]}15` }}
            >
              <div 
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: COLORS[index] }}
              />
              <span className="text-xs flex-1 truncate" title={item.name}>
                {item.name}
              </span>
              <span className="text-xs font-semibold" style={{ color: COLORS[index] }}>
                {item.percent}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
