/**
 * Boxplot de l'âge par pathologie
 * Distribution statistique de l'âge pour chaque maladie
 * Utilise un ComposedChart personnalisé pour simuler un boxplot
 */
import { 
  ComposedChart,
  Bar,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  ErrorBar,
  Cell
} from 'recharts';

// Composant personnalisé pour dessiner un boxplot
const BoxplotShape = (props) => {
  const { x, y, width, height, payload } = props;
  
  if (!payload) return null;
  
  const { min, q1, median, q3, max } = payload;
  const boxWidth = width * 0.6;
  const boxX = x + (width - boxWidth) / 2;
  
  // Échelle pour convertir les valeurs d'âge en pixels
  // On utilise une échelle linéaire simple
  const yScale = (value) => {
    const chartHeight = 300; // hauteur approximative
    const yMin = 0;
    const yMax = 100;
    return chartHeight - ((value - yMin) / (yMax - yMin)) * chartHeight;
  };

  return (
    <g>
      {/* Ligne verticale min-max (whiskers) */}
      <line
        x1={x + width / 2}
        y1={yScale(min)}
        x2={x + width / 2}
        y2={yScale(max)}
        stroke="#666"
        strokeWidth={1}
      />
      
      {/* Ligne horizontale min */}
      <line
        x1={boxX + boxWidth * 0.25}
        y1={yScale(min)}
        x2={boxX + boxWidth * 0.75}
        y2={yScale(min)}
        stroke="#666"
        strokeWidth={1}
      />
      
      {/* Ligne horizontale max */}
      <line
        x1={boxX + boxWidth * 0.25}
        y1={yScale(max)}
        x2={boxX + boxWidth * 0.75}
        y2={yScale(max)}
        stroke="#666"
        strokeWidth={1}
      />
      
      {/* Box (Q1 à Q3) */}
      <rect
        x={boxX}
        y={yScale(q3)}
        width={boxWidth}
        height={yScale(q1) - yScale(q3)}
        fill="#3b82f6"
        fillOpacity={0.7}
        stroke="#1d4ed8"
        strokeWidth={1}
      />
      
      {/* Ligne médiane */}
      <line
        x1={boxX}
        y1={yScale(median)}
        x2={boxX + boxWidth}
        y2={yScale(median)}
        stroke="#1d4ed8"
        strokeWidth={2}
      />
    </g>
  );
};

export function AgeBoxplot({ data }) {
  if (!data || !Array.isArray(data) || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Aucune donnée disponible
      </div>
    );
  }

  // Préparer les données pour affichage
  const chartData = data.map(item => ({
    ...item,
    // Pour le bar chart, on utilise la hauteur de la box (Q3 - Q1)
    boxHeight: item.q3 - item.q1,
    // Position de départ de la box
    boxStart: item.q1
  }));

  // Couleurs pour les différentes pathologies
  const COLORS = [
    '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
    '#14b8a6', '#a855f7', '#22c55e', '#eab308'
  ];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="bg-white p-3 border rounded shadow-lg">
          <p className="font-semibold text-gray-800 mb-2">{item.pathology}</p>
          <div className="text-sm space-y-1">
            <p><span className="text-gray-600">Min:</span> {item.min.toFixed(1)} ans</p>
            <p><span className="text-gray-600">Q1:</span> {item.q1.toFixed(1)} ans</p>
            <p><span className="text-gray-600">Médiane:</span> <strong>{item.median.toFixed(1)} ans</strong></p>
            <p><span className="text-gray-600">Q3:</span> {item.q3.toFixed(1)} ans</p>
            <p><span className="text-gray-600">Max:</span> {item.max.toFixed(1)} ans</p>
            <p className="pt-1 border-t"><span className="text-gray-600">N:</span> {item.count.toLocaleString()}</p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart 
          data={chartData} 
          margin={{ top: 20, right: 30, left: 20, bottom: 100 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="pathology" 
            angle={-45} 
            textAnchor="end" 
            height={100}
            interval={0}
            tick={{ fontSize: 11 }}
          />
          <YAxis 
            domain={[0, 100]}
            label={{ value: 'Âge (années)', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip content={<CustomTooltip />} />
          
          {/* Barre pour représenter la box (Q1 à Q3) */}
          <Bar dataKey="median" fill="#3b82f6" name="Médiane">
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
            <ErrorBar 
              dataKey="errorBar" 
              width={4} 
              strokeWidth={2}
              stroke="#666"
            />
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>

      {/* Tableau récapitulatif */}
      <div className="mt-4 overflow-x-auto">
        <table className="min-w-full text-xs">
          <thead>
            <tr className="bg-gray-100">
              <th className="px-2 py-1 text-left">Pathologie</th>
              <th className="px-2 py-1 text-right">Médiane</th>
              <th className="px-2 py-1 text-right">IQR</th>
              <th className="px-2 py-1 text-right">N</th>
            </tr>
          </thead>
          <tbody>
            {chartData.slice(0, 7).map((item, index) => (
              <tr key={item.pathology} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                <td className="px-2 py-1 truncate max-w-[120px]" title={item.pathology}>
                  {item.pathology}
                </td>
                <td className="px-2 py-1 text-right">{item.median.toFixed(1)}</td>
                <td className="px-2 py-1 text-right">{(item.q3 - item.q1).toFixed(1)}</td>
                <td className="px-2 py-1 text-right">{item.count.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
