/**
 * Bar Chart de Prévalence des pathologies
 * Fréquence des 14 observations, triées de manière décroissante
 */
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell,
  LabelList
} from 'recharts';

export function PrevalenceBarChart({ data }) {
  if (!data || !Array.isArray(data) || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Aucune donnée disponible
      </div>
    );
  }

  // Palette de couleurs dégradée
  const COLORS = [
    '#1e40af', '#1d4ed8', '#2563eb', '#3b82f6', '#60a5fa',
    '#93c5fd', '#10b981', '#34d399', '#6ee7b7', '#fbbf24',
    '#f59e0b', '#f97316', '#ef4444', '#dc2626'
  ];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="bg-white p-3 border rounded shadow-lg">
          <p className="font-semibold text-gray-800 mb-2">{item.pathology}</p>
          <div className="text-sm space-y-1">
            <p>
              <span className="text-green-600">Positifs:</span>{' '}
              <strong>{item.positive_count.toLocaleString()}</strong>
            </p>
            <p>
              <span className="text-gray-600">Négatifs:</span>{' '}
              {item.negative_count.toLocaleString()}
            </p>
            <p className="pt-1 border-t">
              <span className="text-blue-600">Prévalence:</span>{' '}
              <strong>{item.prevalence_percent}%</strong>
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  // Formateur personnalisé pour les labels
  const renderCustomLabel = (props) => {
    const { x, y, width, value } = props;
    if (value > 10000) {
      return (
        <text 
          x={x + width + 5} 
          y={y + 12} 
          fill="#666" 
          fontSize={10}
        >
          {(value / 1000).toFixed(0)}k
        </text>
      );
    }
    return null;
  };

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={450}>
        <BarChart 
          data={data} 
          layout="vertical"
          margin={{ top: 20, right: 60, left: 120, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
          <XAxis 
            type="number"
            tickFormatter={(value) => value >= 1000 ? `${(value/1000).toFixed(0)}k` : value}
          />
          <YAxis 
            type="category" 
            dataKey="pathology" 
            width={110}
            tick={{ fontSize: 11 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar 
            dataKey="positive_count" 
            name="Cas positifs"
            radius={[0, 4, 4, 0]}
          >
            {data.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={COLORS[index % COLORS.length]} 
              />
            ))}
            <LabelList 
              dataKey="prevalence_percent" 
              position="right" 
              formatter={(value) => `${value}%`}
              style={{ fontSize: 10, fill: '#666' }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Top 5 pathologies */}
      <div className="mt-4">
        <h4 className="text-sm font-semibold text-gray-700 mb-2">Top 5 Pathologies</h4>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-2">
          {data.slice(0, 5).map((item, index) => (
            <div 
              key={item.pathology}
              className="p-2 rounded text-center"
              style={{ backgroundColor: `${COLORS[index]}20` }}
            >
              <p className="text-xs text-gray-600 truncate" title={item.pathology}>
                {item.pathology}
              </p>
              <p className="text-lg font-bold" style={{ color: COLORS[index] }}>
                {item.prevalence_percent}%
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
