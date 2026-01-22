/**
 * Analyse de l'Incertitude par ViewType
 * Est-ce que les vues AP génèrent plus de labels '-1' que les PA ?
 */
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  Cell
} from 'recharts';

export function UncertaintyByViewType({ data }) {
  const hasFrontalLateral = data?.by_frontal_lateral && Array.isArray(data.by_frontal_lateral) && data.by_frontal_lateral.length > 0;
  const hasApPa = data?.by_ap_pa && Array.isArray(data.by_ap_pa) && data.by_ap_pa.length > 0;
  
  if (!data || (!hasFrontalLateral && !hasApPa)) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Aucune donnée disponible
      </div>
    );
  }

  const COLORS = {
    'Frontal': '#3b82f6',
    'Lateral': '#8b5cf6',
    'AP': '#10b981',
    'PA': '#f59e0b',
    'LL': '#ec4899'
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="bg-white p-3 border rounded shadow-lg">
          <p className="font-semibold text-gray-800 mb-2">Vue {item.view_type}</p>
          <div className="text-sm space-y-1">
            <p className="text-yellow-600">
              Incertains (-1): <strong>{item.uncertain_count.toLocaleString()}</strong> ({item.uncertain_percent}%)
            </p>
            <p className="text-green-600">
              Positifs (1): <strong>{item.positive_count.toLocaleString()}</strong>
            </p>
            <p className="text-gray-600">
              Négatifs (0): <strong>{item.negative_count.toLocaleString()}</strong>
            </p>
            <p className="pt-1 border-t text-gray-500">
              Total labels: {item.total_labels.toLocaleString()}
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full">
      {/* Graphique Frontal/Lateral */}
      {hasFrontalLateral && (
        <div className="mb-8">
          <h4 className="text-sm font-semibold text-gray-700 mb-3">
            Incertitude par Orientation (Frontal vs Latéral)
          </h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart 
              data={data.by_frontal_lateral} 
              margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="view_type" />
              <YAxis 
                yAxisId="left"
                tickFormatter={(value) => value >= 1000 ? `${(value/1000).toFixed(0)}k` : value}
              />
              <YAxis 
                yAxisId="right" 
                orientation="right"
                tickFormatter={(value) => `${value}%`}
                domain={[0, 'auto']}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Bar 
                yAxisId="left"
                dataKey="uncertain_count" 
                name="Labels incertains" 
                fill="#f59e0b"
              >
                {data.by_frontal_lateral.map((entry, index) => (
                  <Cell key={index} fill={COLORS[entry.view_type] || '#6b7280'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          
          {/* Comparaison en pourcentage */}
          <div className="grid grid-cols-2 gap-4 mt-3">
            {data.by_frontal_lateral.map(item => (
              <div 
                key={item.view_type}
                className="p-3 rounded"
                style={{ backgroundColor: `${COLORS[item.view_type]}15` }}
              >
                <p className="text-sm font-medium" style={{ color: COLORS[item.view_type] }}>
                  {item.view_type}
                </p>
                <p className="text-2xl font-bold text-gray-800">
                  {item.uncertain_percent}%
                </p>
                <p className="text-xs text-gray-500">
                  d'incertitude
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Graphique AP/PA */}
      {hasApPa && (
        <div>
          <h4 className="text-sm font-semibold text-gray-700 mb-3">
            Incertitude par Position (AP vs PA)
          </h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart 
              data={data.by_ap_pa} 
              margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="view_type" />
              <YAxis 
                tickFormatter={(value) => value >= 1000 ? `${(value/1000).toFixed(0)}k` : value}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Bar 
                dataKey="uncertain_count" 
                name="Labels incertains"
              >
                {data.by_ap_pa.map((entry, index) => (
                  <Cell key={index} fill={COLORS[entry.view_type] || '#6b7280'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {/* Comparaison détaillée */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-3">
            {data.by_ap_pa.map(item => (
              <div 
                key={item.view_type}
                className="p-2 rounded text-center"
                style={{ backgroundColor: `${COLORS[item.view_type] || '#6b7280'}15` }}
              >
                <p className="text-xs font-medium text-gray-600">
                  {item.view_type}
                </p>
                <p 
                  className="text-lg font-bold"
                  style={{ color: COLORS[item.view_type] || '#6b7280' }}
                >
                  {item.uncertain_percent}%
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Analyse comparative */}
      <div className="mt-6 p-4 bg-gray-50 rounded">
        <h4 className="text-sm font-semibold text-gray-700 mb-2">Analyse</h4>
        <p className="text-xs text-gray-600">
          Ce graphique permet d'identifier si certains types de vues radiologiques 
          génèrent plus d'incertitude dans les diagnostics. Une différence significative 
          pourrait indiquer des difficultés de lecture spécifiques à certaines positions.
        </p>
      </div>
    </div>
  );
}
