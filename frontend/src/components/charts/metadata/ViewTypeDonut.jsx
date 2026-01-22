/**
 * Donut Chart des View Types
 * Ratio Frontal / Lateral / AP / PA
 */
import { 
  PieChart, 
  Pie, 
  Cell, 
  ResponsiveContainer, 
  Tooltip, 
  Legend 
} from 'recharts';

export function ViewTypeDonut({ data }) {
  const hasFrontalLateral = data?.frontal_lateral && Array.isArray(data.frontal_lateral) && data.frontal_lateral.length > 0;
  const hasApPa = data?.ap_pa && Array.isArray(data.ap_pa) && data.ap_pa.length > 0;
  
  if (!data || (!hasFrontalLateral && !hasApPa)) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Aucune donnée disponible
      </div>
    );
  }

  const COLORS_FL = ['#3b82f6', '#8b5cf6'];
  const COLORS_APPA = ['#10b981', '#f59e0b', '#ec4899', '#06b6d4'];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="bg-white p-3 border rounded shadow-lg">
          <p className="font-semibold text-gray-800">{item.name}</p>
          <p className="text-gray-600">
            Nombre: <strong>{item.value.toLocaleString()}</strong>
          </p>
          <p className="text-blue-600">
            Pourcentage: <strong>{item.percent}%</strong>
          </p>
        </div>
      );
    }
    return null;
  };

  // Label personnalisé pour le donut
  const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }) => {
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return percent > 0.05 ? (
      <text 
        x={x} 
        y={y} 
        fill="white" 
        textAnchor="middle" 
        dominantBaseline="central"
        fontSize={12}
        fontWeight="bold"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    ) : null;
  };

  return (
    <div className="w-full">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Donut Frontal/Lateral */}
        {hasFrontalLateral && (
          <div>
            <h4 className="text-sm font-semibold text-gray-700 mb-2 text-center">
              Orientation (Frontal / Latéral)
            </h4>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={data.frontal_lateral}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={90}
                  paddingAngle={2}
                  dataKey="value"
                  labelLine={false}
                  label={renderCustomizedLabel}
                >
                  {data.frontal_lateral.map((entry, index) => (
                    <Cell 
                      key={`cell-fl-${index}`} 
                      fill={COLORS_FL[index % COLORS_FL.length]} 
                    />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
            
            {/* Stats détaillées */}
            <div className="grid grid-cols-2 gap-2 mt-2">
              {data.frontal_lateral.map((item, index) => (
                <div 
                  key={item.name}
                  className="p-2 rounded text-center"
                  style={{ backgroundColor: `${COLORS_FL[index]}15` }}
                >
                  <p className="text-xs text-gray-600">{item.name}</p>
                  <p 
                    className="text-lg font-bold"
                    style={{ color: COLORS_FL[index] }}
                  >
                    {item.value.toLocaleString()}
                  </p>
                  <p className="text-xs text-gray-500">{item.percent}%</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Donut AP/PA */}
        {hasApPa && (
          <div>
            <h4 className="text-sm font-semibold text-gray-700 mb-2 text-center">
              Position (AP / PA / Autres)
            </h4>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={data.ap_pa}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={90}
                  paddingAngle={2}
                  dataKey="value"
                  labelLine={false}
                  label={renderCustomizedLabel}
                >
                  {data.ap_pa.map((entry, index) => (
                    <Cell 
                      key={`cell-ap-${index}`} 
                      fill={COLORS_APPA[index % COLORS_APPA.length]} 
                    />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>

            {/* Stats détaillées */}
            <div className="grid grid-cols-2 gap-2 mt-2">
              {data.ap_pa.slice(0, 4).map((item, index) => (
                <div 
                  key={item.name}
                  className="p-2 rounded text-center"
                  style={{ backgroundColor: `${COLORS_APPA[index]}15` }}
                >
                  <p className="text-xs text-gray-600">{item.name}</p>
                  <p 
                    className="text-lg font-bold"
                    style={{ color: COLORS_APPA[index] }}
                  >
                    {item.value.toLocaleString()}
                  </p>
                  <p className="text-xs text-gray-500">{item.percent}%</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Résumé total */}
      <div className="mt-6 p-4 bg-gray-50 rounded text-center">
        <p className="text-sm text-gray-600">Total des images analysées</p>
        <p className="text-2xl font-bold text-gray-800">
          {(data.frontal_lateral?.reduce((sum, item) => sum + item.value, 0) || 0).toLocaleString()}
        </p>
      </div>
    </div>
  );
}
