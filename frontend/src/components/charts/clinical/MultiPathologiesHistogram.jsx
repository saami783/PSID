/**
 * Histogramme de Multi-Pathologies (Sévérité)
 * Montre combien de patients ont 0, 1, 2, 3+ pathologies simultanées
 * Objectif: Visualiser la complexité des cas
 */
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

export function MultiPathologiesHistogram({ data }) {
  if (!data || !data.bins || data.bins.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500 italic">Aucune donnée disponible</p>
      </div>
    );
  }

  // Transformer les données pour Recharts
  const chartData = data.bins.map((bin, index) => ({
    name: `${bin} pathologie${bin !== '1' ? 's' : ''}`,
    count: data.counts[index],
    bin: bin
  }));

  // Couleurs dégradées selon la sévérité
  const getColor = (bin) => {
    if (bin === '0') return '#10B981'; // Green - Aucune pathologie
    if (bin === '1') return '#3B82F6'; // Blue - Une pathologie
    if (bin === '2') return '#F59E0B'; // Amber - Deux pathologies
    if (bin === '3') return '#EF4444'; // Red - Trois pathologies
    if (bin === '4') return '#DC2626'; // Dark Red
    return '#991B1B'; // Very Dark Red - 5+
  };

  // Tooltip personnalisé
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const percentage = ((data.count / chartData.reduce((sum, d) => sum + d.count, 0)) * 100).toFixed(2);
      
      return (
        <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
          <p className="font-semibold text-gray-800">{data.name}</p>
          <p className="text-sm text-gray-600">
            Nombre de patients : <span className="font-bold text-blue-600">{data.count.toLocaleString()}</span>
          </p>
          <p className="text-sm text-gray-600">
            Pourcentage : <span className="font-bold text-green-600">{percentage}%</span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-4">
      {/* Statistiques résumées */}
      <div className="grid grid-cols-2 gap-3 text-sm bg-gray-50 p-3 rounded">
        <div>
          <span className="text-gray-600">Total patients:</span>
          <span className="ml-2 font-semibold text-gray-800">
            {data.total_patients?.toLocaleString() || '-'}
          </span>
        </div>
        <div>
          <span className="text-gray-600">Moyenne:</span>
          <span className="ml-2 font-semibold text-blue-600">
            {data.mean_pathologies?.toFixed(2) || '-'} pathologies/patient
          </span>
        </div>
        <div>
          <span className="text-gray-600">Médiane:</span>
          <span className="ml-2 font-semibold text-green-600">
            {data.median_pathologies?.toFixed(0) || '-'}
          </span>
        </div>
        <div>
          <span className="text-gray-600">Maximum:</span>
          <span className="ml-2 font-semibold text-red-600">
            {data.max_pathologies || '-'} pathologies
          </span>
        </div>
      </div>

      {/* Graphique */}
      <ResponsiveContainer width="100%" height={320}>
        <BarChart 
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
          <XAxis 
            dataKey="name" 
            angle={-30}
            textAnchor="end"
            height={80}
            tick={{ fontSize: 12, fill: '#4B5563' }}
          />
          <YAxis 
            label={{ 
              value: 'Nombre de patients', 
              angle: -90, 
              position: 'insideLeft',
              style: { fontSize: 12, fill: '#4B5563' }
            }}
            tick={{ fontSize: 12, fill: '#4B5563' }}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(59, 130, 246, 0.1)' }} />
          <Legend 
            wrapperStyle={{ paddingTop: '20px' }}
            content={() => (
              <p className="text-center text-xs text-gray-500 mt-2">
                📊 Complexité des cas - distribution du nombre de pathologies par patient
              </p>
            )}
          />
          <Bar 
            dataKey="count" 
            name="Nombre de patients"
            radius={[8, 8, 0, 0]}
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getColor(entry.bin)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Légende des couleurs */}
      <div className="flex flex-wrap items-center justify-center gap-3 text-xs">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-green-500 rounded"></div>
          <span>Aucune</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-blue-500 rounded"></div>
          <span>Une</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-amber-500 rounded"></div>
          <span>Deux</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-red-500 rounded"></div>
          <span>Trois+</span>
        </div>
      </div>
    </div>
  );
}
