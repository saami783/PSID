/**
 * Histogramme de fréquence patient
 * Distribution du nombre de clichés par PatientID pour détecter les outliers
 */
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from 'recharts';

export function PatientFrequencyHistogram({ data }) {
  if (!data || !Array.isArray(data.bins) || !Array.isArray(data.counts) || data.bins.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Aucune donnée disponible
      </div>
    );
  }

  // Transformer les données pour le graphique
  const chartData = data.bins.map((bin, index) => ({
    bin: bin,
    count: data.counts[index],
    // Calculer le pourcentage
    percent: ((data.counts[index] / data.total_patients) * 100).toFixed(1)
  }));

  // Définir les couleurs (plus foncé pour les valeurs élevées = outliers potentiels)
  const getBarColor = (index, total) => {
    const ratio = index / total;
    if (ratio < 0.3) return '#10b981'; // Vert pour les valeurs normales
    if (ratio < 0.6) return '#f59e0b'; // Orange pour les valeurs moyennes
    return '#ef4444'; // Rouge pour les outliers potentiels
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="bg-white p-3 border rounded shadow-lg">
          <p className="font-semibold text-gray-800 mb-2">
            {item.bin} image{item.bin !== '1' ? 's' : ''} par patient
          </p>
          <p className="text-gray-600">
            Nombre de patients: <strong>{item.count.toLocaleString()}</strong>
          </p>
          <p className="text-gray-600">
            Pourcentage: <strong>{item.percent}%</strong>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={350}>
        <BarChart 
          data={chartData} 
          margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="bin" 
            angle={-45} 
            textAnchor="end" 
            height={60}
            interval={0}
            tick={{ fontSize: 11 }}
            label={{ 
              value: 'Nombre d\'images par patient', 
              position: 'bottom', 
              offset: 40 
            }}
          />
          <YAxis 
            tickFormatter={(value) => value.toLocaleString()}
            label={{ 
              value: 'Nombre de patients', 
              angle: -90, 
              position: 'insideLeft' 
            }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="count" name="Patients">
            {chartData.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={getBarColor(index, chartData.length)} 
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Statistiques résumées */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-blue-50 p-3 rounded">
          <p className="text-xs text-blue-600 uppercase font-medium">Total Patients</p>
          <p className="text-lg font-bold text-blue-800">
            {data.total_patients.toLocaleString()}
          </p>
        </div>
        <div className="bg-green-50 p-3 rounded">
          <p className="text-xs text-green-600 uppercase font-medium">Moyenne</p>
          <p className="text-lg font-bold text-green-800">
            {data.mean_images.toFixed(2)} img/patient
          </p>
        </div>
        <div className="bg-yellow-50 p-3 rounded">
          <p className="text-xs text-yellow-600 uppercase font-medium">Médiane</p>
          <p className="text-lg font-bold text-yellow-800">
            {data.median_images.toFixed(1)} img/patient
          </p>
        </div>
        <div className="bg-red-50 p-3 rounded">
          <p className="text-xs text-red-600 uppercase font-medium">Maximum</p>
          <p className="text-lg font-bold text-red-800">
            {data.max_images} images
          </p>
        </div>
      </div>

      {/* Légende des couleurs */}
      <div className="mt-3 flex items-center justify-center gap-4 text-xs">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-green-500 rounded"></div>
          <span className="text-gray-600">Normal</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-yellow-500 rounded"></div>
          <span className="text-gray-600">Élevé</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-red-500 rounded"></div>
          <span className="text-gray-600">Outlier</span>
        </div>
      </div>
    </div>
  );
}
