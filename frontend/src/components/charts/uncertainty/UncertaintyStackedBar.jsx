/**
 * Stacked Bar Chart - Positive vs Uncertain
 * Pour chaque maladie, proportion de labels '1' vs '-1'
 */
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';

export function UncertaintyStackedBar({ data }) {
  if (!data || !Array.isArray(data) || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Aucune donnée disponible
      </div>
    );
  }

  // Raccourcir les noms pour l'affichage
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
    return shortNames[name] || name.substring(0, 10);
  };

  const chartData = data.map(item => ({
    ...item,
    shortName: shortenName(item.pathology)
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      const total = item.positive_count + item.uncertain_count + item.negative_count;
      
      return (
        <div className="bg-white p-3 border rounded shadow-lg">
          <p className="font-semibold text-gray-800 mb-2">{item.pathology}</p>
          <div className="text-sm space-y-1">
            <p className="flex justify-between gap-4">
              <span className="text-green-600">Positifs (1):</span>
              <strong>{item.positive_count.toLocaleString()} ({item.positive_percent}%)</strong>
            </p>
            <p className="flex justify-between gap-4">
              <span className="text-yellow-600">Incertains (-1):</span>
              <strong>{item.uncertain_count.toLocaleString()} ({item.uncertain_percent}%)</strong>
            </p>
            <p className="flex justify-between gap-4">
              <span className="text-gray-600">Négatifs (0):</span>
              <strong>{item.negative_count.toLocaleString()}</strong>
            </p>
            <p className="pt-1 border-t text-gray-500">
              Total: {total.toLocaleString()}
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={400}>
        <BarChart 
          data={chartData} 
          margin={{ top: 20, right: 30, left: 20, bottom: 80 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="shortName" 
            angle={-45} 
            textAnchor="end" 
            height={80}
            interval={0}
            tick={{ fontSize: 10 }}
          />
          <YAxis 
            tickFormatter={(value) => value >= 1000 ? `${(value/1000).toFixed(0)}k` : value}
            label={{ value: 'Nombre de labels', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ paddingTop: 20 }} />
          <Bar 
            dataKey="positive_count" 
            stackId="a" 
            fill="#10b981" 
            name="Positifs (1)"
          />
          <Bar 
            dataKey="uncertain_count" 
            stackId="a" 
            fill="#f59e0b" 
            name="Incertains (-1)"
          />
        </BarChart>
      </ResponsiveContainer>

      {/* Résumé */}
      <div className="mt-4 grid grid-cols-2 gap-4">
        <div className="bg-green-50 p-3 rounded">
          <p className="text-sm text-green-700 font-medium">Total Labels Positifs</p>
          <p className="text-xl font-bold text-green-800">
            {data.reduce((sum, item) => sum + item.positive_count, 0).toLocaleString()}
          </p>
        </div>
        <div className="bg-yellow-50 p-3 rounded">
          <p className="text-sm text-yellow-700 font-medium">Total Labels Incertains</p>
          <p className="text-xl font-bold text-yellow-800">
            {data.reduce((sum, item) => sum + item.uncertain_count, 0).toLocaleString()}
          </p>
        </div>
      </div>

      {/* Top pathologies avec le plus d'incertitude */}
      <div className="mt-4">
        <h4 className="text-sm font-semibold text-gray-700 mb-2">
          Pathologies avec le plus d'incertitude
        </h4>
        <div className="space-y-2">
          {data.slice(0, 3).map((item, index) => (
            <div key={item.pathology} className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-600 w-6">{index + 1}.</span>
              <span className="text-sm flex-1">{item.pathology}</span>
              <div className="w-32 bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-yellow-500 h-2 rounded-full"
                  style={{ width: `${item.uncertain_percent}%` }}
                />
              </div>
              <span className="text-sm text-yellow-600 font-medium w-12 text-right">
                {item.uncertain_percent}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
