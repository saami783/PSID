/**
 * Grouped Bar Chart - Pathologies vs Support Devices
 * Impact de la présence de matériel médical sur les diagnostics
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

export function PathologyVsDevices({ data }) {
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
    shortName: shortenName(item.pathology),
    // Calculer la différence de prévalence
    difference: (item.with_devices_percent - item.without_devices_percent).toFixed(1)
  }));

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      const diff = parseFloat(item.difference);
      
      return (
        <div className="bg-white p-3 border rounded shadow-lg">
          <p className="font-semibold text-gray-800 mb-2">{item.pathology}</p>
          <div className="text-sm space-y-1">
            <p className="flex justify-between gap-4">
              <span className="text-blue-600">Avec Support Devices:</span>
              <strong>{item.with_devices.toLocaleString()} ({item.with_devices_percent}%)</strong>
            </p>
            <p className="flex justify-between gap-4">
              <span className="text-green-600">Sans Support Devices:</span>
              <strong>{item.without_devices.toLocaleString()} ({item.without_devices_percent}%)</strong>
            </p>
            <p className={`pt-1 border-t ${diff > 0 ? 'text-red-600' : 'text-green-600'}`}>
              Différence: <strong>{diff > 0 ? '+' : ''}{diff}%</strong>
              {Math.abs(diff) > 5 && (
                <span className="ml-2 text-xs">
                  {diff > 0 ? '(Plus fréquent avec devices)' : '(Moins fréquent avec devices)'}
                </span>
              )}
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={450}>
        <BarChart 
          data={chartData} 
          margin={{ top: 20, right: 30, left: 20, bottom: 100 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="shortName" 
            angle={-45} 
            textAnchor="end" 
            height={100}
            interval={0}
            tick={{ fontSize: 10 }}
          />
          <YAxis 
            tickFormatter={(value) => value >= 1000 ? `${(value/1000).toFixed(0)}k` : value}
            label={{ value: 'Nombre de cas positifs', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ paddingTop: 20 }} />
          <Bar 
            dataKey="with_devices" 
            name="Avec Support Devices" 
            fill="#3b82f6"
            radius={[4, 4, 0, 0]}
          />
          <Bar 
            dataKey="without_devices" 
            name="Sans Support Devices" 
            fill="#10b981"
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>

      {/* Analyse des différences significatives */}
      <div className="mt-4">
        <h4 className="text-sm font-semibold text-gray-700 mb-2">
          Pathologies les plus impactées par les Support Devices
        </h4>
        <div className="space-y-2">
          {chartData
            .filter(item => Math.abs(parseFloat(item.difference)) > 2)
            .slice(0, 5)
            .map((item, index) => {
              const diff = parseFloat(item.difference);
              const isPositive = diff > 0;
              
              return (
                <div 
                  key={item.pathology}
                  className="flex items-center gap-3 p-2 rounded bg-gray-50"
                >
                  <span className="text-sm text-gray-600 w-6">{index + 1}.</span>
                  <span className="text-sm flex-1">{item.pathology}</span>
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-semibold ${isPositive ? 'text-red-600' : 'text-green-600'}`}>
                      {isPositive ? '+' : ''}{diff}%
                    </span>
                    <span className={`text-xs px-2 py-0.5 rounded ${isPositive ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                      {isPositive ? 'Plus fréquent' : 'Moins fréquent'}
                    </span>
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      {/* Note explicative */}
      <div className="mt-4 p-4 bg-blue-50 rounded">
        <h4 className="text-sm font-semibold text-blue-800 mb-1">Note</h4>
        <p className="text-xs text-blue-700">
          Les "Support Devices" incluent les équipements médicaux visibles sur les radiographies 
          (tubes, cathéters, pacemakers, etc.). Une différence significative peut indiquer une 
          corrélation entre la présence de dispositifs médicaux et certaines conditions cliniques, 
          ou un biais dans les données.
        </p>
      </div>
    </div>
  );
}
