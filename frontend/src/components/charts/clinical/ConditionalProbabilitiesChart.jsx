/**
 * Graphique de Probabilités Conditionnelles (Comorbidités)
 * Montre P(Pathologie Y | Pathologie X)
 * Question: "Sachant que le patient a [Maladie X], quelle est la probabilité qu'il ait aussi les autres ?"
 * 
 * Ce graphique est PILOTÉ par le menu déroulant "Focus Pathologie" de la GlobalFilterBar
 */
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export function ConditionalProbabilitiesChart({ data }) {
  // Cas où aucune pathologie n'est sélectionnée
  if (!data || !data.target_disease) {
    return (
      <div className="flex flex-col items-center justify-center h-64 bg-gradient-to-br from-green-50 to-blue-50 rounded-lg border-2 border-dashed border-green-300">
        <svg 
          className="w-16 h-16 text-green-400 mb-3"
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth={2} 
            d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" 
          />
        </svg>
        <p className="text-gray-700 font-medium text-center px-4">
          🎯 Sélectionnez une pathologie dans le menu <strong>"Focus Pathologie"</strong>
        </p>
        <p className="text-sm text-gray-500 text-center px-4 mt-2">
          Ce graphique affichera les comorbidités associées
        </p>
      </div>
    );
  }

  // Cas où il y a une erreur ou aucun patient
  if (data.error || !data.comorbidities || data.comorbidities.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <p className="text-amber-600 font-medium">
            {data.message || data.error || 'Aucune donnée disponible'}
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Pathologie sélectionnée: <strong>{data.target_disease}</strong>
          </p>
        </div>
      </div>
    );
  }

  // Préparer les données pour le graphique
  const chartData = data.comorbidities.map((item) => ({
    name: item.pathology,
    probability: item.probability_percent,
    count: item.count
  }));

  // Ne garder que les 12 comorbidités les plus fréquentes pour la lisibilité
  const topComorbidities = chartData.slice(0, 12);

  // Tooltip personnalisé
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      
      return (
        <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
          <p className="font-semibold text-gray-800">{data.name}</p>
          <p className="text-sm text-gray-600">
            Probabilité conditionnelle : 
            <span className="ml-1 font-bold text-green-600">{data.probability.toFixed(2)}%</span>
          </p>
          <p className="text-sm text-gray-600">
            Patients concernés : 
            <span className="ml-1 font-bold text-blue-600">{data.count.toLocaleString()}</span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-4">
      {/* En-tête avec titre dynamique */}
      <div className="bg-gradient-to-r from-green-50 to-blue-50 p-3 rounded-lg border border-green-200">
        <h3 className="text-sm font-bold text-gray-800 mb-1">
          🎯 Comorbidités associées à : <span className="text-green-700">{data.target_disease}</span>
        </h3>
        <p className="text-xs text-gray-600">
          Sur <strong>{data.target_count?.toLocaleString()}</strong> patients ayant {data.target_disease}
          {data.total_patients_in_dataset && (
            <span> (sur {data.total_patients_in_dataset.toLocaleString()} total dans le dataset filtré)</span>
          )}
        </p>
      </div>

      {/* Graphique */}
      <ResponsiveContainer width="100%" height={350}>
        <BarChart 
          data={topComorbidities}
          margin={{ top: 20, right: 30, left: 20, bottom: 80 }}
          layout="horizontal"
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
          <XAxis 
            dataKey="name" 
            angle={-45}
            textAnchor="end"
            height={100}
            interval={0}
            tick={{ fontSize: 11, fill: '#4B5563' }}
          />
          <YAxis 
            label={{ 
              value: 'Probabilité (%)', 
              angle: -90, 
              position: 'insideLeft',
              style: { fontSize: 12, fill: '#4B5563' }
            }}
            tick={{ fontSize: 12, fill: '#4B5563' }}
            domain={[0, 100]}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(34, 197, 94, 0.1)' }} />
          <Legend 
            wrapperStyle={{ paddingTop: '10px' }}
            content={() => (
              <p className="text-center text-xs text-gray-500 mt-2">
                📈 P(Pathologie Y | {data.target_disease}) - probabilités conditionnelles
              </p>
            )}
          />
          <Bar 
            dataKey="probability" 
            name="Probabilité conditionnelle (%)"
            fill="#22C55E"
            radius={[8, 8, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>

      {/* Note explicative */}
      <div className="bg-blue-50 p-2 rounded text-xs text-gray-700 border border-blue-200">
        <p>
          <strong>💡 Interprétation :</strong> Ce graphique montre la fréquence des autres pathologies 
          <strong> parmi les patients ayant {data.target_disease}</strong>. 
          Une valeur de 50% signifie que la moitié des patients avec {data.target_disease} 
          ont aussi cette comorbidité.
        </p>
      </div>
    </div>
  );
}
