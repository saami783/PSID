/**
 * Pyramide des âges - Comparaison de la structure de population par genre
 * Graphique horizontal avec barres négatives pour les hommes et positives pour les femmes
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
  ReferenceLine
} from 'recharts';

export function AgePyramid({ data }) {
  // Vérification complète des données
  if (!data || !Array.isArray(data.age_bins) || !Array.isArray(data.male_counts) || !Array.isArray(data.female_counts)) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Aucune donnée disponible
      </div>
    );
  }

  if (data.age_bins.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Aucune donnée disponible
      </div>
    );
  }

  // Transformer les données pour le graphique en pyramide
  // Les hommes sont affichés avec des valeurs négatives
  const chartData = data.age_bins.map((bin, index) => ({
    age_bin: bin,
    male: -(data.male_counts[index] || 0), // Valeurs négatives pour les hommes
    female: data.female_counts[index] || 0,
    maleLabel: data.male_counts[index] || 0, // Pour le tooltip
    femaleLabel: data.female_counts[index] || 0
  }));

  // Calculer le max pour l'axe X symétrique (avec protection contre les tableaux vides)
  const maxMale = data.male_counts.length > 0 ? Math.max(...data.male_counts) : 0;
  const maxFemale = data.female_counts.length > 0 ? Math.max(...data.female_counts) : 0;
  const maxValue = Math.max(maxMale, maxFemale, 1); // Au moins 1 pour éviter division par 0

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const maleValue = Math.abs(payload.find(p => p.dataKey === 'male')?.value || 0);
      const femaleValue = payload.find(p => p.dataKey === 'female')?.value || 0;
      
      return (
        <div className="bg-white p-3 border rounded shadow-lg">
          <p className="font-semibold text-gray-800 mb-2">{label} ans</p>
          <p className="text-blue-600">
            Hommes: {maleValue.toLocaleString()}
          </p>
          <p className="text-pink-600">
            Femmes: {femaleValue.toLocaleString()}
          </p>
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
          layout="vertical"
          margin={{ top: 20, right: 30, left: 50, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            type="number" 
            domain={[-maxValue * 1.1, maxValue * 1.1]}
            tickFormatter={(value) => Math.abs(value).toLocaleString()}
          />
          <YAxis 
            type="category" 
            dataKey="age_bin" 
            width={60}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <ReferenceLine x={0} stroke="#666" />
          <Bar 
            dataKey="male" 
            fill="#3b82f6" 
            name="Hommes" 
            radius={[4, 0, 0, 4]}
          />
          <Bar 
            dataKey="female" 
            fill="#ec4899" 
            name="Femmes" 
            radius={[0, 4, 4, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
      
      {/* Statistiques résumées */}
      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
        <div className="bg-blue-50 p-3 rounded">
          <p className="text-blue-800 font-medium">Total Hommes</p>
          <p className="text-blue-600 text-lg font-bold">
            {data.male_counts.reduce((a, b) => a + b, 0).toLocaleString()}
          </p>
        </div>
        <div className="bg-pink-50 p-3 rounded">
          <p className="text-pink-800 font-medium">Total Femmes</p>
          <p className="text-pink-600 text-lg font-bold">
            {data.female_counts.reduce((a, b) => a + b, 0).toLocaleString()}
          </p>
        </div>
      </div>
    </div>
  );
}
