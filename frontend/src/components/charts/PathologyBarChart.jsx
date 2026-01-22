/**
 * Composant de graphique en barres pour les pathologies
 * Utilise Recharts pour la visualisation
 */
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export function PathologyBarChart({ data }) {
  if (!data || data.length === 0) {
    return <div className="text-center text-gray-500">Aucune donnée disponible</div>;
  }

  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="pathology" 
          angle={-45} 
          textAnchor="end" 
          height={100}
          interval={0}
        />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="positive_count" fill="#3b82f6" name="Cas positifs" />
      </BarChart>
    </ResponsiveContainer>
  );
}
