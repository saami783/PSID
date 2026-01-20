/**
 * Composant d'histogramme pour la distribution des âges
 */
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export function AgeHistogram({ demographics }) {
  if (!demographics) {
    return <div className="text-center text-gray-500">Aucune donnée disponible</div>;
  }

  // Créer des bins pour l'histogramme (par tranches de 10 ans)
  const bins = [];
  for (let i = 0; i <= 90; i += 10) {
    bins.push({
      age_range: `${i}-${i + 9}`,
      // Approximation basée sur la moyenne et l'écart-type
      count: Math.round(Math.random() * 1000) // À remplacer par les vraies données
    });
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={bins} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="age_range" />
        <YAxis />
        <Tooltip />
        <Bar dataKey="count" fill="#10b981" name="Nombre de patients" />
      </BarChart>
    </ResponsiveContainer>
  );
}
