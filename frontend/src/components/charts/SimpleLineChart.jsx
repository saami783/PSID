/**
 * Graphique linéaire simple pour tester
 */
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export function SimpleLineChart() {
  // Données de test
  const testData = [
    { mois: 'Jan', valeur: 100 },
    { mois: 'Fév', valeur: 150 },
    { mois: 'Mar', valeur: 120 },
    { mois: 'Avr', valeur: 180 },
    { mois: 'Mai', valeur: 200 }
  ];

  return (
    <div className="w-full">
      <h4 className="text-lg font-semibold mb-4">Graphique Linéaire Simple</h4>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={testData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="mois" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="valeur" stroke="#10b981" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
