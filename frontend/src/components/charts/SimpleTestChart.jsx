/**
 * Graphique de test simple pour vérifier que Recharts fonctionne
 */
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export function SimpleTestChart() {
  // Données de test très simples
  const testData = [
    { name: 'A', value: 10 },
    { name: 'B', value: 20 },
    { name: 'C', value: 15 },
    { name: 'D', value: 25 },
    { name: 'E', value: 30 }
  ];

  return (
    <div className="w-full">
      <h4 className="text-lg font-semibold mb-4">Graphique de Test Simple</h4>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={testData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="value" fill="#3b82f6" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
