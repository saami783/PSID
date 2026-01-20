/**
 * Graphique en camembert simple pour tester
 */
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

export function SimplePieChart() {
  // Données de test
  const testData = [
    { name: 'Rouge', value: 30 },
    { name: 'Bleu', value: 25 },
    { name: 'Vert', value: 20 },
    { name: 'Jaune', value: 15 },
    { name: 'Violet', value: 10 }
  ];

  const COLORS = ['#ef4444', '#3b82f6', '#10b981', '#eab308', '#a855f7'];

  return (
    <div className="w-full">
      <h4 className="text-lg font-semibold mb-4">Graphique en Camembert Simple</h4>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={testData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
          >
            {testData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
