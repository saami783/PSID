/**
 * Composant Card réutilisable (UI pur)
 */
export function Card({ title, children, className = '' }) {
  return (
    <div className={`bg-white rounded-lg shadow-md p-6 ${className}`}>
      {title && (
        <h3 className="text-xl font-bold mb-4 text-gray-800">{title}</h3>
      )}
      {children}
    </div>
  );
}
