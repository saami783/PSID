import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { UploadDropzone } from '../components/ui/UploadDropzone';
import { predictImage, inferenceHealth } from '../services/inference';

function formatPercent(value) {
  return `${value.toFixed(2)}%`;
}

export default function InferencePage() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const response = await predictImage(file);
      setResult(response.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const checkHealth = async () => {
    setError(null);
    try {
      const res = await inferenceHealth();
      setHealth(res.data);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-blue-900 text-white">
      <div className="container mx-auto px-4 py-10 max-w-6xl">
        <div className="flex items-center gap-3 mb-8">
          <Link to="/" className="text-blue-300 hover:text-blue-200 flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Retour
          </Link>
          <h1 className="text-3xl font-bold">Tester le modèle CheXpert</h1>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white/5 border border-white/10 rounded-2xl p-6 shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Déposez une radiographie</h2>
              <button
                onClick={checkHealth}
                className="text-xs px-3 py-1 rounded-full bg-emerald-500/20 border border-emerald-400/40 hover:bg-emerald-500/30"
              >
                Vérifier le service
              </button>
            </div>
            <UploadDropzone onFileSelected={setFile} disabled={loading} />
            <button
              onClick={handlePredict}
              disabled={!file || loading}
              className="mt-4 w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 text-white font-semibold py-3 rounded-xl shadow-lg shadow-blue-500/30 transition"
            >
              {loading ? 'Analyse en cours...' : 'Lancer le diagnostic'}
            </button>
            {error && (
              <p className="mt-3 text-sm text-red-300">{error}</p>
            )}
            {health && (
              <p className="mt-3 text-sm text-emerald-300">Modèle prêt sur {health.device}</p>
            )}
          </div>

          <div className="bg-white/5 border border-white/10 rounded-2xl p-6 shadow-2xl min-h-[320px]">
            <h2 className="text-xl font-semibold mb-4">Résultats</h2>
            {!result && <p className="text-gray-300">Déposez une image pour voir le diagnostic.</p>}

            {result && (
              <div className="space-y-4">
                <div className="flex items-center justify-between text-sm text-gray-300">
                  <span>Temps d'inférence</span>
                  <span className="font-semibold text-white">{result.inference_time_ms} ms</span>
                </div>
                <div className="flex items-center justify-between text-sm text-gray-300">
                  <span>Device</span>
                  <span className="font-semibold text-white">{result.device}</span>
                </div>
                <div className="border-t border-white/10 pt-3">
                  <p className="text-sm text-gray-300 mb-2">Top 5</p>
                  <div className="space-y-2">
                    {result.predictions.slice(0, 5).map((p, idx) => (
                      <div key={p.label} className="flex items-center gap-3">
                        <div className="w-6 text-gray-400">#{idx + 1}</div>
                        <div className="flex-1">
                          <div className="flex items-center justify-between">
                            <span className="font-semibold">{p.label}</span>
                            <span className="text-blue-200">{formatPercent(p.percentage)}</span>
                          </div>
                          <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-blue-500"
                              style={{ width: `${p.percentage}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <details className="mt-2 text-sm text-gray-300">
                  <summary className="cursor-pointer text-white font-semibold">Voir toutes les pathologies</summary>
                  <ul className="mt-2 space-y-1 max-h-60 overflow-y-auto pr-2">
                    {result.predictions.map((p) => (
                      <li key={p.label} className="flex justify-between">
                        <span>{p.label}</span>
                        <span className="text-blue-200">{formatPercent(p.percentage)}</span>
                      </li>
                    ))}
                  </ul>
                </details>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

