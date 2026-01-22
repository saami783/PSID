import React from 'react';
import { Link } from 'react-router-dom';

/**
 * Page d'accueil DeepChex
 * Présente le projet, le dataset CheXpert et les objectifs de la plateforme
 */
function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-950 via-slate-900 to-blue-900">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
        
        <div className="container mx-auto px-4 py-20 relative z-10">
          <div className="max-w-5xl mx-auto text-center">
            {/* Logo et Titre */}
            <div className="mb-8 flex items-center justify-center space-x-4">
              <div className="w-16 h-16 bg-blue-500 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/50">
                <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <h1 className="text-6xl font-bold text-white tracking-tight">
                Deep<span className="text-blue-400">Chex</span>
              </h1>
            </div>

            {/* Phrase d'accroche */}
            <p className="text-2xl text-blue-200 font-medium mb-6">
              L'intelligence artificielle au service de la radiologie thoracique
            </p>

            {/* Paragraphe d'introduction */}
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 mb-12 border border-white/20 shadow-2xl">
              <p className="text-lg text-gray-200 leading-relaxed">
                DeepChex est né de la volonté de transformer le diagnostic médical en utilisant la puissance 
                du dataset <span className="font-semibold text-blue-300">CheXpert</span> pour entraîner un modèle 
                capable d'identifier avec précision les anomalies pulmonaires. Ce projet ne se contente pas de 
                prédire des résultats, il s'appuie sur une compréhension profonde des données cliniques et 
                techniques pour offrir une assistance au diagnostic <span className="font-semibold text-blue-300">transparente et robuste</span>.
              </p>
            </div>

            {/* Call to Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link 
                to="/analytics"
                className="group px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-xl shadow-lg shadow-blue-500/50 transition-all duration-300 transform hover:scale-105 flex items-center space-x-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span>Explorer les données</span>
              </Link>
              
              <button
                className="group px-8 py-4 bg-white/10 hover:bg-white/20 text-white font-semibold rounded-xl border-2 border-white/30 transition-all duration-300 transform hover:scale-105 flex items-center space-x-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span>Tester le modèle</span>
                <span className="text-xs bg-yellow-500 text-yellow-900 px-2 py-1 rounded-full ml-2">Bientôt</span>
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Section Dataset CheXpert */}
      <section className="py-20 bg-slate-900/50">
        <div className="container mx-auto px-4">
          <div className="max-w-5xl mx-auto">
            {/* Titre de section */}
            <div className="text-center mb-16">
              <h2 className="text-4xl font-bold text-white mb-4">
                Le Dataset CheXpert
              </h2>
              <div className="w-24 h-1 bg-blue-500 mx-auto rounded-full"></div>
            </div>

            {/* Contenu en grille */}
            <div className="grid md:grid-cols-2 gap-8">
              {/* Origine des données */}
              <div className="bg-gradient-to-br from-blue-900/50 to-slate-800/50 backdrop-blur-sm rounded-2xl p-8 border border-blue-500/20 hover:border-blue-500/40 transition-all duration-300 shadow-xl">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center">
                      <svg className="w-7 h-7 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                      </svg>
                    </div>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white mb-3">Origine des Données</h3>
                    <p className="text-gray-300 leading-relaxed">
                      Le projet repose sur <span className="font-semibold text-blue-300">CheXpert</span>, une vaste 
                      base de données contenant des centaines de milliers de radiographies thoraciques interprétées 
                      par des experts. Cette base de données constitue le socle de notre apprentissage machine, 
                      permettant au modèle de se confronter à une immense diversité de cas cliniques réels, allant 
                      des observations les plus communes aux pathologies les plus complexes comme 
                      <span className="font-semibold text-blue-300"> l'épanchement pleural</span> ou 
                      <span className="font-semibold text-blue-300"> l'atélectasie</span>.
                    </p>
                  </div>
                </div>
              </div>

              {/* Contexte Technique */}
              <div className="bg-gradient-to-br from-slate-800/50 to-blue-900/50 backdrop-blur-sm rounded-2xl p-8 border border-slate-500/20 hover:border-slate-500/40 transition-all duration-300 shadow-xl">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 bg-slate-500/20 rounded-lg flex items-center justify-center">
                      <svg className="w-7 h-7 text-slate-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                    </div>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white mb-3">Contexte Technique</h3>
                    <p className="text-gray-300 leading-relaxed">
                      L'analyse de l'impact des <span className="font-semibold text-slate-300">dispositifs médicaux</span> et 
                      des <span className="font-semibold text-slate-300">types de vues (AP/PA)</span> garantit que le modèle 
                      reste performant dans des conditions réelles, même lorsque le patient est appareillé ou en situation 
                      d'urgence. Cette approche rigoureuse assure une fiabilité clinique optimale.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Statistiques clés */}
            <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="text-center p-6 bg-white/5 rounded-xl border border-white/10">
                <div className="text-4xl font-bold text-blue-400 mb-2">224k+</div>
                <div className="text-sm text-gray-400 uppercase tracking-wide">Radiographies</div>
              </div>
              <div className="text-center p-6 bg-white/5 rounded-xl border border-white/10">
                <div className="text-4xl font-bold text-blue-400 mb-2">65k+</div>
                <div className="text-sm text-gray-400 uppercase tracking-wide">Patients</div>
              </div>
              <div className="text-center p-6 bg-white/5 rounded-xl border border-white/10">
                <div className="text-4xl font-bold text-blue-400 mb-2">14</div>
                <div className="text-sm text-gray-400 uppercase tracking-wide">Pathologies</div>
              </div>
              <div className="text-center p-6 bg-white/5 rounded-xl border border-white/10">
                <div className="text-4xl font-bold text-blue-400 mb-2">100%</div>
                <div className="text-sm text-gray-400 uppercase tracking-wide">Transparence</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section Piliers (optionnelle - peut être développée) */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-5xl mx-auto">
            <div className="text-center mb-16">
              <h2 className="text-4xl font-bold text-white mb-4">
                Nos Piliers
              </h2>
              <div className="w-24 h-1 bg-blue-500 mx-auto rounded-full"></div>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              {/* Pilier 1 */}
              <div className="text-center group">
                <div className="w-16 h-16 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-blue-500/30 transition-all duration-300">
                  <svg className="w-8 h-8 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Transparence</h3>
                <p className="text-gray-400">Chaque décision du modèle est expliquée et traçable</p>
              </div>

              {/* Pilier 2 */}
              <div className="text-center group">
                <div className="w-16 h-16 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-blue-500/30 transition-all duration-300">
                  <svg className="w-8 h-8 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Fiabilité</h3>
                <p className="text-gray-400">Validation rigoureuse sur des données cliniques réelles</p>
              </div>

              {/* Pilier 3 */}
              <div className="text-center group">
                <div className="w-16 h-16 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-blue-500/30 transition-all duration-300">
                  <svg className="w-8 h-8 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Équité</h3>
                <p className="text-gray-400">Analyse des biais pour un diagnostic juste et impartial</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 border-t border-white/10">
        <div className="container mx-auto px-4">
          <div className="text-center text-gray-400">
            <p className="mb-2">© 2026 DeepChex - Plateforme de diagnostic assisté par IA</p>
            <p className="text-sm">Projet basé sur le dataset CheXpert de Stanford</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default Home;
