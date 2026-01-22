/**
 * Error Boundary pour capturer les erreurs React
 */
import { Component } from 'react';

export class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-red-50 p-8">
          <div className="max-w-4xl mx-auto">
            <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-red-500">
              <h1 className="text-2xl font-bold text-red-700 mb-4">
                Une erreur est survenue
              </h1>
              <div className="bg-red-100 rounded p-4 mb-4">
                <p className="font-mono text-sm text-red-800">
                  {this.state.error?.toString()}
                </p>
              </div>
              {this.state.errorInfo && (
                <details className="mt-4">
                  <summary className="cursor-pointer text-gray-600 hover:text-gray-800">
                    Stack trace
                  </summary>
                  <pre className="mt-2 p-4 bg-gray-100 rounded text-xs overflow-auto max-h-96">
                    {this.state.errorInfo.componentStack}
                  </pre>
                </details>
              )}
              <button
                onClick={() => window.location.reload()}
                className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
              >
                Recharger la page
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
