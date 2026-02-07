/**
 * Service API pour l'inférence du modèle CheXpert
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5050';

async function handleResponse(response) {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Erreur réseau' }));
    throw new Error(error.message || `HTTP error! status: ${response.status}`);
  }
  return response.json();
}

export async function predictImage(file) {
  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(`${API_BASE_URL}/api/inference/predict`, {
    method: 'POST',
    body: formData,
  });

  return handleResponse(response);
}

export async function inferenceHealth() {
  const response = await fetch(`${API_BASE_URL}/api/inference/health`);
  return handleResponse(response);
}

