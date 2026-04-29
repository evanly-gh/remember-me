/**
 * Embedding generation using Hugging Face Inference API
 * Model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
 */

const HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction";
const HF_TOKEN = process.env.EXPO_PUBLIC_HF_API_KEY;

/**
 * Generate embedding vector from text using Hugging Face API
 * @param {string} text - Text to embed
 * @param {number} retries - Number of retry attempts for rate limits
 * @returns {Promise<number[]>} 384-dimensional embedding vector
 */
export async function generateEmbedding(text, retries = 3) {
  if (!text || !text.trim()) {
    throw new Error('Text is required for embedding generation');
  }

  if (!HF_TOKEN) {
    console.warn('HF_API_KEY not found. Embeddings disabled.');
    return null;
  }

  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const response = await fetch(HF_API_URL, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${HF_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          inputs: text,
          options: {
            wait_for_model: true,  // Wait if model is loading (cold start)
            use_cache: true,       // Use cached results when possible
          }
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();

        // Handle rate limiting
        if (response.status === 429 && attempt < retries - 1) {
          const waitTime = Math.pow(2, attempt) * 1000; // Exponential backoff
          console.log(`Rate limited. Retrying in ${waitTime}ms...`);
          await new Promise(resolve => setTimeout(resolve, waitTime));
          continue;
        }

        throw new Error(`HF API error (${response.status}): ${errorText}`);
      }

      const embedding = await response.json();

      // HF returns the embedding directly as an array
      if (!Array.isArray(embedding)) {
        throw new Error('Invalid embedding format received from API');
      }

      return embedding;

    } catch (error) {
      if (attempt === retries - 1) {
        console.error('Embedding generation failed after retries:', error);
        throw error;
      }

      // Wait before retry
      await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
    }
  }
}

/**
 * Build searchable text from contact data for embedding generation
 * Combines all relevant fields into a coherent text description
 * @param {Object} contactData - Contact data including facial features
 * @returns {string} Searchable text description
 */
export function buildSearchableText(contactData) {
  const parts = [];

  // Basic information
  if (contactData.name) {
    parts.push(`Name: ${contactData.name}`);
  }

  if (contactData.title) {
    parts.push(`Relation: ${contactData.title}`);
  }

  if (contactData.event) {
    parts.push(`Event: ${contactData.event}`);
  }

  if (contactData.location) {
    parts.push(`Location: ${contactData.location}`);
  }

  if (contactData.notes) {
    parts.push(`Notes: ${contactData.notes}`);
  }

  // Facial features
  const facial = contactData.facial_details?.data || contactData.facial_details;
  if (facial && typeof facial === 'object') {
    const features = [];
    const demographics = [];
    const appearance = [];

    // Demographics
    if (facial.gender) {
      demographics.push(facial.gender);
    }

    if (facial.ethnicity) {
      demographics.push(facial.ethnicity);
    }

    if (facial.age_range) {
      demographics.push(`age ${facial.age_range}`);
    } else if (facial.age_estimate) {
      demographics.push(`age ${facial.age_estimate}`);
    }

    if (demographics.length > 0) {
      parts.push(`Demographics: ${demographics.join(', ')}`);
    }

    // Physical appearance
    if (facial.wearing_glasses || facial.glasses_detected) {
      appearance.push('wearing glasses');
    }

    if (facial.has_beard) {
      appearance.push('has beard');
    }

    if (facial.wearing_hat || facial.hat_detected) {
      appearance.push('wearing hat');
    }

    if (facial.is_bald) {
      appearance.push('bald');
    }

    if (facial.smiling || facial.smiling_celeba) {
      appearance.push('smiling');
    }

    if (facial.wearing_earrings || facial.earring_detected) {
      appearance.push('wearing earrings');
    }

    if (facial.heavy_makeup) {
      appearance.push('wearing makeup');
    }

    if (appearance.length > 0) {
      parts.push(`Appearance: ${appearance.join(', ')}`);
    }

    // Hair color
    const hairColor = facial.hair_color?.name || facial.hair_color_celeba;
    if (hairColor) {
      features.push(`${hairColor} hair`);
    }

    if (facial.hair_length) {
      features.push(`${facial.hair_length} hair`);
    }

    // Eye color
    const eyeColor = typeof facial.eye_color === 'string'
      ? facial.eye_color
      : facial.eye_color?.name;
    if (eyeColor) {
      features.push(`${eyeColor} eyes`);
    }

    if (facial.eye_shape) {
      features.push(`${facial.eye_shape} eyes`);
    }

    if (features.length > 0) {
      parts.push(`Features: ${features.join(', ')}`);
    }

    // Face structure
    if (facial.face_shape) {
      parts.push(`Face shape: ${facial.face_shape}`);
    }

    // Emotion
    if (facial.primary_emotion) {
      parts.push(`Emotion: ${facial.primary_emotion}`);
    }

    if (facial.mood) {
      parts.push(`Mood: ${facial.mood}`);
    }
  }

  return parts.join('\n');
}

/**
 * Calculate cosine similarity between two vectors
 * Used for local similarity comparison if needed
 * @param {number[]} vecA - First vector
 * @param {number[]} vecB - Second vector
 * @returns {number} Similarity score between 0 and 1
 */
export function cosineSimilarity(vecA, vecB) {
  if (!vecA || !vecB || vecA.length !== vecB.length) {
    throw new Error('Invalid vectors for similarity calculation');
  }

  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));

  if (magA === 0 || magB === 0) {
    return 0;
  }

  return dotProduct / (magA * magB);
}
