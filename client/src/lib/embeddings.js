/**
 * Embedding generation using Hugging Face Inference API
 * Model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
 */

const HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction";
const HF_TOKEN = process.env.EXPO_PUBLIC_HF_API_KEY;

let warnedAboutMissingKey = false;

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
    if (!warnedAboutMissingKey) {
      console.warn('EXPO_PUBLIC_HF_API_KEY not set. Embeddings disabled.');
      warnedAboutMissingKey = true;
    }
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

    // Physical appearance — only fields backed by reliable models:
    //   ObstructionViT (glasses/sunglasses/mask), SegFormer (hat),
    //   MediaPipe blendshapes (smiling). Older FaRL-derived flags
    //   (has_beard, is_bald, wearing_earrings, heavy_makeup, etc.)
    //   were noisy enough to hurt search quality and have been dropped.
    if (facial.wearing_glasses) appearance.push('wearing glasses');
    if (facial.wearing_sunglasses) appearance.push('wearing sunglasses');
    if (facial.wearing_mask) appearance.push('wearing mask');
    if (facial.hat_detected) appearance.push('wearing hat');
    if (facial.smiling) appearance.push('smiling');
    if (appearance.length > 0) parts.push(`Appearance: ${appearance.join(', ')}`);

    // Hair (color = ColorAnalyzer, length = SegFormer, type = HairTypeViT).
    if (facial.hair_color?.name) features.push(`${facial.hair_color.name} hair`);
    if (facial.hair_length) features.push(`${facial.hair_length} hair`);
    if (facial.hair_type) features.push(`${facial.hair_type} hair texture`);

    // Eye color + shape.
    const eyeColor = typeof facial.eye_color === 'string'
      ? facial.eye_color
      : facial.eye_color?.name;
    if (eyeColor) features.push(`${eyeColor} eyes`);
    if (facial.eye_shape) features.push(`${facial.eye_shape} eyes`);

    if (features.length > 0) parts.push(`Features: ${features.join(', ')}`);

    // Face structure (all MediaPipe-derived).
    const faceStructure = [];
    if (facial.face_shape) faceStructure.push(`${facial.face_shape} face`);
    if (facial.jawline_type) faceStructure.push(`${facial.jawline_type} jawline`);
    if (facial.chin_type) faceStructure.push(`${facial.chin_type} chin`);
    if (facial.cheekbone_prominence) faceStructure.push(`${facial.cheekbone_prominence} cheekbones`);
    if (facial.cheek_fullness) faceStructure.push(`${facial.cheek_fullness} cheeks`);
    if (facial.forehead_width) faceStructure.push(`${facial.forehead_width} forehead`);
    if (faceStructure.length > 0) parts.push(`Face: ${faceStructure.join(', ')}`);

    // Eye geometry (MediaPipe).
    const eyeDetails = [];
    if (facial.eye_depth) eyeDetails.push(`${facial.eye_depth} eyes`);
    if (facial.eye_spacing) eyeDetails.push(`${facial.eye_spacing} eyes`);
    if (facial.eye_size) eyeDetails.push(`${facial.eye_size} eyes`);
    if (eyeDetails.length > 0) parts.push(`Eye details: ${eyeDetails.join(', ')}`);

    // Eyebrow geometry (MediaPipe).
    const eyebrowDetails = [];
    if (facial.eyebrow_shape) eyebrowDetails.push(`${facial.eyebrow_shape} eyebrows`);
    if (facial.eyebrow_thickness) eyebrowDetails.push(`${facial.eyebrow_thickness} eyebrows`);
    if (facial.eyebrow_arch_height) eyebrowDetails.push(`${facial.eyebrow_arch_height} arch eyebrows`);
    if (facial.possible_unibrow) eyebrowDetails.push('possible unibrow');
    if (eyebrowDetails.length > 0) parts.push(`Eyebrows: ${eyebrowDetails.join(', ')}`);

    // Nose geometry (MediaPipe).
    const noseDetails = [];
    if (facial.nose_shape) noseDetails.push(`${facial.nose_shape} nose`);
    if (facial.nose_bridge) noseDetails.push(`${facial.nose_bridge} bridge`);
    if (facial.nose_tip_shape) noseDetails.push(`${facial.nose_tip_shape} tip`);
    if (facial.nostril_width) noseDetails.push(`${facial.nostril_width} nostrils`);
    if (noseDetails.length > 0) parts.push(`Nose: ${noseDetails.join(', ')}`);

    // Lips & mouth (MediaPipe + ColorAnalyzer lip shade).
    const mouthDetails = [];
    if (facial.lip_fullness) mouthDetails.push(`${facial.lip_fullness} lips`);
    if (facial.lip_balance) mouthDetails.push(`${facial.lip_balance} lips`);
    if (facial.mouth_width) mouthDetails.push(`${facial.mouth_width} mouth`);
    if (facial.cupids_bow) mouthDetails.push(`${facial.cupids_bow} cupid's bow`);
    if (facial.lip_color?.shade && facial.lip_color.shade !== 'unknown') {
      mouthDetails.push(`${facial.lip_color.shade} lips`);
    }
    if (mouthDetails.length > 0) parts.push(`Mouth: ${mouthDetails.join(', ')}`);

    // Skin (ColorAnalyzer + SegFormer-mask OpenCV stats).
    const skinDetails = [];
    if (facial.skin_tone?.fitzpatrick && facial.skin_tone.fitzpatrick !== 'unknown') {
      skinDetails.push(facial.skin_tone.fitzpatrick);
    }
    if (facial.skin_undertone && facial.skin_undertone !== 'unknown') {
      skinDetails.push(`${facial.skin_undertone} undertone`);
    }
    if (facial.wrinkle_level && facial.wrinkle_level !== 'unknown') {
      skinDetails.push(`${facial.wrinkle_level} wrinkles`);
    }
    if (facial.freckles_or_moles && facial.freckles_or_moles !== 'unknown') {
      skinDetails.push(`${facial.freckles_or_moles} freckles/moles`);
    }
    if (skinDetails.length > 0) parts.push(`Skin: ${skinDetails.join(', ')}`);

    // Misc heuristics still worth indexing.
    const otherFeatures = [];
    if (facial.possible_dimples) otherFeatures.push('dimples');
    if (otherFeatures.length > 0) parts.push(`Other: ${otherFeatures.join(', ')}`);

    // Emotion + mood (HSEmotion).
    if (facial.primary_emotion) parts.push(`Emotion: ${facial.primary_emotion}`);
    if (facial.mood) parts.push(`Mood: ${facial.mood}`);
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
