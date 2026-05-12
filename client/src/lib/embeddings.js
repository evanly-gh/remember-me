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
    const faceStructure = [];
    if (facial.face_shape) {
      faceStructure.push(facial.face_shape);
    }
    if (facial.jawline_type) {
      faceStructure.push(`${facial.jawline_type} jawline`);
    }
    if (facial.chin_type) {
      faceStructure.push(`${facial.chin_type} chin`);
    }
    if (facial.cheekbone_prominence) {
      faceStructure.push(`${facial.cheekbone_prominence} cheekbones`);
    }
    if (facial.cheek_fullness) {
      faceStructure.push(`${facial.cheek_fullness} cheeks`);
    }
    if (facial.forehead_width) {
      faceStructure.push(`${facial.forehead_width} forehead`);
    }
    if (faceStructure.length > 0) {
      parts.push(`Face: ${faceStructure.join(', ')}`);
    }

    // Detailed eye features
    const eyeDetails = [];
    if (facial.eye_depth) {
      eyeDetails.push(`${facial.eye_depth} eyes`);
    }
    if (facial.eye_spacing) {
      eyeDetails.push(`${facial.eye_spacing} eyes`);
    }
    if (facial.eye_size) {
      eyeDetails.push(`${facial.eye_size} eyes`);
    }
    if (facial.narrow_eyes) {
      eyeDetails.push('narrow eyes');
    }
    if (eyeDetails.length > 0) {
      parts.push(`Eye details: ${eyeDetails.join(', ')}`);
    }

    // Eyebrow features
    const eyebrowDetails = [];
    if (facial.eyebrow_shape) {
      eyebrowDetails.push(`${facial.eyebrow_shape} eyebrows`);
    }
    if (facial.eyebrow_thickness) {
      eyebrowDetails.push(`${facial.eyebrow_thickness} eyebrows`);
    }
    if (facial.eyebrow_arch_height) {
      eyebrowDetails.push(`${facial.eyebrow_arch_height} arch eyebrows`);
    }
    if (facial.bushy_eyebrows) {
      eyebrowDetails.push('bushy eyebrows');
    }
    if (facial.arched_eyebrows) {
      eyebrowDetails.push('arched eyebrows');
    }
    if (eyebrowDetails.length > 0) {
      parts.push(`Eyebrows: ${eyebrowDetails.join(', ')}`);
    }

    // Nose features
    const noseDetails = [];
    if (facial.nose_shape) {
      noseDetails.push(`${facial.nose_shape} nose`);
    }
    if (facial.nose_bridge) {
      noseDetails.push(`${facial.nose_bridge} bridge`);
    }
    if (facial.nose_tip_shape) {
      noseDetails.push(`${facial.nose_tip_shape} tip`);
    }
    if (facial.nostril_width) {
      noseDetails.push(`${facial.nostril_width} nostrils`);
    }
    if (facial.big_nose) {
      noseDetails.push('big nose');
    }
    if (facial.pointy_nose) {
      noseDetails.push('pointy nose');
    }
    if (noseDetails.length > 0) {
      parts.push(`Nose: ${noseDetails.join(', ')}`);
    }

    // Lip and mouth features
    const mouthDetails = [];
    if (facial.lip_fullness) {
      mouthDetails.push(`${facial.lip_fullness} lips`);
    }
    if (facial.lip_balance) {
      mouthDetails.push(`${facial.lip_balance} lips`);
    }
    if (facial.mouth_width) {
      mouthDetails.push(`${facial.mouth_width} mouth`);
    }
    if (facial.cupids_bow) {
      mouthDetails.push(`${facial.cupids_bow} cupid's bow`);
    }
    if (facial.big_lips) {
      mouthDetails.push('big lips');
    }
    if (facial.wearing_lipstick) {
      mouthDetails.push('wearing lipstick');
    }
    if (mouthDetails.length > 0) {
      parts.push(`Mouth: ${mouthDetails.join(', ')}`);
    }

    // Facial hair details
    const facialHair = [];
    if (facial.mustache || facial.facial_hair?.mustache) {
      facialHair.push('mustache');
    }
    if (facial.goatee || facial.facial_hair?.goatee) {
      facialHair.push('goatee');
    }
    if (facial.sideburns || facial.facial_hair?.sideburns) {
      facialHair.push('sideburns');
    }
    if (facial.facial_hair?.['5_o_clock_shadow']) {
      facialHair.push('stubble');
    }
    if (facial.facial_hair?.full_beard) {
      facialHair.push('full beard');
    }
    if (facialHair.length > 0) {
      parts.push(`Facial hair: ${facialHair.join(', ')}`);
    }

    // Hair texture
    const hairDetails = [];
    if (facial.hair_texture || facial.hair_texture_celeba) {
      const texture = facial.hair_texture || facial.hair_texture_celeba;
      hairDetails.push(`${texture} hair`);
    }
    if (facial.receding_hairline) {
      hairDetails.push('receding hairline');
    }
    if (facial.has_bangs) {
      hairDetails.push('has bangs');
    }
    if (hairDetails.length > 0) {
      parts.push(`Hair texture: ${hairDetails.join(', ')}`);
    }

    // Skin features
    const skinDetails = [];
    if (facial.skin_tone?.fitzpatrick) {
      skinDetails.push(facial.skin_tone.fitzpatrick);
    }
    if (facial.skin_undertone) {
      skinDetails.push(`${facial.skin_undertone} undertone`);
    }
    if (facial.wrinkle_level) {
      skinDetails.push(`${facial.wrinkle_level} wrinkles`);
    }
    if (facial.freckles_or_moles) {
      skinDetails.push(`${facial.freckles_or_moles} freckles/moles`);
    }
    if (facial.pale_skin) {
      skinDetails.push('pale skin');
    }
    if (facial.rosy_cheeks) {
      skinDetails.push('rosy cheeks');
    }
    if (skinDetails.length > 0) {
      parts.push(`Skin: ${skinDetails.join(', ')}`);
    }

    // Additional features
    const otherFeatures = [];
    if (facial.bags_under_eyes) {
      otherFeatures.push('bags under eyes');
    }
    if (facial.double_chin) {
      otherFeatures.push('double chin');
    }
    if (facial.chubby) {
      otherFeatures.push('chubby face');
    }
    if (facial.high_cheekbones) {
      otherFeatures.push('high cheekbones');
    }
    if (facial.oval_face_celeba) {
      otherFeatures.push('oval face');
    }
    if (facial.possible_dimples) {
      otherFeatures.push('dimples');
    }
    if (otherFeatures.length > 0) {
      parts.push(`Other features: ${otherFeatures.join(', ')}`);
    }

    // Accessories
    const accessories = [];
    if (facial.wearing_necklace || facial.necklace_detected) {
      accessories.push('necklace');
    }
    if (facial.wearing_necktie) {
      accessories.push('necktie');
    }
    if (accessories.length > 0) {
      parts.push(`Accessories: ${accessories.join(', ')}`);
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
