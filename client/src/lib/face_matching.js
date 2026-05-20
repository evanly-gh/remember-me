/**
 * Face-matching utility.
 *
 * The face-service returns a 512-d ArcFace embedding for every analysed
 * photo (`facial_details.data.face_embedding`). This module stores it
 * on the contact row and finds existing contacts whose embedding is
 * within a cosine-similarity threshold of a new photo.
 *
 * Backed by the `people.face_embedding` column (pgvector, 512-d). See
 * `client/src/lib/face_matching_schema.sql` for the migration that
 * adds that column + index.
 */

import { supabase } from './supabase';

/**
 * Threshold above which two face embeddings are considered the same
 * person. InsightFace ArcFace gives 99.83% LFW accuracy at 0.5; 0.55
 * is slightly stricter so the auto-merge prompt only fires on
 * confident matches.
 */
export const FACE_MATCH_THRESHOLD = 0.55;

/**
 * Find existing contacts with a face embedding close to `embedding`.
 *
 * Uses pgvector's `<=>` operator (cosine distance). Note that
 * `cosine_distance = 1 - cosine_similarity`, so we filter on
 * `<=> < (1 - threshold)`.
 *
 * @param {number[]} embedding - 512-d face embedding
 * @param {string} userId - only search this user's contacts
 * @param {number} threshold - cosine similarity threshold (default 0.55)
 * @returns {Promise<Array>} matching contacts, sorted by similarity desc
 */
export async function findSimilarFaces(embedding, userId, threshold = FACE_MATCH_THRESHOLD) {
  if (!Array.isArray(embedding) || embedding.length !== 512) {
    return [];
  }

  // pgvector accepts both array literals and the bracket string syntax;
  // supabase-js sends the array fine when the column type is `vector`.
  const distanceLimit = 1 - threshold;

  // Calls a Postgres RPC defined in the SQL migration that wraps the
  // vector search. We keep this server-side so the threshold logic
  // stays consistent across clients and the cosine math runs in the
  // database.
  const { data, error } = await supabase.rpc('find_similar_faces', {
    query_embedding: embedding,
    match_user_id: userId,
    max_distance: distanceLimit,
  });

  if (error) {
    console.error('[face_matching] RPC failed:', error.message);
    return [];
  }
  return data || [];
}

/**
 * Cosine similarity between two arrays. Useful for client-side
 * sanity checks; production matching goes through the RPC above.
 *
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number} similarity in [-1, 1]
 */
export function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, ma = 0, mb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    ma += a[i] * a[i];
    mb += b[i] * b[i];
  }
  if (ma === 0 || mb === 0) return 0;
  return dot / (Math.sqrt(ma) * Math.sqrt(mb));
}
