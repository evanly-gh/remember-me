-- Face matching schema migration.
--
-- Run this once in the Supabase SQL editor (or via `supabase db push`)
-- to enable cross-photo identity grouping.
--
-- After running, every new contact saved through the Record flow will
-- store its 512-d ArcFace embedding in `people.face_embedding`. The
-- `findSimilarFaces` JS helper in client/src/lib/face_matching.js
-- calls the `find_similar_faces` RPC defined here to look up nearby
-- embeddings.
--
-- This script is idempotent: re-running it is safe and just no-ops
-- on objects that already exist.

-- 1. pgvector extension (no-op if already enabled).
create extension if not exists vector;

-- 2. The face_embedding column. Nullable so existing rows stay valid
--    until they're re-analysed.
alter table public.people
  add column if not exists face_embedding vector(512);

-- 3. IVFFlat index for fast cosine-distance search.
--    `lists = 100` is a reasonable default for collections up to ~100k
--    rows; tune up as your dataset grows.
create index if not exists people_face_embedding_idx
  on public.people
  using ivfflat (face_embedding vector_cosine_ops)
  with (lists = 100);

-- 4. The RPC the client calls. Returns matching rows sorted by
--    cosine distance ascending (closer = better match), filtered to
--    the current user's contacts only and below the supplied
--    distance threshold.
--
--    Drop first because PostgreSQL won't let CREATE OR REPLACE change
--    a function's return type. If the previous attempt declared
--    `id uuid` and your `people.id` is `bigint`, you'd hit
--    "42P13: return type mismatch in function declared to return record".
drop function if exists public.find_similar_faces(vector(512), uuid, float);

create function public.find_similar_faces(
  query_embedding vector(512),
  match_user_id uuid,
  max_distance float default 0.45  -- cosine distance, ~ similarity >= 0.55
)
returns table (
  id bigint,            -- people.id is bigint (serial), not uuid
  name text,
  photo_url text,
  similarity float
)
language sql stable as $$
  select
    p.id,
    p.name,
    p.photo_url,
    (1 - (p.face_embedding <=> query_embedding))::float as similarity
  from public.people p
  where p.user_id = match_user_id
    and p.face_embedding is not null
    and (p.face_embedding <=> query_embedding) < max_distance
  order by p.face_embedding <=> query_embedding asc
  limit 10;
$$;

-- 5. Grant RPC execute to authenticated users.
grant execute on function public.find_similar_faces(vector(512), uuid, float)
  to authenticated;
