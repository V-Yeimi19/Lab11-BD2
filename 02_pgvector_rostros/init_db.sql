-- Inicialización de la base del laboratorio (PostgreSQL + pgvector).
-- Uso:
--   docker cp 02_pgvector_rostros\init_db.sql pg_yeimi:/tmp/init_db.sql
--   docker exec pg_yeimi psql -U postgres -f /tmp/init_db.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS face_embeddings (
    id        BIGSERIAL PRIMARY KEY,
    name      TEXT NOT NULL,
    path      TEXT NOT NULL UNIQUE,
    embedding vector(128) NOT NULL
);

-- La parte 2 usa búsqueda lineal (Seq Scan), por lo que no se necesitan
-- índices vectoriales. Para la parte 3 se pueden crear, por ejemplo:
--   CREATE INDEX face_embeddings_l2_idx
--       ON face_embeddings USING hnsw (embedding vector_l2_ops);
--   CREATE INDEX face_embeddings_cos_idx
--       ON face_embeddings USING hnsw (embedding vector_cosine_ops);
