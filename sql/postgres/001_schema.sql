-- PostgreSQL schema for the logged interaction stream.
-- Same three grains as the DuckDB file. The types differ where the dialects differ: DuckDB's
-- DOUBLE is DOUBLE PRECISION here, and SMALLINT means the same thing in both.

CREATE TABLE IF NOT EXISTS sessions (
    session_id     BIGINT           NOT NULL,
    context_id     BIGINT           NOT NULL,
    log_propensity DOUBLE PRECISION NOT NULL,
    reward         BIGINT           NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
    session_id BIGINT  NOT NULL,
    slot       INTEGER NOT NULL,
    item_id    BIGINT  NOT NULL
);

CREATE TABLE IF NOT EXISTS impressions (
    session_id BIGINT   NOT NULL,
    position   INTEGER  NOT NULL,
    item_id    BIGINT   NOT NULL,
    clicked    SMALLINT NOT NULL
);
