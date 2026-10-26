-- DuckDB schema for the logged interaction stream.
-- Three tables, because the three things a production log carries have different grains: one row
-- per session, one row per retrieved candidate, one row per impression.

CREATE TABLE IF NOT EXISTS sessions (
    session_id     BIGINT  NOT NULL,
    context_id     BIGINT  NOT NULL,
    log_propensity DOUBLE  NOT NULL,
    reward         BIGINT  NOT NULL
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
