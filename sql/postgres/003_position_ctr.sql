-- Click through rate by slate position, and each position's share of all clicks.
-- The casts are the dialect difference that matters. In DuckDB the division operator is always
-- floating point; in PostgreSQL, integer divided by integer truncates, so the unCAST version of
-- this query returns a click through rate of exactly 0 for every position and reports nothing
-- wrong while doing it.
SELECT
    position                                                           AS position,
    COUNT(*)                                                           AS impressions,
    SUM(clicked)                                                       AS clicks,
    SUM(clicked)::numeric / COUNT(*)                                   AS ctr,
    SUM(clicked)::numeric / SUM(SUM(clicked)) OVER ()                  AS share_of_clicks
FROM impressions
GROUP BY position
ORDER BY position;
