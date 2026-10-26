-- Reward against how confident the logging policy was in the slate it served.
-- Two dialect differences here: the numeric cast on the mean, and MEDIAN, which DuckDB has as an
-- aggregate and PostgreSQL only offers as an ordered set aggregate.
WITH ranked AS (
    SELECT
        session_id,
        log_propensity,
        reward,
        NTILE(10) OVER (ORDER BY log_propensity) AS decile
    FROM sessions
)
SELECT
    decile                                                                   AS decile,
    COUNT(*)                                                                 AS sessions,
    MIN(EXP(log_propensity))                                                 AS min_propensity,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXP(log_propensity))         AS median_propensity,
    MAX(EXP(log_propensity))                                                 AS max_propensity,
    SUM(reward)::numeric / COUNT(*)                                          AS mean_reward
FROM ranked
GROUP BY decile
ORDER BY decile;
