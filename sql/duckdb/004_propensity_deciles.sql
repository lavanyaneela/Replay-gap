-- Reward against how confident the logging policy was in the slate it served.
-- This is the confounder in one table: the sessions a replay keeps are drawn from the top
-- deciles, and the top deciles earn more reward whatever the candidate policy would have done.
WITH ranked AS (
    SELECT
        session_id,
        log_propensity,
        reward,
        NTILE(10) OVER (ORDER BY log_propensity) AS decile
    FROM sessions
)
SELECT
    decile                            AS decile,
    COUNT(*)                          AS sessions,
    MIN(EXP(log_propensity))          AS min_propensity,
    MEDIAN(EXP(log_propensity))       AS median_propensity,
    MAX(EXP(log_propensity))          AS max_propensity,
    SUM(reward) / COUNT(*)            AS mean_reward
FROM ranked
GROUP BY decile
ORDER BY decile;
