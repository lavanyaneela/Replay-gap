-- Recompute each session's reward from the impressions and compare it against the column the
-- generator wrote. Any row returned by this query is a corrupt fixture, which is exactly the
-- check a committed fixture needs and the one that is normally skipped.
SELECT
    s.session_id              AS session_id,
    s.reward                  AS recorded_reward,
    COALESCE(i.clicks, 0)     AS recomputed_reward
FROM sessions AS s
LEFT JOIN (
    SELECT session_id, SUM(clicked) AS clicks
    FROM impressions
    GROUP BY session_id
) AS i
    ON i.session_id = s.session_id
WHERE s.reward <> COALESCE(i.clicks, 0);
