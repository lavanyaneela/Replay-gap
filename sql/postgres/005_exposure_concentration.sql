-- How concentrated first slot exposure is across the catalogue.
-- Same window frame as the DuckDB file, with the numeric cast the cumulative share needs here.
WITH first_slot AS (
    SELECT item_id, COUNT(*) AS impressions
    FROM impressions
    WHERE position = 0
    GROUP BY item_id
),
ordered AS (
    SELECT
        item_id,
        impressions,
        SUM(impressions) OVER (
            ORDER BY impressions DESC, item_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_total,
        SUM(impressions) OVER () AS total
    FROM first_slot
)
SELECT
    item_id                            AS item_id,
    impressions                        AS impressions,
    running_total::numeric / total     AS cumulative_share
FROM ordered
ORDER BY impressions DESC, item_id;
