-- How concentrated first slot exposure is across the catalogue.
-- The cumulative share column answers the question an importance weight cannot: for the items a
-- candidate policy wants to promote, is there any first slot data in the log at all.
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
    item_id                      AS item_id,
    impressions                  AS impressions,
    running_total / total        AS cumulative_share
FROM ordered
ORDER BY impressions DESC, item_id;
