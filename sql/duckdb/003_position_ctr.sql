-- Click through rate by slate position, and each position's share of all clicks.
-- The share column is a window function over the grouped result, which is the shape that makes
-- position bias legible: the first slot takes most of the clicks whatever is in it.
SELECT
    position                                                   AS position,
    COUNT(*)                                                   AS impressions,
    SUM(clicked)                                               AS clicks,
    SUM(clicked) / COUNT(*)                                    AS ctr,
    SUM(clicked) / SUM(SUM(clicked)) OVER ()                   AS share_of_clicks
FROM impressions
GROUP BY position
ORDER BY position;
