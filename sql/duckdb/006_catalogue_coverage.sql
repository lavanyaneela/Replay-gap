-- Items the retrieval stage offered, against items the logging policy ever placed first.
-- The gap is the part of the action space the log is silent about.
SELECT
    (SELECT COUNT(DISTINCT item_id) FROM candidates)                        AS retrieved_items,
    (SELECT COUNT(DISTINCT item_id) FROM impressions)                       AS impressed_items,
    (SELECT COUNT(DISTINCT item_id) FROM impressions WHERE position = 0)    AS first_slot_items;
