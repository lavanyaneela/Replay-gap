-- Items the retrieval stage offered, against items the logging policy ever placed first.
SELECT
    (SELECT COUNT(DISTINCT item_id) FROM candidates)                        AS retrieved_items,
    (SELECT COUNT(DISTINCT item_id) FROM impressions)                       AS impressed_items,
    (SELECT COUNT(DISTINCT item_id) FROM impressions WHERE position = 0)    AS first_slot_items;
