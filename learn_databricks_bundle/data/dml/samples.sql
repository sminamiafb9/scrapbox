WITH ranked AS (
    SELECT
        sentence,
        dataset_split,
        label_3class,
        ROW_NUMBER() OVER (
            PARTITION BY dataset_split
            ORDER BY RAND(42)
        ) AS rn
    FROM sandbox.shotaro_minami8e40a5.wrime_gold_sentiment_3class
)

SELECT
    sentence,
    dataset_split,
    label_3class
FROM ranked
WHERE
    (dataset_split = 'train' AND rn <= 3000)
    OR (dataset_split = 'dev' AND rn <= 300)
    OR (dataset_split = 'test' AND rn <= 300)
