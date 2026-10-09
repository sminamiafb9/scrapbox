%sql
CREATE OR REPLACE FUNCTION sandbox.shotaro_minami8e40a5.ma(text STRING)
RETURNS ARRAY<STRUCT<surface: STRING, base: STRING, pos: STRING>>
LANGUAGE PYTHON
PARAMETER STYLE PANDAS
HANDLER 'handler_function'
ENVIRONMENT (
  dependencies = '["fugashi", "unidic-lite"]',
  environment_version = '6'
)
AS $$
import pandas as pd
from typing import Iterator
from fugashi import Tagger

_tagger = Tagger()

def analyze(text):
    if pd.isna(text):
        return None

    result = []

    for word in _tagger(text):
        result.append({
            "surface": word.surface,
            "base": word.feature.orthBase,
            "pos": word.feature.pos1,
        })

    return result

def handler_function(
    batch_iter: Iterator[pd.Series]
) -> Iterator[pd.Series]:
    for text_series in batch_iter:
        yield text_series.apply(analyze)
$$;