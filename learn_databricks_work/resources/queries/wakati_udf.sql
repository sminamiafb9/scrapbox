CREATE OR REPLACE FUNCTION sandbox.shotaro_minami8e40a5.wakati(text STRING)
RETURNS STRING
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

def handler_function(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    for text_series in batch_iter:
        tagger = Tagger('-Owakati')
        yield text_series.apply(
            lambda text: tagger.parse(text) if pd.notna(text) else None
        )
$$;