from typing import List, Union

import pandas as pd
from xfeat.base import TransformerMixin
from xfeat.types import XDataFrame

class DateTimeEncoder(TransformerMixin):
    def __init__(self, input_cols: List[str]):
        self.input_cols = input_cols
        
    def fit(self, input_df: XDataFrame):
        return self
        
    def transform(self,  input_df: XDataFrame) -> XDataFrame:
        new_df = input_df.copy()
        
        for col in self.input_cols:
            new_df[col] = pd.to_datetime(input_df[col])
            
        return new_df[self.input_cols]

class DateTimeTransformEncoder(TransformerMixin):
    def __init__(
        self,
        input_cols:List[str],
        to:List[str] = ['dayofweek', 'is_weekend', 'hour', 'minute', 'second'],
        ) -> None:
        self.input_cols = input_cols
        self.to = to

    def fit(self, input_df: XDataFrame):
        return self

    def transform(self, input_df: XDataFrame) -> XDataFrame:
        new_df = input_df[self.input_cols].copy()
        
        transformed_cols = []
        for col in self.input_cols:
            for t in self.to:
                col_name = f'{col}_{t}'
                transformed_cols.append(col_name)

                if t == 'is_weekend':
                    new_df[col_name] = (input_df[col].dt.dayofweek >= 5).astype(int)
                else:
                    new_df[col_name] = getattr(input_df[col].dt, t)

        return new_df
