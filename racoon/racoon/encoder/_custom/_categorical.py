import warnings
from typing import List, Union

import pandas as pd
import category_encoders
from xfeat.base import TransformerMixin
from xfeat.types import XDataFrame

class OneHotEncoder(TransformerMixin):
    def __init__(self, input_cols:List[str]):
        self.input_cols = input_cols
        self.oh = {col: category_encoders.OneHotEncoder() for col in input_cols}

    def fit(self, input_df: XDataFrame, y=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for col in self.input_cols:
                self.oh[col].fit(input_df[col])
            
            return self

    def transform(self, input_df: XDataFrame):
        _df = []
        for col in self.input_cols:
            new_df = input_df[[col]].copy()
            oh_df = self.oh[col].transform(new_df).add_suffix('_oh')
            new_df = pd.concat([new_df, oh_df], axis=1)

            _df.append(new_df)

        return pd.concat(_df, axis=1)

    def fit_transform(self, input_df: XDataFrame, y=None):
        self.fit(input_df[self.input_cols])
        return self.transform(input_df)