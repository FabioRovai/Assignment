from typing import Union

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_column(col: Union[int, float, pd.Series], le: LabelEncoder) -> Union[int, float, pd.Series]:
    """
    Encode a column of data using a label encoder if it is not already an integer or float type.

    Parameters:
    col (Union[int, float, pd.Series]): The column to be encoded.
    le (LabelEncoder): The label encoder to use for encoding the column.

    Returns:
    Union[int, float, pd.Series]: The encoded column.
    """
    if col.dtype not in (int, float):
        le.fit(col)
        col = le.transform(col)
    return col
