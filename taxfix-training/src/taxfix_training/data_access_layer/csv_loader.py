from pathlib import Path

import pandas as pd


def load_dataset(path: Path | str, **load_kwargs):
    return pd.read_csv(path, **load_kwargs)
