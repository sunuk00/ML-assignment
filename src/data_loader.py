import os
import numpy as np
import pandas as pd


def load_data(name: str, data_dir: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    """
    CSV 파일을 로드하여 (feature DataFrame, label 배열)로 반환합니다.

    Parameters
    ----------
    name     : 'train' | 'val' | 'test_public' | 'test_hidden_no_labels'
    data_dir : CSV 파일이 있는 디렉토리 경로

    Returns
    -------
    df     : t 컬럼 + x_ feature 컬럼 (label은 분리됨)
    labels : 0/1 정수 배열, 라벨이 없는 파일이면 None

    Examples
    --------
    >>> train_df, _           = load_data('train',                DATA_DIR)
    >>> val_df,   val_labels  = load_data('val',                  DATA_DIR)
    >>> test_df,  test_labels = load_data('test_public',          DATA_DIR)
    >>> hidden_df, _          = load_data('test_hidden_no_labels', DATA_DIR)
    """
    raw = pd.read_csv(os.path.join(data_dir, f"{name}.csv"))
    if "label" in raw.columns:
        return raw.drop(columns=["label"]), raw["label"].to_numpy().astype(int)
    return raw, None
