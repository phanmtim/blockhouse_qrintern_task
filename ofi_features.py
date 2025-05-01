"""
ofi_features.py

Compute Order Flow Imbalance (OFI) features: Best-Level, Multi-Level, Integrated, and Cross-Asset.
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import List, Dict


def compute_level_ofi(df: pd.DataFrame, level: int) -> pd.Series:
    """
    Compute instantaneous OFI at a given LOB level (m).

    OFI_m,n = OF_m,b_n + OF_m,a_n, where:
      OF_m,b_n = { +q_m,b_n if price up
                   q_m,b_n - q_m,b_{n-1} if price same
                   -q_m,b_{n-1} if price down }
      OF_m,a_n = { -q_m,a_n if price up
                   q_m,a_n - q_m,a_{n-1} if price same
                   +q_m,a_n if price down }

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed wide-format DF with bid_px_{:02d}, bid_sz_{:02d},
        ask_px_{:02d}, ask_sz_{:02d} for each level.
    level : int
        LOB level (1-based index).

    Returns
    -------
    pd.Series
        Instantaneous OFI for that level, aligned with df index.
    """
    # Bid side
    px_b = df[f'bid_px_{level:02d}']
    sz_b = df[f'bid_sz_{level:02d}']
    px_b_prev = px_b.shift(1)
    sz_b_prev = sz_b.shift(1)
    ofi_b = pd.Series(0, index=df.index)
    inc = px_b > px_b_prev
    same = px_b == px_b_prev
    dec = px_b < px_b_prev
    ofi_b.loc[inc] = sz_b.loc[inc]
    ofi_b.loc[same] = sz_b.loc[same] - sz_b_prev.loc[same]
    ofi_b.loc[dec] = -sz_b_prev.loc[dec]

    # Ask side
    px_a = df[f'ask_px_{level:02d}']
    sz_a = df[f'ask_sz_{level:02d}']
    px_a_prev = px_a.shift(1)
    sz_a_prev = sz_a.shift(1)
    ofi_a = pd.Series(0, index=df.index)
    inc_a = px_a > px_a_prev
    same_a = px_a == px_a_prev
    dec_a = px_a < px_a_prev
    ofi_a.loc[inc_a] = -sz_a.loc[inc_a]
    ofi_a.loc[same_a] = sz_a.loc[same_a] - sz_a_prev.loc[same_a]
    ofi_a.loc[dec_a] = sz_a.loc[dec_a]

    return ofi_b + ofi_a


def compute_all_levels_ofi(df: pd.DataFrame, levels: int = 9) -> pd.DataFrame:
    """
    Compute OFI for all levels from 1 to `levels`.

    Returns a DataFrame with columns ['ofi_1', ..., f'ofi_{levels}'].
    """
    ofi_dict: Dict[str, pd.Series] = {}
    for lvl in range(1, levels + 1):
        ofi_dict[f'ofi_{lvl}'] = compute_level_ofi(df, lvl)
    return pd.DataFrame(ofi_dict, index=df.index)


def compute_best_level_ofi(df: pd.DataFrame) -> pd.Series:
    """
    Best-Level OFI (level 1).
    """
    return compute_level_ofi(df, 1)


def compute_multi_level_ofi(df: pd.DataFrame, levels: int = 9) -> pd.Series:
    """
    Multi-Level OFI: sum of OFIs across levels 1 to `levels`.
    """
    ofi_mat = compute_all_levels_ofi(df, levels)
    return ofi_mat.sum(axis=1)


def compute_integrated_ofi(df: pd.DataFrame, levels: int = 9) -> pd.Series:
    """
    Integrated OFI: first principal component of multi-level OFIs,
    normalized so component weights sum to 1 (L1 norm).
    """
    ofi_mat = compute_all_levels_ofi(df, levels)
    pca = PCA(n_components=1)
    pca.fit(ofi_mat.fillna(0))
    weights = pca.components_[0]
    norm_weights = weights / np.linalg.norm(weights, ord=1)
    integrated = ofi_mat.values.dot(norm_weights)
    return pd.Series(integrated, index=df.index, name='integrated_ofi')


def compute_cross_asset_ofi(ofi_df: pd.DataFrame,
                            ofi_col: str = 'integrated_ofi',
                            symbol_col: str = 'symbol') -> pd.DataFrame:
    """
    Cross-Asset OFI: for each symbol and timestamp, sum of OFI of all other assets.

    Parameters
    ----------
    ofi_df : pd.DataFrame
        Long-form DataFrame with columns ['ts_event', symbol_col, ofi_col].
    ofi_col : str
        Column name for the OFI values (default 'integrated_ofi').
    symbol_col : str
        Column name for asset symbol (default 'symbol').

    Returns
    -------
    pd.DataFrame
        DataFrame with ['ts_event', symbol_col, 'cross_asset_ofi'].
    """
    pivot = ofi_df.pivot(index='ts_event', columns=symbol_col, values=ofi_col)
    cross = pivot.apply(lambda row: row.sum() - row)
    cross_df = cross.stack().rename('cross_asset_ofi').reset_index()
    return cross_df
