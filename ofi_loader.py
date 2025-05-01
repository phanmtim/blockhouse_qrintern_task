"""
ofi_loader.py

Module for loading and preprocessing order book data for Order Flow Imbalance (OFI) feature construction.
"""
import pandas as pd
from typing import Optional, List


def loadData(filePath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load raw order book data from a CSV file, parsing timestamp columns.

    Parameters
    ----------
    filePath : str
        Path to the CSV file.
    nrows : int, optional
        Number of rows to read (for sampling/debug). Default is None (all rows).

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed `ts_recv` and `ts_event` columns.
    """
    df = pd.read_csv(
        filePath,
        parse_dates=['ts_recv', 'ts_event'],
        nrows=nrows
    )
    return df


def preprocessData(
    df: pd.DataFrame,
    filterRtypes: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Preprocess raw order book DataFrame:
      - Optionally filter by record types (rtype).
      - Sort by event timestamp.
      - Reset index.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from `loadData`.
    filterRtypes : list of int, optional
        List of `rtype` codes to keep (e.g. [1] for book updates).

    Returns
    -------
    pd.DataFrame
        Filtered and sorted DataFrame.
    """
    dfOut = df.copy()
    if filterRtypes is not None:
        dfOut = dfOut[dfOut['rtype'].isin(filterRtypes)]
    dfOut = dfOut.sort_values('ts_event').reset_index(drop=True)
    return dfOut


def reshapeBookData(
    df: pd.DataFrame,
    levels: int = 9
) -> pd.DataFrame:
    """
    Reshape wide-format order book levels into long-form DataFrame.

    Converts bid/ask price, size, and count columns for each level into:
      ['ts_event', 'symbol', 'side', 'level', 'price', 'size', 'count']

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame containing wide book levels.
    levels : int
        Number of price levels (default 9).

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame ready for OFI computations.
    """
    # Work on a copy to avoid side-effects
    dfCopy = df.reset_index().rename(columns={'index': 'recordId'})

    # Melt bids
    bidLong = pd.wide_to_long(
        dfCopy,
        stubnames=['bid_px', 'bid_sz', 'bid_ct'],
        i='recordId',
        j='level',
        sep='_',
        suffix='\d+'
    ).reset_index()
    bidLong['side'] = 'bid'
    bidLong = bidLong.rename(
        columns={'bid_px': 'price', 'bid_sz': 'size', 'bid_ct': 'count'}
    )

    # Melt asks
    askLong = pd.wide_to_long(
        dfCopy,
        stubnames=['ask_px', 'ask_sz', 'ask_ct'],
        i='recordId',
        j='level',
        sep='_',
        suffix='\d+'
    ).reset_index()
    askLong['side'] = 'ask'
    askLong = askLong.rename(
        columns={'ask_px': 'price', 'ask_sz': 'size', 'ask_ct': 'count'}
    )

    # Combine and restore timestamp & symbol
    longDf = pd.concat([bidLong, askLong], ignore_index=True)
    longDf = longDf.merge(
        dfCopy[['recordId', 'ts_event', 'symbol']],
        on='recordId',
        how='left'
    )
    # Reorder columns
    longDf = longDf[['ts_event', 'symbol', 'side', 'level', 'price', 'size', 'count']]

    return longDf


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Load and preprocess order book data for OFI features.'
    )
    parser.add_argument('inputFile', help='Path to first_25000_rows.csv')
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to read (optional)')
    parser.add_argument(
        '--filterRtypes',
        type=int,
        nargs='+',
        default=None,
        help='List of rtype codes to filter (e.g., 1 for updates)'
    )
    parser.add_argument(
        '--output',
        default='preprocessed_book.csv',
        help='Path to save preprocessed CSV'
    )
    args = parser.parse_args()

    # Load, preprocess, and save
    dfRaw = loadData(args.inputFile, args.nrows)
    dfPre = preprocessData(dfRaw, args.filterRtypes)
    dfPre.to_csv(args.output, index=False)
    print(f'Preprocessed data saved to {args.output}')