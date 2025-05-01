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
    records = []
    # Iterate through each row and level to build long-form data
    for row in df.itertuples(index=False):
        timestamp = row.ts_event
        symbol = row.symbol
        for lvl in range(1, levels + 1):
            # Bid side
            price_b = getattr(row, f'bid_px_{lvl:02d}')
            size_b = getattr(row, f'bid_sz_{lvl:02d}')
            count_b = getattr(row, f'bid_ct_{lvl:02d}')
            records.append({
                'ts_event': timestamp,
                'symbol': symbol,
                'side': 'bid',
                'level': lvl,
                'price': price_b,
                'size': size_b,
                'count': count_b
            })
            # Ask side
            price_a = getattr(row, f'ask_px_{lvl:02d}')
            size_a = getattr(row, f'ask_sz_{lvl:02d}')
            count_a = getattr(row, f'ask_ct_{lvl:02d}')
            records.append({
                'ts_event': timestamp,
                'symbol': symbol,
                'side': 'ask',
                'level': lvl,
                'price': price_a,
                'size': size_a,
                'count': count_a
            })
    longDf = pd.DataFrame.from_records(records)
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
