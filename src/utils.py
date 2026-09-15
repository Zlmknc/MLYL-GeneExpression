## Veri okuma, etiketleme ve dosya I/O operasyonlarını barındırır.

import pandas as pd
from typing import Tuple, List, Set

def load_periodic_genes(filepath: str) -> Set[str]:
    """Spellman referans periyodik gen listesini yükler."""
    df = pd.read_csv(filepath, header=None, names=["ORF"])
    return set(df["ORF"].str.strip().tolist())

def prepare_labeled_dataset(
    raw_csv_path: str,
    periodic_genes_path: str,
    output_path: str = None
) -> pd.DataFrame:
    """Ham veriyi okur, ilk sütundaki ORF isimlerine göre etiketler ve kaydeder."""
    periodic_genes = load_periodic_genes(periodic_genes_path)
    df = pd.read_csv(raw_csv_path)
    
    orf_col = df.columns[0]
    df["label"] = df[orf_col].isin(periodic_genes).astype(int)
    
    if output_path:
        df.to_csv(output_path, index=False)
    return df

def get_feature_and_target_matrices(df: pd.DataFrame) -> Tuple[pd.DataFrame, list, pd.Series]:
    """Veriyi gen isimleri, zaman serisi kolonları ve etiket olarak ayırır."""
    orf_col = df.columns[0]
    time_cols = [col for col in df.columns if col not in [orf_col, "label"]]
    X_raw = df[time_cols].values.astype(float)
    y = df["label"].values
    return X_raw, time_cols, y