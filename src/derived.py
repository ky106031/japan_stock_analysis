# src/derived.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class DerivedResult:
    derived: pd.DataFrame


def _to_fy(year_col: pd.Series) -> pd.Series:
    """
    PeriodDataの「年度」は日付っぽい値（例: 2008-03-01）が入っている前提。
    FYは期末年（year）として int を作る。
    """
    dt = pd.to_datetime(year_col, errors="coerce")
    return dt.dt.year


def _to_numeric_with_nan(series: pd.Series) -> pd.Series:
    """
    "赤字", "-", "" など文字混在を NaN にして数値化する。
    """
    # 代表的な非数値表記を NaN に寄せる（必要なら増やす）
    s = series.replace(
        {
            "赤字": None,
            "－": None,
            "―": None,
            "-": None,
            "": None,
            " ": None,
        }
    )
    return pd.to_numeric(s, errors="coerce")


def build_derived(period_df: pd.DataFrame) -> DerivedResult:
    """
    仕様（プロトタイプ）:
    - DerivedはPeriodDataを基本そのままコピー
    - FY列を追加
    - 配当性向_％ を数値化した列（配当性向_％_num）を追加
    - 他の列は今は触らない（必要になったら後で拡張）
    """
    df = period_df.copy()

    # FY追加（期末年）
    if "年度" not in df.columns:
        raise KeyError("PeriodData に '年度' 列が見つかりません。")
    df["FY"] = _to_fy(df["年度"])

    # 配当性向の数値化（赤字→NaN）
    if "配当性向_％" in df.columns:
        df["配当性向_％_num"] = _to_numeric_with_nan(df["配当性向_％"])
    else:
        # もし列名が違う場合に備えて、わかりやすい例外
        raise KeyError("PeriodData に '配当性向_％' 列が見つかりません。列名を確認してください。")

    return DerivedResult(derived=df)