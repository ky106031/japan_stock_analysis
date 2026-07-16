from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class DerivedResult:
    derived: pd.DataFrame


# 個別銘柄Excel / 統合DB のどちらでも最終的にこの列名へ寄せる
STANDARD_RENAME_MAP = {
    # 基本
    "証券コード": "証券コード",
    "年度": "年度",

    # P/L
    "売上高": "売上高",
    "売上高_億": "売上高",
    "営業利益": "営業利益",
    "営業利益_億": "営業利益",
    "経常利益": "経常利益",
    "経常利益_億": "経常利益",
    "純利益": "当期純利益",
    "純利益_億": "当期純利益",
    "当期純利益": "当期純利益",

    # 指標
    "EPS": "EPS",
    "EPS_円": "EPS",
    "ROE": "ROE",
    "ROE_％": "ROE",
    "ROA": "ROA",
    "ROA_％": "ROA",
    "営業利益率": "営業利益率",
    "営業利益率_％": "営業利益率",

    # B/S
    "総資産": "総資産",
    "総資産_億": "総資産",
    "純資産": "純資産",
    "純資産_億": "純資産",
    "自己資本比率": "自己資本比率",
    "自己資本比率_％": "自己資本比率",
    "BPS": "BPS",
    "BPS_円": "BPS",

    # CF / 現金
    "営業CF": "営業CF",
    "営業CF_億": "営業CF",
    "現金等": "現金等",
    "現金等_億": "現金等",

    # 配当
    "1株配当": "１株配当",
    "１株配当": "１株配当",
    "１株配当_円": "１株配当",
    "配当性向": "配当性向",
    "配当性向_％": "配当性向",
}


NUMERIC_COLS = [
    "売上高",
    "営業利益",
    "経常利益",
    "当期純利益",
    "EPS",
    "ROE",
    "ROA",
    "営業利益率",
    "総資産",
    "純資産",
    "自己資本比率",
    "BPS",
    "営業CF",
    "現金等",
    "１株配当",
    "配当性向",
]


def _to_fy(year_col: pd.Series) -> pd.Series:
    """
    「年度」列から FY（期末年）を作る。
    例:
      2024-03-31 -> 2024
      2024/03    -> 2024
    """
    dt = pd.to_datetime(year_col, errors="coerce")
    return dt.dt.year


def _to_numeric_with_nan(series: pd.Series) -> pd.Series:
    """
    "赤字", "-", "" などを NaN に寄せて数値化する。
    """
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


def normalize_period_df(period_df: pd.DataFrame) -> pd.DataFrame:
    """
    個別銘柄Excelの「決算情報」または統合DBの PeriodData を、
    表示・派生計算しやすい標準列名へ寄せる。

    方針:
    - カラム名から単位を外す
    - 数値はそのまま保持（円・%）
    - 「配当性向_％」のような旧列名も吸収する
    """
    df = period_df.copy()

    # 列名の標準化
    rename_map = {col: STANDARD_RENAME_MAP[col] for col in df.columns if col in STANDARD_RENAME_MAP}
    df = df.rename(columns=rename_map)

    # 必須列チェック
    if "年度" not in df.columns:
        raise KeyError("決算情報またはPeriodData に '年度' 列が見つかりません。")

    # 証券コードがなければ空で追加（個別ファイルで不足していても落ちないように）
    if "証券コード" not in df.columns:
        df["証券コード"] = pd.NA

    # 数値化
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = _to_numeric_with_nan(df[col])

    return df


def build_derived(period_df: pd.DataFrame) -> DerivedResult:
    """
    標準化後のPeriodDataからDerivedを生成する。
    出力は「単位なし列名」のデータ構造。
    """
    df = normalize_period_df(period_df).copy()

    # FY追加
    df["FY"] = _to_fy(df["年度"])

    # 配当性向の数値列
    if "配当性向" in df.columns:
        df["配当性向_num"] = _to_numeric_with_nan(df["配当性向"])
    else:
        raise KeyError("決算情報またはPeriodData に '配当性向' 列が見つかりません。")

    # 証券コードを文字列寄せ
    df["証券コード"] = pd.to_numeric(df["証券コード"], errors="coerce").astype("Int64").astype(str)

    return DerivedResult(derived=df)


def update_single_excel_derived(
    file_path: str | Path,
    period_sheet: str = "決算情報",
    derived_sheet: str = "Derived",
) -> pd.DataFrame:
    """
    個別銘柄Excelの「決算情報」から Derived を作り、
    同じExcelに Derived シートとして保存する。
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    period = pd.read_excel(file_path, sheet_name=period_sheet)
    derived = build_derived(period).derived

    with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        derived.to_excel(writer, sheet_name=derived_sheet, index=False)

    return derived