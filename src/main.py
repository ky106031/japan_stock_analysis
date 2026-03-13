# src/main.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from derived import build_derived

# プロジェクト直下
BASE_DIR = Path(__file__).resolve().parent.parent

# 入力Excel（あなたのdata配下にあるファイル名に合わせてください）
INPUT_EXCEL = BASE_DIR / "data" / "japan_stock_dividend_db.xlsx"

# 出力Excel（上書きが怖いので、まずは別名出力を推奨）
OUTPUT_EXCEL = BASE_DIR / "data" / "japan_stock_dividend_db_out.xlsx"


def main() -> None:
    if not INPUT_EXCEL.exists():
        raise FileNotFoundError(f"入力Excelが見つかりません: {INPUT_EXCEL}")

    # 読み込み
    companies = pd.read_excel(INPUT_EXCEL, sheet_name="Companies")
    period = pd.read_excel(INPUT_EXCEL, sheet_name="PeriodData")

    # Derived生成
    derived = build_derived(period).derived

    # 書き込み（Companies / PeriodData / Derived）
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        companies.to_excel(writer, sheet_name="Companies", index=False)
        period.to_excel(writer, sheet_name="PeriodData", index=False)
        derived.to_excel(writer, sheet_name="Derived", index=False)

    print("✅ 出力しました:", OUTPUT_EXCEL)
    print("✅ Derived 先頭5行:")
    print(derived.head())


if __name__ == "__main__":
    main()