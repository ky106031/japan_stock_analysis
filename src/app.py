from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

# 日本語フォント設定（Mac向け優先順）
jp_fonts = [
    "Hiragino Sans",          # macOS
    "Hiragino Kaku Gothic ProN",
    "YuGothic",               # macOS/Windows
    "Yu Gothic",
    "Noto Sans CJK JP",       # 入っていれば強い
    "IPAexGothic",            # 入っていれば強い
]
mpl.rcParams["font.family"] = jp_fonts
mpl.rcParams["axes.unicode_minus"] = False  # マイナス記号が文字化けするのを防ぐ

st.set_page_config(page_title="Japan Stock DB Viewer", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

st.title("日本株データベース（Derivedビュー）")

# --- data/配下のxlsx（~$除外）
xlsx_files = sorted([p for p in DATA_DIR.glob("*.xlsx") if not p.name.startswith("~$")])
if not xlsx_files:
    st.error(f"{DATA_DIR} に .xlsx が見つかりません。Excelを data/ に置いてください。")
    st.stop()

file_names = [f.name for f in xlsx_files]

# out をデフォルト選択
default_index = 0
for i, name in enumerate(file_names):
    if name.endswith("_out.xlsx"):
        default_index = i
        break

selected_name = st.sidebar.selectbox("Excelファイル", file_names, index=default_index)
excel_path = DATA_DIR / selected_name

@st.cache_data
def load_sheet(path: Path, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet)

# --- 読み込み（Derivedが無ければエラー）
try:
    companies = load_sheet(excel_path, "Companies")
    df = load_sheet(excel_path, "Derived")
except Exception as e:
    st.error(f"読み込みに失敗しました: {e}")
    st.stop()

# --- 基本整形
# 証券コード
df["証券コード"] = pd.to_numeric(df.get("証券コード"), errors="coerce").astype("Int64").astype(str)

# FY（無ければ年度から作る）
if "FY" not in df.columns:
    dt = pd.to_datetime(df.get("年度"), errors="coerce")
    df["FY"] = dt.dt.year
df["FY"] = pd.to_numeric(df["FY"], errors="coerce").astype("Int64")

# 数値化（必要な列だけ）
need_num_cols = [
    "売上高_億", "EPS_円", "営業利益率_％", "自己資本比率_％",
    "営業CF_億", "現金等_億", "１株配当_円", "配当性向_％_num",
    # 追加6グラフ用
    "BPS_円", "配当性向_％_num", "ROE_％",
    "総資産_億", "純資産_億", "自己資本比率_％",
]
for c in need_num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# --- 銘柄選択
codes = sorted(df["証券コード"].dropna().unique().tolist())
code = st.sidebar.selectbox("証券コード", codes)

df_code = df[df["証券コード"] == code].copy()
df_code = df_code.sort_values("FY")

st.sidebar.markdown("---")
show_ma = st.sidebar.checkbox("移動平均を表示", value=True)
ma_window = st.sidebar.selectbox("移動平均の期間（年）", [3, 5], index=0)

# ----------------------------
# ショック年（オプションで追加・編集）
# ----------------------------
DEFAULT_SHOCKS = [
    {"year": 2008, "label": "リーマン"},
    {"year": 2011, "label": "震災"},
    {"year": 2020, "label": "コロナ"},
]

st.sidebar.markdown("---")
show_shocks = st.sidebar.checkbox("ショック年を表示", value=True)

if "shock_df" not in st.session_state:
    st.session_state["shock_df"] = pd.DataFrame(DEFAULT_SHOCKS)

st.sidebar.markdown("### ショック年（追加・編集）")
shock_df = st.sidebar.data_editor(
    st.session_state["shock_df"],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "year": st.column_config.NumberColumn("年", min_value=1900, max_value=2100, step=1),
        "label": st.column_config.TextColumn("ラベル"),
    },
)

st.session_state["shock_df"] = shock_df

shock_labels = None
if show_shocks:
    tmp = shock_df.dropna(subset=["year", "label"]).copy()
    tmp["year"] = tmp["year"].astype(int)
    shock_labels = dict(zip(tmp["year"], tmp["label"]))

# FY範囲
fy_min = int(df_code["FY"].min()) if df_code["FY"].notna().any() else 2000
fy_max = int(df_code["FY"].max()) if df_code["FY"].notna().any() else 2000
fy_range = st.sidebar.slider("FY範囲", min_value=fy_min, max_value=fy_max, value=(fy_min, fy_max))
df_code = df_code[(df_code["FY"] >= fy_range[0]) & (df_code["FY"] <= fy_range[1])].copy()
df_code = df_code.sort_values("FY")

# 会社名
company_name = ""
if "証券コード" in companies.columns and "会社名" in companies.columns:
    comp = companies.copy()
    comp["証券コード"] = pd.to_numeric(comp["証券コード"], errors="coerce").astype("Int64").astype(str)
    hit = comp[comp["証券コード"] == code]
    if len(hit) > 0:
        company_name = str(hit.iloc[0]["会社名"])

st.subheader(f"{code} {company_name}".strip())

# CompaniesからYahooURLを引いて表示
yahoo_url = None
if "YahooURL" in companies.columns:
    comp = companies.copy()
    comp["証券コード"] = pd.to_numeric(comp["証券コード"], errors="coerce").astype("Int64").astype(str)
    hit = comp[comp["証券コード"] == code]
    if len(hit) > 0:
        yahoo_url = hit.iloc[0].get("YahooURL")

if isinstance(yahoo_url, str) and yahoo_url.strip():
    st.markdown(f"🔗 Yahoo!ファイナンス: [リンクはこちら]({yahoo_url.strip()})")
else:
    st.info("YahooURL が Companies に未設定です（Companiesに YahooURL 列を作って貼り付けてください）")

# ----------------------------
# ユーティリティ
# ----------------------------
def latest_non_nan(series: pd.Series):
    """欠損でない最新値を返す（無ければNone）"""
    s = series.dropna()
    if len(s) == 0:
        return None
    return s.iloc[-1]

def fmt(v, suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    # 見やすさ用（必要なら調整）
    if isinstance(v, (int, np.integer)):
        return f"{v}{suffix}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.2f}{suffix}" if abs(v) < 100 else f"{v:.1f}{suffix}"
    return f"{v}{suffix}"

def make_line_fig(x, y, title, y_label,
                  show_ma=False,
                  window=3,
                  shock_labels=None,
                  threshold_lines=None):
    """
    threshold_lines:
        [
            {"y": 5, "color": "red", "style": ":", "label": "5%ライン"},
            ...
        ]
    """

    fig, ax = plt.subplots()

    tmp = pd.DataFrame({
        "FY": list(x),
        "y": pd.Series(y).to_list()
    }).dropna().sort_values("FY")

    # 実績
    ax.plot(
        tmp["FY"],
        tmp["y"],
        marker="o",
        linewidth=1.8,
        label="実績"
    )

    # 移動平均
    if show_ma and window and window >= 2:
        ma = tmp["y"].rolling(window=window, min_periods=window).mean()
        ax.plot(
            tmp["FY"],
            ma,
            linestyle="--",
            linewidth=3.0,
            color="black",
            label=f"{window}年移動平均",
            zorder=5
        )

    # --- 閾値ライン追加 ---
    if threshold_lines:
        for line in threshold_lines:
            ax.axhline(
                y=line["y"],
                color=line["color"],
                linestyle=line.get("style", ":"),
                linewidth=2
            )

    # --- ショック年 ---
    if shock_labels:
        ymin, ymax = ax.get_ylim()
        y_text = ymax - (ymax - ymin) * 0.03
        years_in_plot = set(tmp["FY"].astype(int).tolist())

        for yr, name in shock_labels.items():
            if yr in years_in_plot:
                ax.axvline(yr, linestyle=":", linewidth=2, color="gray", alpha=0.8)
                ax.text(yr, y_text, name, rotation=90, va="top", ha="right", color="gray")

    ax.set_title(title)
    ax.set_xlabel("FY")
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.legend()

    return fig

def make_bar_signed_fig(x, y, title, y_label, shock_labels=None):
    fig, ax = plt.subplots()

    tmp = pd.DataFrame({"FY": list(x), "y": pd.Series(y).to_list()}).dropna().sort_values("FY")

    ax.bar(tmp["FY"], tmp["y"])
    ax.axhline(0, linewidth=1)

    if shock_labels:
        ymin, ymax = ax.get_ylim()
        y_text = ymax - (ymax - ymin) * 0.03
        years_in_plot = set(tmp["FY"].astype(int).tolist())
        for yr, name in shock_labels.items():
            if yr in years_in_plot:
                ax.axvline(yr, linestyle=":", linewidth=2, color="gray", alpha=0.8)
                ax.text(yr, y_text, name, rotation=90, va="top", ha="right", color="gray")

    ax.set_title(title)
    ax.set_xlabel("FY")
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y")
    return fig

# ----------------------------
# KPI（現在値：最新の有効値）
# ----------------------------
st.markdown("## 現在の主要指標（最新の有効値）")

# --- helper: 連続非減配年数を計算する
def calc_non_cut_years(df_code, div_col="１株配当_円"):
    """
    df_code: 銘柄ごとの DataFrame（FY 昇順にソート済みを前提）
    div_col: 配当列名（文字列）
    戻り値: latest_year_included の連続年数（int）
    - データ点が1個なら 1 を返す
    - 直近で減配していれば 1（最新年のみ）または 0（配当データ無し）
    """
    if div_col not in df_code.columns:
        return 0
    tmp = df_code[["FY", div_col]].dropna().sort_values("FY")
    if tmp.shape[0] == 0:
        return 0

    vals = tmp[div_col].astype(float).tolist()
    # 1点だけなら非減配年数は1
    if len(vals) == 1:
        return 1

    # 最新年から遡り、前年度より「減っていない（>=）」間カウント
    cnt = 1  # 最新年を含める
    for i in range(len(vals) - 1, 0, -1):
        if vals[i] >= vals[i - 1]:
            cnt += 1
        else:
            break
    return cnt

# （オプション）連続増配年数もあれば一緒に出せる関数
def calc_consecutive_increase_years(df_code, div_col="１株配当_円"):
    if div_col not in df_code.columns:
        return 0

    tmp = df_code[["FY", div_col]].dropna().sort_values("FY")
    if tmp.shape[0] <= 1:
        return 0

    vals = tmp[div_col].astype(float).tolist()

    cnt = 0
    for i in range(len(vals) - 1, 0, -1):
        if vals[i] > vals[i - 1]:
            cnt += 1
        else:
            break

    return cnt

def calc_dividend_cut_count(df_code, div_col="１株配当_円", years=10):
    """
    直近years年分について、前年より減配した回数を数える
    """
    if div_col not in df_code.columns:
        return None

    tmp = df_code[["FY", div_col]].dropna().sort_values("FY").copy()
    if tmp.empty:
        return None

    # 直近10年分に絞る
    latest_fy = tmp["FY"].max()
    tmp = tmp[tmp["FY"] >= latest_fy - years + 1]

    vals = tmp[div_col].astype(float).tolist()
    if len(vals) <= 1:
        return 0

    count = 0
    for i in range(1, len(vals)):
        if vals[i] < vals[i - 1]:
            count += 1

    return count


def calc_no_dividend_count(df_code, div_col="１株配当_円", years=10):
    """
    直近years年分について、無配（0円）の年数を数える
    """
    if div_col not in df_code.columns:
        return None

    tmp = df_code[["FY", div_col]].dropna().sort_values("FY").copy()
    if tmp.empty:
        return None

    latest_fy = tmp["FY"].max()
    tmp = tmp[tmp["FY"] >= latest_fy - years + 1]

    vals = tmp[div_col].astype(float)
    return int((vals == 0).sum())

# 最新値（既存）
div_now = latest_non_nan(df_code["１株配当_円"]) if "１株配当_円" in df_code.columns else None
payout_now = latest_non_nan(df_code["配当性向_％_num"]) if "配当性向_％_num" in df_code.columns else None
roe_now = latest_non_nan(df_code["ROE_％"]) if "ROE_％" in df_code.columns else None
equity_ratio_now = latest_non_nan(df_code["自己資本比率_％"]) if "自己資本比率_％" in df_code.columns else None

# 非減配年数（新規）
non_cut_years = calc_non_cut_years(df_code, "１株配当_円")
# （任意）連続増配年数
consec_increase = calc_consecutive_increase_years(df_code, "１株配当_円")
div_cut_count_10y = calc_dividend_cut_count(df_code, "１株配当_円", years=10)
no_div_count_10y = calc_no_dividend_count(df_code, "１株配当_円", years=10)

# 表示（1行目：既存の4指標）
c1, c2, c3, c4 = st.columns(4)
c1.metric("現在の1株配当金(円)", fmt(div_now))
c2.metric("最新の配当性向(%)", fmt(payout_now))
c3.metric("ROE(%)", fmt(roe_now))
c4.metric("自己資本比率(%)", fmt(equity_ratio_now))

c5, c6, c7, c8 = st.columns(4)

c5.metric("非減配年数（連続）", f"{non_cut_years} 年" if non_cut_years is not None else "データなし")
c6.metric("連続増配年数", f"{consec_increase} 年" if consec_increase is not None else "データなし")
c7.metric("過去10年の減配回数", f"{div_cut_count_10y} 回" if div_cut_count_10y is not None else "データなし")
c8.metric("過去10年の無配回数", f"{no_div_count_10y} 回" if no_div_count_10y is not None else "データなし")

x = df_code["FY"].astype(int).tolist()

# 描画する6つを用意（Noneはスキップ）
figs = []

if "売上高_億" in df_code.columns:
    figs.append(make_line_fig(x, df_code["売上高_億"], "売上高の推移", "売上高（億円）", show_ma=show_ma, window=ma_window, shock_labels=shock_labels))

if "EPS_円" in df_code.columns:
    figs.append(make_line_fig(x, df_code["EPS_円"], "EPSの推移", "EPS（円）", show_ma=show_ma, window=ma_window, shock_labels=shock_labels))

if "営業利益率_％" in df_code.columns:
    figs.append(make_line_fig(
        x, df_code["営業利益率_％"],
        "営業利益率の推移", "営業利益率（%）",
        show_ma=show_ma,
        window=ma_window,
        shock_labels=shock_labels,
        threshold_lines=[
            {"y": 5, "color": "red"},
            {"y": 10, "color": "green"}
        ]
    ))

if "営業CF_億" in df_code.columns:
    figs.append(make_bar_signed_fig(x, df_code["営業CF_億"], "営業活動によるCFの推移（±）", "営業CF（億円）", shock_labels=shock_labels))

if "現金等_億" in df_code.columns:
    figs.append(make_line_fig(x, df_code["現金等_億"], "現金等の推移", "現金等（億円）", show_ma=show_ma, window=ma_window, shock_labels=shock_labels))

if "１株配当_円" in df_code.columns:
    figs.append(make_line_fig(x, df_code["１株配当_円"], "1株配当金の推移", "1株配当（円）", shock_labels=shock_labels))

# ----------------------------
# 追加グラフ（6枚）
# ----------------------------

if "BPS_円" in df_code.columns:
    figs.append(make_line_fig(
        x, df_code["BPS_円"],
        "BPSの推移", "BPS（円）",
        show_ma=show_ma, window=ma_window, shock_labels=shock_labels
    ))

if "自己資本比率_％" in df_code.columns:
    figs.append(make_line_fig(
        x, df_code["自己資本比率_％"],
        "自己資本比率の推移", "自己資本比率（%）",
        show_ma=show_ma,
        window=ma_window,
        shock_labels=shock_labels,
        threshold_lines=[
            {"y": 40, "color": "red"},
            {"y": 60, "color": "green"},
            {"y": 80, "color": "green"}
        ]
    ))

if "配当性向_％_num" in df_code.columns:
    figs.append(make_line_fig(
        x, df_code["配当性向_％_num"],
        "配当性向の推移", "配当性向（%）",
        show_ma=show_ma,
        window=ma_window,
        shock_labels=shock_labels,
        threshold_lines=[
            {"y": 30, "color": "green"},
            {"y": 50, "color": "green"},
            {"y": 70, "color": "red"}
        ]
    ))

if "ROE_％" in df_code.columns:
    figs.append(make_line_fig(
        x, df_code["ROE_％"],
        "ROEの推移", "ROE（%）",
        show_ma=show_ma, window=ma_window, shock_labels=shock_labels
    ))

if "総資産_億" in df_code.columns:
    figs.append(make_line_fig(
        x, df_code["総資産_億"],
        "総資産の推移", "総資産（億円）",
        show_ma=show_ma, window=ma_window, shock_labels=shock_labels
    ))

if "純資産_億" in df_code.columns:
    figs.append(make_line_fig(
        x, df_code["純資産_億"],
        "純資産の推移", "純資産（億円）",
        show_ma=show_ma, window=ma_window, shock_labels=shock_labels
    ))

rows = [figs[:3], figs[3:6], figs[6:9], figs[9:12]]
for r in rows:
    cols = st.columns(3)
    for col, fig in zip(cols, r):
        with col:
            st.pyplot(fig, use_container_width=True)