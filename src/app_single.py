from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

from derived import build_derived, update_single_excel_derived

try:
    import yfinance as yf
except ImportError:
    yf = None

# 日本語フォント設定（Mac向け優先順）
jp_fonts = [
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "YuGothic",
    "Yu Gothic",
    "Noto Sans CJK JP",
    "IPAexGothic",
]
mpl.rcParams["font.family"] = jp_fonts
mpl.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="個別銘柄分析", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

st.title("個別銘柄分析（Derivedビュー）")


@st.cache_data
def load_sheet(path: Path, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet)


@st.cache_data(ttl=21600, show_spinner=False)
def get_latest_price_yfinance(code: str):
    """
    yfinanceで最新株価を軽く取得する。
    負荷を抑えるため:
    - 1銘柄ごとに6時間キャッシュ
    - 直近5営業日の日足だけ取得
    """
    if yf is None:
        return None

    code = str(code).strip()
    if not code:
        return None

    ticker_symbol = f"{code}.T"

    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="5d", interval="1d", auto_adjust=False)

        if hist is None or hist.empty or "Close" not in hist.columns:
            return None

        close_series = hist["Close"].dropna()
        if close_series.empty:
            return None

        return float(close_series.iloc[-1])
    except Exception:
        return None


def ensure_free_cf_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    フリーキャッシュフロー列名の揺れを吸収して、
    最終的に 'フリーキャッシュフロー_億' 列を持つようにする。
    """
    out = df.copy()

    candidates = [
        "フリーキャッシュフロー_億",
        "フリーCF_億",
        "フリーキャッシュフロー",
        "フリーCF",
        "FCF",
    ]

    found = None
    for c in candidates:
        if c in out.columns:
            found = c
            break

    if found is not None and "フリーキャッシュフロー_億" not in out.columns:
        out["フリーキャッシュフロー_億"] = out[found]

    return out


def add_legacy_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derivedから、表示用の旧列名エイリアスを追加する。
    """
    out = df.copy()

    # 列名の前後空白を除去
    out.columns = [str(c).strip() for c in out.columns]

    alias_map = {
        "売上高": "売上高_億",
        "EPS": "EPS_円",
        "営業利益率": "営業利益率_％",
        "自己資本比率": "自己資本比率_％",
        "営業CF": "営業CF_億",
        "現金等": "現金等_億",
        "１株配当": "１株配当_円",
        "配当性向_num": "配当性向_％_num",
        "BPS": "BPS_円",
        "ROE": "ROE_％",
        "ROA": "ROA_％",
        "総資産": "総資産_億",
        "純資産": "純資産_億",
        "投資キャッシュフロー": "投資CF_億",
        "財務キャッシュフロー": "財務CF_億",
    }

    for new_col, old_col in alias_map.items():
        if new_col in out.columns and old_col not in out.columns:
            out[old_col] = out[new_col]

    # 余剰金の配当は今回の計算用にそのまま使う
    if "余剰金の配当" in out.columns:
        out["余剰金の配当"] = pd.to_numeric(out["余剰金の配当"], errors="coerce")

    out = ensure_free_cf_column(out)
    return out


# ----------------------------
# ユーティリティ
# ----------------------------
def latest_non_nan(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return None
    return s.iloc[-1]


def fmt(v, suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{v}{suffix}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.2f}{suffix}" if abs(v) < 100 else f"{v:.1f}{suffix}"
    return f"{v}{suffix}"


def fmt_price(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:,.0f}円"


def calc_per(price, eps):
    if price is None or eps is None:
        return None
    if pd.isna(price) or pd.isna(eps) or eps <= 0:
        return None
    return price / eps

def calc_de_ratio(interest_bearing_debt, equity):
    """
    D/Eレシオ = 有利子負債 ÷ 自己資本（ここでは純資産で代用）
    """
    if interest_bearing_debt is None or equity is None:
        return None
    if pd.isna(interest_bearing_debt) or pd.isna(equity) or equity <= 0:
        return None
    return interest_bearing_debt / equity


def calc_pbr(price, bps):
    if price is None or bps is None:
        return None
    if pd.isna(price) or pd.isna(bps) or bps <= 0:
        return None
    return price / bps


def calc_dividend_yield(price, dividend):
    if price is None or dividend is None:
        return None
    if pd.isna(price) or pd.isna(dividend) or price <= 0:
        return None
    return dividend / price * 100


def calc_cash_dividend_coverage_ratio(operating_cf, cash_dividend):
    """
    現金配当カバレッジレシオ = 営業CF ÷ 余剰金の配当
    ※ 余剰金の配当は正負どちらでも来る可能性があるので絶対値を使う
    """
    if operating_cf is None or cash_dividend is None:
        return None
    if pd.isna(operating_cf) or pd.isna(cash_dividend):
        return None

    dividend_abs = abs(cash_dividend)
    if dividend_abs == 0:
        return None

    return operating_cf / dividend_abs


def make_line_fig(x, y, title, y_label,
                  show_ma=False,
                  window=3,
                  shock_labels=None,
                  threshold_lines=None):
    fig, ax = plt.subplots()

    tmp = pd.DataFrame({
        "FY": list(x),
        "y": pd.Series(y).to_list()
    }).dropna().sort_values("FY")

    ax.plot(
        tmp["FY"],
        tmp["y"],
        marker="o",
        linewidth=1.8,
        label="実績"
    )

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

    if threshold_lines:
        for line in threshold_lines:
            ax.axhline(
                y=line["y"],
                color=line["color"],
                linestyle=line.get("style", ":"),
                linewidth=2
            )

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
# KPI計算関数
# ----------------------------
def calc_non_cut_years(df_code, div_col="１株配当_円"):
    if div_col not in df_code.columns:
        return 0
    tmp = df_code[["FY", div_col]].dropna().sort_values("FY")
    if tmp.shape[0] == 0:
        return 0

    vals = tmp[div_col].astype(float).tolist()
    if len(vals) == 1:
        return 1

    cnt = 1
    for i in range(len(vals) - 1, 0, -1):
        if vals[i] >= vals[i - 1]:
            cnt += 1
        else:
            break
    return cnt


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
    if div_col not in df_code.columns:
        return None

    tmp = df_code[["FY", div_col]].dropna().sort_values("FY").copy()
    if tmp.empty:
        return None

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


def calc_dividend_cut_count_all(df_code, div_col="１株配当_円"):
    if div_col not in df_code.columns:
        return None

    tmp = df_code[["FY", div_col]].dropna().sort_values("FY").copy()
    if tmp.empty:
        return None

    vals = tmp[div_col].astype(float).tolist()
    if len(vals) <= 1:
        return 0

    count = 0
    for i in range(1, len(vals)):
        if vals[i] < vals[i - 1]:
            count += 1

    return count


def calc_no_dividend_count(df_code, div_col="１株配当_円", years=10):
    if div_col not in df_code.columns:
        return None

    tmp = df_code[["FY", div_col]].dropna().sort_values("FY").copy()
    if tmp.empty:
        return None

    latest_fy = tmp["FY"].max()
    tmp = tmp[tmp["FY"] >= latest_fy - years + 1]

    vals = tmp[div_col].astype(float)
    return int((vals == 0).sum())


def calc_no_dividend_count_all(df_code, div_col="１株配当_円"):
    if div_col not in df_code.columns:
        return None

    tmp = df_code[[div_col]].dropna().copy()
    if tmp.empty:
        return None

    vals = tmp[div_col].astype(float)
    return int((vals == 0).sum())

def calc_cagr(df_code, value_col, years=5):
    """
    CAGR = (最終値 / 初期値) ** (1 / 年数) - 1
    指定年数分のデータがある場合のみ計算する
    """
    if value_col not in df_code.columns:
        return None

    tmp = df_code[["FY", value_col]].dropna().sort_values("FY").copy()
    if tmp.empty:
        return None

    latest_fy = tmp["FY"].max()
    start_fy_required = latest_fy - years

    tmp = tmp[tmp["FY"] >= start_fy_required]

    if tmp.shape[0] < 2:
        return None

    start_fy = tmp["FY"].iloc[0]
    end_fy = tmp["FY"].iloc[-1]
    start_val = tmp[value_col].iloc[0]
    end_val = tmp[value_col].iloc[-1]

    period = end_fy - start_fy

    if period < years:
        return None

    if period <= 0 or start_val <= 0 or end_val <= 0:
        return None

    return ((end_val / start_val) ** (1 / period) - 1) * 100

# ----------------------------
# raw配下のxlsx（~$除外）
# ----------------------------
xlsx_files = sorted([p for p in RAW_DIR.glob("*.xlsx") if not p.name.startswith("~$")])
if not xlsx_files:
    st.error(f"{RAW_DIR} に .xlsx が見つかりません。個別銘柄Excelを data/raw/ に置いてください。")
    st.stop()

selected_name = st.sidebar.selectbox("銘柄ファイル", [p.name for p in xlsx_files])
excel_path = RAW_DIR / selected_name

st.sidebar.markdown("---")
if st.sidebar.button("Derivedシートを再生成"):
    try:
        update_single_excel_derived(excel_path)
        st.cache_data.clear()
        st.sidebar.success("Derivedシートを更新しました。")
    except Exception as e:
        st.sidebar.error(f"Derived更新に失敗しました: {e}")

# --- 読み込み
try:
    basic = load_sheet(excel_path, "基本情報")
except Exception as e:
    st.error(f"基本情報シートの読み込みに失敗しました: {e}")
    st.stop()

try:
    df = load_sheet(excel_path, "Derived")
except Exception:
    try:
        period = load_sheet(excel_path, "決算情報")
        df = build_derived(period).derived
        st.info("Derivedシートが無かったため、決算情報から一時生成して表示しています。必要なら左のボタンでExcelに保存してください。")
    except Exception as e:
        st.error(f"Derived/決算情報の読み込みに失敗しました: {e}")
        st.stop()

# 旧app互換の列名を追加
df = add_legacy_alias_columns(df)

# --- 基本整形
df.columns = [str(c).strip() for c in df.columns]
df["証券コード"] = pd.to_numeric(df.get("証券コード"), errors="coerce").astype("Int64").astype(str)

if "FY" not in df.columns:
    dt = pd.to_datetime(df.get("年度"), errors="coerce")
    df["FY"] = dt.dt.year
df["FY"] = pd.to_numeric(df["FY"], errors="coerce").astype("Int64")

need_num_cols = [
    "売上高_億", "EPS_円", "営業利益率_％", "自己資本比率_％",
    "営業CF_億", "投資CF_億", "財務CF_億", "現金等_億",
    "１株配当_円", "配当性向_％_num",
    "BPS_円", "ROE_％", "ROA_％",
    "総資産_億", "純資産_億", "フリーキャッシュフロー_億",
    "余剰金の配当", "現金配当カバレッジレシオ",
    "有利子負債", "D/Eレシオ",
]
for c in need_num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

code = ""
company_name = ""
if len(basic) > 0:
    row = basic.iloc[0]
    code = str(row.get("証券コード", "")).strip()
    company_name = str(row.get("会社名", "")).strip()

df_code = df.copy().sort_values("FY")

# 現金配当カバレッジレシオを計算
if "営業CF_億" in df_code.columns and "余剰金の配当" in df_code.columns:
    df_code["営業CF_億"] = pd.to_numeric(df_code["営業CF_億"], errors="coerce")
    df_code["余剰金の配当"] = pd.to_numeric(df_code["余剰金の配当"], errors="coerce")

    dividend_abs = df_code["余剰金の配当"].abs()
    df_code["現金配当カバレッジレシオ"] = np.where(
        dividend_abs > 0,
        df_code["営業CF_億"] / dividend_abs,
        np.nan
    )
    
# D/Eレシオを計算
if "有利子負債" in df_code.columns and "純資産_億" in df_code.columns:
    df_code["有利子負債"] = pd.to_numeric(df_code["有利子負債"], errors="coerce")
    df_code["純資産_億"] = pd.to_numeric(df_code["純資産_億"], errors="coerce")

    df_code["D/Eレシオ"] = np.where(
        df_code["純資産_億"] > 0,
        df_code["有利子負債"] / df_code["純資産_億"],
        np.nan
    )

st.sidebar.markdown("---")
show_ma = st.sidebar.checkbox("移動平均を表示", value=True)
ma_window = st.sidebar.selectbox("移動平均の期間（年）", [3, 5], index=0)

# ----------------------------
# ショック年
# ----------------------------
DEFAULT_SHOCKS = [
    {"year": 2008, "label": "リーマン"},
    {"year": 2011, "label": "震災"},
    {"year": 2020, "label": "コロナ"},
]

st.sidebar.markdown("---")
show_shocks = st.sidebar.checkbox("ショック年を表示", value=True)

if "shock_df_single" not in st.session_state:
    st.session_state["shock_df_single"] = pd.DataFrame(DEFAULT_SHOCKS)

st.sidebar.markdown("### ショック年（追加・編集）")
shock_df = st.sidebar.data_editor(
    st.session_state["shock_df_single"],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "year": st.column_config.NumberColumn("年", min_value=1900, max_value=2100, step=1),
        "label": st.column_config.TextColumn("ラベル"),
    },
)

st.session_state["shock_df_single"] = shock_df

shock_labels = None
if show_shocks:
    tmp = shock_df.dropna(subset=["year", "label"]).copy()
    tmp["year"] = tmp["year"].astype(int)
    shock_labels = dict(zip(tmp["year"], tmp["label"]))

# FY範囲
fy_min = int(df_code["FY"].min()) if df_code["FY"].notna().any() else 2000
fy_max = int(df_code["FY"].max()) if df_code["FY"].notna().any() else 2000
fy_range = st.sidebar.slider("FY範囲", min_value=fy_min, max_value=fy_max, value=(fy_min, fy_max))
df_code = df_code[(df_code["FY"] >= fy_range[0]) & (df_code["FY"] <= fy_range[1])].copy().sort_values("FY")

st.subheader(f"{code} {company_name}".strip())

# 基本情報から各種URLを表示
yahoo_url = None
buffett_url = None
irbank_url = None

if len(basic) > 0:
    row = basic.iloc[0]
    yahoo_url = row.get("YahooURL")
    buffett_url = row.get("BuffettCodeURL")
    irbank_url = row.get("IRBANKURL")

link_items = []

if isinstance(yahoo_url, str) and yahoo_url.strip():
    link_items.append(f"🔗 Yahoo!ファイナンス: [リンクはこちら]({yahoo_url.strip()})")

if isinstance(buffett_url, str) and buffett_url.strip():
    link_items.append(f"🔗 BuffettCode: [リンクはこちら]({buffett_url.strip()})")

if isinstance(irbank_url, str) and irbank_url.strip():
    link_items.append(f"🔗 IR BANK: [リンクはこちら]({irbank_url.strip()})")

if link_items:
    for item in link_items:
        st.markdown(item)
else:
    st.info("YahooURL / BuffettCodeURL / IRBANKURL が 基本情報 に未設定です。")

# ----------------------------
# 価格系指標（入口フィルター）
# ----------------------------
price_now = get_latest_price_yfinance(code)
eps_now_for_price = latest_non_nan(df_code["EPS_円"]) if "EPS_円" in df_code.columns else None
bps_now_for_price = latest_non_nan(df_code["BPS_円"]) if "BPS_円" in df_code.columns else None
div_now_for_price = latest_non_nan(df_code["１株配当_円"]) if "１株配当_円" in df_code.columns else None

per_now = calc_per(price_now, eps_now_for_price)
pbr_now = calc_pbr(price_now, bps_now_for_price)
div_yield_now = calc_dividend_yield(price_now, div_now_for_price)

st.markdown("## 現在の価格指標")
m1, m2, m3, m4 = st.columns(4)
m1.metric("現在株価", fmt_price(price_now))
m2.metric("実績PER", fmt(per_now))
m3.metric("実績PBR", fmt(pbr_now))
m4.metric("実績配当利回り(%)", fmt(div_yield_now))

if yf is None:
    st.warning("yfinance がインストールされていないため、株価・PER・PBR・配当利回りは表示されていません。`pip install yfinance` を実行してください。")
elif price_now is None:
    st.info("現在株価を取得できなかったため、PER・PBR・配当利回りは表示できませんでした。時間をおいて再読み込みしてください。")

x = df_code["FY"].astype(int).tolist()

# ----------------------------
# セクション別表示
# ----------------------------
st.markdown("## 指標の可視化")

# ① 収益性
st.markdown("### ① 収益性")

roe_now = latest_non_nan(df_code["ROE_％"]) if "ROE_％" in df_code.columns else None
roa_now = latest_non_nan(df_code["ROA_％"]) if "ROA_％" in df_code.columns else None
eps_now = latest_non_nan(df_code["EPS_円"]) if "EPS_円" in df_code.columns else None
op_margin_now = latest_non_nan(df_code["営業利益率_％"]) if "営業利益率_％" in df_code.columns else None

k1, k2, k3, k4 = st.columns(4)
k1.metric("EPS(円)", fmt(eps_now))
k2.metric("ROE(%)", fmt(roe_now))
k3.metric("ROA(%)", fmt(roa_now))
k4.metric("営業利益率(%)", fmt(op_margin_now))

cols = st.columns(3)
with cols[0]:
    if "売上高_億" in df_code.columns:
        st.pyplot(
            make_line_fig(
                x, df_code["売上高_億"],
                "売上高の推移", "売上高（億円）",
                show_ma=show_ma, window=ma_window, shock_labels=shock_labels
            ),
            use_container_width=True
        )
with cols[1]:
    if "EPS_円" in df_code.columns:
        st.pyplot(
            make_line_fig(
                x, df_code["EPS_円"],
                "EPSの推移", "EPS（円）",
                show_ma=show_ma, window=ma_window, shock_labels=shock_labels
            ),
            use_container_width=True
        )
with cols[2]:
    if "営業利益率_％" in df_code.columns:
        st.pyplot(
            make_line_fig(
                x, df_code["営業利益率_％"],
                "営業利益率の推移", "営業利益率（%）",
                show_ma=show_ma,
                window=ma_window,
                shock_labels=shock_labels,
                threshold_lines=[
                    {"y": 5, "color": "red"},
                    {"y": 10, "color": "green"}
                ]
            ),
            use_container_width=True
        )

cols = st.columns(3)
with cols[0]:
    if "ROE_％" in df_code.columns:
        st.pyplot(
            make_line_fig(
                x, df_code["ROE_％"],
                "ROEの推移", "ROE（%）",
                show_ma=show_ma, window=ma_window, shock_labels=shock_labels
            ),
            use_container_width=True
        )
with cols[1]:
    if "ROA_％" in df_code.columns:
        st.pyplot(
            make_line_fig(
                x, df_code["ROA_％"],
                "ROAの推移", "ROA（%）",
                show_ma=show_ma, window=ma_window, shock_labels=shock_labels
            ),
            use_container_width=True
        )
with cols[2]:
    if "BPS_円" in df_code.columns:
        st.pyplot(
            make_line_fig(
                x, df_code["BPS_円"],
                "BPSの推移", "BPS（円）",
                show_ma=show_ma, window=ma_window, shock_labels=shock_labels
            ),
            use_container_width=True
        )

# ② 財務健全性
st.markdown("### ② 財務健全性")

equity_ratio_now = latest_non_nan(df_code["自己資本比率_％"]) if "自己資本比率_％" in df_code.columns else None
de_ratio_now = latest_non_nan(df_code["D/Eレシオ"]) if "D/Eレシオ" in df_code.columns else None

k5, k5_2 = st.columns(2)
k5.metric("自己資本比率(%)", fmt(equity_ratio_now))
k5_2.metric("D/Eレシオ", fmt(de_ratio_now))

cols = st.columns(4)
with cols[0]:
    if "自己資本比率_％" in df_code.columns:
        st.pyplot(
            make_line_fig(
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
            ),
            use_container_width=True
        )
with cols[1]:
    if "D/Eレシオ" in df_code.columns:
        st.pyplot(
            make_line_fig(
                x, df_code["D/Eレシオ"],
                "D/Eレシオの推移", "倍率",
                show_ma=show_ma,
                window=ma_window,
                shock_labels=shock_labels,
                threshold_lines=[
                    {"y": 0.5, "color": "green"},
                    {"y": 1.0, "color": "green"},
                    {"y": 2.0, "color": "red"}
                ]
            ),
            use_container_width=True
        )
with cols[2]:
    if "総資産_億" in df_code.columns:
        st.pyplot(
            make_line_fig(
                x, df_code["総資産_億"],
                "総資産の推移", "総資産（億円）",
                show_ma=show_ma, window=ma_window, shock_labels=shock_labels
            ),
            use_container_width=True
        )
with cols[3]:
    if "純資産_億" in df_code.columns:
        st.pyplot(
            make_line_fig(
                x, df_code["純資産_億"],
                "純資産の推移", "純資産（億円）",
                show_ma=show_ma, window=ma_window, shock_labels=shock_labels
            ),
            use_container_width=True
        )

# ③ キャッシュフロー
st.markdown("### ③ キャッシュフロー")

cash_now = latest_non_nan(df_code["現金等_億"]) if "現金等_億" in df_code.columns else None
ocf_now = latest_non_nan(df_code["営業CF_億"]) if "営業CF_億" in df_code.columns else None
icf_now = latest_non_nan(df_code["投資CF_億"]) if "投資CF_億" in df_code.columns else None
fcf_now = latest_non_nan(df_code["フリーキャッシュフロー_億"]) if "フリーキャッシュフロー_億" in df_code.columns else None
fincf_now = latest_non_nan(df_code["財務CF_億"]) if "財務CF_億" in df_code.columns else None

k6, k7, k8, k9, k10 = st.columns(5)
k6.metric("営業CF(億円)", fmt(ocf_now))
k7.metric("投資CF(億円)", fmt(icf_now))
k8.metric("フリーCF(億円)", fmt(fcf_now))
k9.metric("財務CF(億円)", fmt(fincf_now))
k10.metric("現金等(億円)", fmt(cash_now))

cols = st.columns(2)
with cols[0]:
    if "営業CF_億" in df_code.columns:
        st.pyplot(
            make_bar_signed_fig(
                x, df_code["営業CF_億"],
                "営業活動によるCFの推移（±）", "営業CF（億円）",
                shock_labels=shock_labels
            ),
            use_container_width=True
        )
with cols[1]:
    if "投資CF_億" in df_code.columns:
        st.pyplot(
            make_bar_signed_fig(
                x, df_code["投資CF_億"],
                "投資活動によるCFの推移（±）", "投資CF（億円）",
                shock_labels=shock_labels
            ),
            use_container_width=True
        )

cols = st.columns(3)
with cols[0]:
    if "フリーキャッシュフロー_億" in df_code.columns:
        st.pyplot(
            make_line_fig(
                x, df_code["フリーキャッシュフロー_億"],
                "フリーキャッシュフローの推移", "フリーCF（億円）",
                show_ma=show_ma, window=ma_window, shock_labels=shock_labels
            ),
            use_container_width=True
        )
with cols[1]:
    if "財務CF_億" in df_code.columns:
        st.pyplot(
            make_bar_signed_fig(
                x, df_code["財務CF_億"],
                "財務活動によるCFの推移（±）", "財務CF（億円）",
                shock_labels=shock_labels
            ),
            use_container_width=True
        )
with cols[2]:
    if "現金等_億" in df_code.columns:
        st.pyplot(
            make_line_fig(
                x, df_code["現金等_億"],
                "現金等の推移", "現金等（億円）",
                show_ma=show_ma, window=ma_window, shock_labels=shock_labels
            ),
            use_container_width=True
        )

# ④ 配当
st.markdown("### ④ 配当")

div_now = latest_non_nan(df_code["１株配当_円"]) if "１株配当_円" in df_code.columns else None
div_cagr_3y = calc_cagr(df_code, "１株配当_円", years=3)
div_cagr_5y = calc_cagr(df_code, "１株配当_円", years=5)
div_cagr_10y = calc_cagr(df_code, "１株配当_円", years=10)
payout_now = latest_non_nan(df_code["配当性向_％_num"]) if "配当性向_％_num" in df_code.columns else None
cash_div_coverage_now = latest_non_nan(df_code["現金配当カバレッジレシオ"]) if "現金配当カバレッジレシオ" in df_code.columns else None

non_cut_years = calc_non_cut_years(df_code, "１株配当_円")
consec_increase = calc_consecutive_increase_years(df_code, "１株配当_円")
div_cut_count_all = calc_dividend_cut_count_all(df_code, "１株配当_円")
div_cut_count_10y = calc_dividend_cut_count(df_code, "１株配当_円", years=10)
no_div_count_all = calc_no_dividend_count_all(df_code, "１株配当_円")
no_div_count_10y = calc_no_dividend_count(df_code, "１株配当_円", years=10)

k11, k12, k13, k14, k15 = st.columns(5)
k11.metric("現在の1株配当金(円)", fmt(div_now))
k12.metric("最新の配当性向(%)", fmt(payout_now))
k13.metric("現金配当カバレッジレシオ", fmt(cash_div_coverage_now))
k14.metric("非減配年数（連続）", f"{non_cut_years} 年" if non_cut_years is not None else "データなし")
k15.metric("連続増配年数", f"{consec_increase} 年" if consec_increase is not None else "データなし")

k16, k17 = st.columns(2)
k16.metric("減配回数（全期間）", f"{div_cut_count_all} 回" if div_cut_count_all is not None else "データなし")
k17.metric("減配回数（過去10年）", f"{div_cut_count_10y} 回" if div_cut_count_10y is not None else "データなし")

k18, k19 = st.columns(2)
k18.metric("無配回数（全期間）", f"{no_div_count_all} 回" if no_div_count_all is not None else "データなし")
k19.metric("無配回数（過去10年）", f"{no_div_count_10y} 回" if no_div_count_10y is not None else "データなし")
st.markdown("#### 増配率")

d1, d2, d3 = st.columns(3)
d1.metric("3年増配率 CAGR(%)", fmt(div_cagr_3y))
d2.metric("5年増配率 CAGR(%)", fmt(div_cagr_5y))
d3.metric("10年増配率 CAGR(%)", fmt(div_cagr_10y))

cols = st.columns(3)
with cols[0]:
    if "１株配当_円" in df_code.columns:
        st.pyplot(
            make_line_fig(
                x, df_code["１株配当_円"],
                "1株配当金の推移", "1株配当（円）",
                show_ma=show_ma, window=ma_window, shock_labels=shock_labels
            ),
            use_container_width=True
        )
with cols[1]:
    if "配当性向_％_num" in df_code.columns:
        st.pyplot(
            make_line_fig(
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
            ),
            use_container_width=True
        )
with cols[2]:
    if "現金配当カバレッジレシオ" in df_code.columns:
        st.pyplot(
            make_line_fig(
                x, df_code["現金配当カバレッジレシオ"],
                "現金配当カバレッジレシオの推移", "倍率",
                show_ma=show_ma,
                window=ma_window,
                shock_labels=shock_labels,
                threshold_lines=[
                    {"y": 1, "color": "red"},
                    {"y": 2, "color": "green"}
                ]
            ),
            use_container_width=True
        )

# ⑤ 成長性
st.markdown("### ⑤ 成長性")

eps_cagr_3y = calc_cagr(df_code, "EPS_円", years=3)
eps_cagr_5y = calc_cagr(df_code, "EPS_円", years=5)
eps_cagr_10y = calc_cagr(df_code, "EPS_円", years=10)

g1, g2, g3 = st.columns(3)
g1.metric("3年EPS CAGR(%)", fmt(eps_cagr_3y))
g2.metric("5年EPS CAGR(%)", fmt(eps_cagr_5y))
g3.metric("10年EPS CAGR(%)", fmt(eps_cagr_10y))

cols = st.columns(2)
with cols[0]:
    if "売上高_億" in df_code.columns:
        yoy_sales = df_code["売上高_億"].pct_change() * 100
        st.pyplot(
            make_line_fig(
                x, yoy_sales,
                "売上高成長率の推移", "売上高成長率（%）",
                show_ma=show_ma, window=ma_window, shock_labels=shock_labels
            ),
            use_container_width=True
        )
with cols[1]:
    if "EPS_円" in df_code.columns:
        yoy_eps = df_code["EPS_円"].pct_change() * 100
        st.pyplot(
            make_line_fig(
                x, yoy_eps,
                "EPS成長率の推移", "EPS成長率（%）",
                show_ma=show_ma, window=ma_window, shock_labels=shock_labels
            ),
            use_container_width=True
        )

st.markdown("---")
st.markdown("### 現在表示中のDerivedデータ")
st.dataframe(df_code, use_container_width=True)