"""
train_nonlisted()

train_tf_bs()
"""
from typing import Iterable
from .real_data_fetcher import RealDataFetcher
from .model import VPConfig, VelezParejaModel
from .predict import BalanceSheetForecasterTF
from .utils import setup_logging, _safe_float, _clip_nonneg, _nan_to_zero, LOGGER
from .yfinance import _FIN_KEYS, _get_line_item
import pandas as pd


# =========================
# Demo / CLI
# =========================

def build_panel_from_yfinance(fetcher: RealDataFetcher, tickers: Iterable[str], years: int = 6) -> pd.DataFrame:
    """
    把多个 ticker 的多年数据拼成 panel：
    列：ticker, year(0..), 以及财报字段；year 是序列索引（非自然年份），用于时间排序。
    """
    rows = []
    for t in tickers:
        rec = fetcher.get_multi_year_data(t, years=years)
        if not rec:
            continue
        yd = rec["yearly_data"]
        for i, r in enumerate(yd):
            r2 = dict(r)
            r2["ticker"] = rec["ticker"]
            r2["year"] = i
            rows.append(r2)
    panel = pd.DataFrame(rows)
    if panel.empty:
        raise RuntimeError("Could not build panel data from yfinance.")
    panel = RealDataFetcher.add_financial_ratios(panel)
    return panel


def demo_costco():
    setup_logging("INFO")
    fetcher = RealDataFetcher()
    df = fetcher.fetch_training_dataset(["COST"])
    row = df.iloc[0].to_dict()

    print("Fetched COST (latest):")
    print({k: row[k] for k in ["ticker", "company", "total_assets", "total_debt", "shareholders_equity", "revenue", "equity_volatility"] if k in row})

    print("\nMerton pricing:")
    merton = MertonModel()
    res = merton.price_loan(
        equity_value=row.get("market_cap", 0.0),
        equity_vol=row.get("equity_volatility", 0.30),
        debt_value=row.get("total_debt", 0.0),
        maturity=5.0,
    )
    for k, v in res.items():
        if "bps" in k:
            print(f"{k}: {v:.0f}")
        elif "prob" in k:
            print(f"{k}: {v:.4%}")
        else:
            print(f"{k}: {v:.4g}" if abs(v) < 1e4 else f"{k}: {v:,.0f}")

    print("\nVelez-Pareja forecast (toy):")
    # 用 COST 的历史 ebit / depreciation / ppe 近似输入
    multi = fetcher.get_multi_year_data("COST", years=6)
    yd = multi["yearly_data"]
    ebit = [0.0] + [float(r.get("ebit", 0.0)) / 1e6 for r in yd]  # scale to "million" for readability
    depr = [0.0] + [float(r.get("depreciation", 0.0)) / 1e6 for r in yd]
    nfa0 = float(yd[-1].get("ppe_net", 0.0)) / 1e6
    eq0 = float(yd[-1].get("shareholders_equity", 0.0)) / 1e6

    vp = VelezParejaModel()
    fcast = vp.forecast(eq0, years=5, ebit_list=ebit, depr_list=depr, nfa0=nfa0)
    print(fcast[["year", "ebitda", "cash", "st_loan", "net_income", "total_assets", "balance_check"]].to_string(index=False))


def train_nonlisted():
    setup_logging("INFO")
    fetcher = RealDataFetcher()
    df = fetcher.fetch_training_dataset()
    model = NonListedModel()
    metrics = model.fit(df)
    print("NonListedModel training metrics:", metrics)

    sample_private = {
        "total_assets": 500e6,
        "current_assets": 200e6,
        "current_liabilities": 80e6,
        "total_debt": 100e6,
        "shareholders_equity": 300e6,
        "retained_earnings": 150e6,
        "revenue": 400e6,
        "operating_income": 60e6,
        "ebit": 65e6,
        "net_income": 40e6,
        "interest_expense": 8e6,
        "operating_cashflow": 70e6,
        "capex": 20e6,
        "equity_volatility": 0.25,
        "revenue_prev": 380e6,
    }
    pred = model.predict(sample_private)
    print("Predicted spread (bps):", pred["spread"] * 10000)
    print("95% CI (bps):", pred["spread_lower"] * 10000, pred["spread_upper"] * 10000)


def train_tf_bs():
    setup_logging("INFO")
    fetcher = RealDataFetcher()
    tickers = list(fetcher.cfg.training_tickers)
    panel = build_panel_from_yfinance(fetcher, tickers=tickers, years=6)

    X, Y = BalanceSheetForecasterTF.build_supervised_pairs(panel, group_col="ticker")

    # 特征列：选一些稳健的比率+规模
    feature_cols = [
        "log_assets", "leverage_ratio", "current_ratio", "quick_ratio",
        "profit_margin", "roa", "roe", "interest_coverage", "capex_to_sales",
        "ocf_to_debt", "altman_z", "equity_volatility", "revenue_growth",
        # 也可直接加入一些状态变量（y_t）
        "cash", "current_assets", "ppe_net", "total_debt", "current_liabilities", "total_liabilities", "shareholders_equity", "total_assets",
    ]
    # 有些列可能在 panel 里不存在，安全过滤
    feature_cols = [c for c in feature_cols if c in X.columns]

    tf_model = BalanceSheetForecasterTF(hidden=(64, 64), lr=1e-3)
    metrics = tf_model.fit(X, Y, feature_cols=feature_cols, epochs=200, batch_size=32, val_split=0.2, verbose=0)
    print("TF BalanceSheetForecaster training history (last epoch):")
    print(metrics)

    # 简单展示预测恒等式（balance_check）
    pred_bs = tf_model.predict_balance_sheet(X.head(5))
    print("\nPredicted balance sheet (first 5 rows):")
    print(pred_bs[["total_assets", "total_liabilities", "shareholders_equity", "balance_check"]].to_string(index=False))
