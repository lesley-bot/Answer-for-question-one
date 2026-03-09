from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
from .yfinance import _FIN_KEYS, _get_line_item
from .utils import setup_logging, _safe_float, _nan_to_zero, LOGGER
import yfinance as yf

@dataclass
class DataFetchConfig:
    training_tickers: Tuple[str, ...] = (
        "COST", "WMT", "TGT", "HD", "LOW",
        "AAPL", "MSFT", "GOOGL", "META", "NVDA",
        "JPM", "BAC", "WFC", "GS", "C",
        "XOM", "CVX", "COP", "SLB", "EOG",
        "UNH", "JNJ", "PFE", "ABBV", "MRK",
    )
    risk_free_rate: float = 0.045
    equity_vol_lookback: str = "2y"
    min_hist_points: int = 100
    annualization: int = 252

class RealDataFetcher:
    """从 yfinance 获取单公司最新财务数据 or 多年序列，并构造常用比率特征。"""

    def __init__(self, cfg: Optional[DataFetchConfig] = None):
        self.cfg = cfg or DataFetchConfig()
        self._ticker_cache: Dict[str, yf.Ticker] = {}

    def _ticker(self, ticker: str) -> yf.Ticker:
        t = ticker.upper().strip()
        if t not in self._ticker_cache:
            self._ticker_cache[t] = yf.Ticker(t)
        return self._ticker_cache[t]

    def get_company_financials(self, ticker: str) -> Optional[Dict[str, float]]:
        """抓取单公司最新一期（列0）财报数据 + 市场数据。"""
        try:
            stock = self._ticker(ticker)
            bs = stock.balance_sheet
            inc = stock.income_stmt
            cf = stock.cashflow
            info = stock.info or {}

            data: Dict[str, float] = {
                "ticker": ticker.upper(),
                "company": info.get("longName", ticker.upper()),
                "sector": info.get("sector", ""),
                "market_cap": _safe_float(info.get("marketCap", 0.0)),
                "beta": _safe_float(info.get("beta", 1.0), 1.0),
            }

            # Balance sheet
            data["total_assets"] = _get_line_item(bs, _FIN_KEYS["total_assets"])
            data["current_assets"] = _get_line_item(bs, _FIN_KEYS["current_assets"])
            data["total_liabilities"] = _get_line_item(bs, _FIN_KEYS["total_liabilities"])
            data["current_liabilities"] = _get_line_item(bs, _FIN_KEYS["current_liabilities"])
            data["shareholders_equity"] = _get_line_item(bs, _FIN_KEYS["shareholders_equity"])
            data["retained_earnings"] = _get_line_item(bs, _FIN_KEYS["retained_earnings"])
            data["cash"] = _get_line_item(bs, _FIN_KEYS["cash"])
            data["inventory"] = _get_line_item(bs, _FIN_KEYS["inventory"])
            data["accounts_receivable"] = _get_line_item(bs, _FIN_KEYS["accounts_receivable"])
            data["accounts_payable"] = _get_line_item(bs, _FIN_KEYS["accounts_payable"])
            data["ppe_net"] = _get_line_item(bs, _FIN_KEYS["ppe_net"])

            # Debt
            total_debt = _get_line_item(bs, _FIN_KEYS["total_debt"])
            if total_debt == 0:
                total_debt = _get_line_item(bs, _FIN_KEYS["long_term_debt"]) + _get_line_item(bs, _FIN_KEYS["short_term_debt"])
            data["total_debt"] = total_debt
            data["long_term_debt"] = _get_line_item(bs, _FIN_KEYS["long_term_debt"])
            data["short_term_debt"] = _get_line_item(bs, _FIN_KEYS["short_term_debt"])

            # Income statement
            data["revenue"] = _get_line_item(inc, _FIN_KEYS["revenue"])
            data["revenue_prev"] = _get_line_item(inc, _FIN_KEYS["revenue"], col=1)
            data["cogs"] = _get_line_item(inc, _FIN_KEYS["cogs"])
            data["gross_profit"] = _get_line_item(inc, _FIN_KEYS["gross_profit"])
            data["operating_income"] = _get_line_item(inc, _FIN_KEYS["operating_income"])
            data["ebit"] = _get_line_item(inc, _FIN_KEYS["ebit"])
            data["ebitda"] = _get_line_item(inc, _FIN_KEYS["ebitda"])
            data["net_income"] = _get_line_item(inc, _FIN_KEYS["net_income"])
            data["interest_expense"] = abs(_get_line_item(inc, _FIN_KEYS["interest_expense"])) or 1.0
            data["depreciation"] = _get_line_item(inc, _FIN_KEYS["depreciation"])

            # Cashflow
            data["operating_cashflow"] = _get_line_item(cf, _FIN_KEYS["operating_cashflow"])
            data["capex"] = abs(_get_line_item(cf, _FIN_KEYS["capex"]))
            data["dividends_paid"] = abs(_get_line_item(cf, _FIN_KEYS["dividends_paid"]))

            # Equity volatility (from price history)
            hist = stock.history(period=self.cfg.equity_vol_lookback)
            if hist is not None and "Close" in hist.columns and len(hist) >= self.cfg.min_hist_points:
                returns = hist["Close"].pct_change().dropna()
                data["equity_volatility"] = float(returns.std() * math.sqrt(self.cfg.annualization))
            else:
                data["equity_volatility"] = 0.30

            if data["total_assets"] <= 0:
                return None

            return data
        except Exception as e:
            LOGGER.exception("Error fetching %s: %s", ticker, e)
            return None

    def get_multi_year_data(self, ticker: str, years: int = 6) -> Optional[Dict[str, object]]:
        """
        抓取多年序列（最多 years 列），并返回按时间升序排列的 yearly_data。
        yfinance 报表列通常是“最新在最左”，我们取前 n_cols 再 reverse 变成从早到晚。
        """
        try:
            stock = self._ticker(ticker)
            bs = stock.balance_sheet
            inc = stock.income_stmt
            cf = stock.cashflow
            info = stock.info or {}

            if bs is None or bs.empty or inc is None or inc.empty:
                return None

            n_cols = min(years, bs.shape[1], inc.shape[1])
            yearly_data: List[Dict[str, float]] = []

            for i in range(n_cols):
                row: Dict[str, float] = {
                    "year_index": i,
                    "total_assets": _get_line_item(bs, _FIN_KEYS["total_assets"], col=i),
                    "current_assets": _get_line_item(bs, _FIN_KEYS["current_assets"], col=i),
                    "total_liabilities": _get_line_item(bs, _FIN_KEYS["total_liabilities"], col=i),
                    "current_liabilities": _get_line_item(bs, _FIN_KEYS["current_liabilities"], col=i),
                    "shareholders_equity": _get_line_item(bs, _FIN_KEYS["shareholders_equity"], col=i),
                    "retained_earnings": _get_line_item(bs, _FIN_KEYS["retained_earnings"], col=i),
                    "cash": _get_line_item(bs, _FIN_KEYS["cash"], col=i),
                    "inventory": _get_line_item(bs, _FIN_KEYS["inventory"], col=i),
                    "ppe_net": _get_line_item(bs, _FIN_KEYS["ppe_net"], col=i),

                    "revenue": _get_line_item(inc, _FIN_KEYS["revenue"], col=i),
                    "cogs": _get_line_item(inc, _FIN_KEYS["cogs"], col=i),
                    "operating_income": _get_line_item(inc, _FIN_KEYS["operating_income"], col=i),
                    "ebit": _get_line_item(inc, _FIN_KEYS["ebit"], col=i),
                    "net_income": _get_line_item(inc, _FIN_KEYS["net_income"], col=i),
                    "interest_expense": abs(_get_line_item(inc, _FIN_KEYS["interest_expense"], col=i)) or 1.0,
                    "depreciation": _get_line_item(inc, _FIN_KEYS["depreciation"], col=i)
                                    or _get_line_item(cf, _FIN_KEYS["depreciation"], col=i),

                    "operating_cashflow": _get_line_item(cf, _FIN_KEYS["operating_cashflow"], col=i),
                    "capex": abs(_get_line_item(cf, _FIN_KEYS["capex"], col=i)),
                    "dividends_paid": abs(_get_line_item(cf, _FIN_KEYS["dividends_paid"], col=i)),
                }

                # total_debt（有些公司没有 Total Debt 行）
                td = _get_line_item(bs, _FIN_KEYS["total_debt"], col=i)
                if td == 0:
                    td = _get_line_item(bs, _FIN_KEYS["long_term_debt"], col=i) + _get_line_item(bs, _FIN_KEYS["short_term_debt"], col=i)
                row["total_debt"] = td

                yearly_data.append(row)

            yearly_data = list(reversed(yearly_data))  # from old -> new

            return {
                "ticker": ticker.upper(),
                "company": info.get("longName", ticker.upper()),
                "sector": info.get("sector", ""),
                "market_cap": _safe_float(info.get("marketCap", 0.0)),
                "yearly_data": yearly_data,
            }
        except Exception as e:
            LOGGER.exception("Error fetching multi-year %s: %s", ticker, e)
            return None

    @staticmethod
    def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """给 DataFrame 增加常用财务比率（对数规模、杠杆、覆盖倍数、增长等）。"""
        df = df.copy()
        ta = df["total_assets"].replace(0, np.nan)
        td = df.get("total_debt", 0).replace(0, np.nan)
        eq = df.get("shareholders_equity", 0).abs().replace(0, np.nan)

        df["leverage_ratio"] = df.get("total_debt", 0) / ta
        df["debt_to_equity"] = df.get("total_debt", 0) / eq
        df["current_ratio"] = df.get("current_assets", 0) / df.get("current_liabilities", 0).replace(0, np.nan)
        df["quick_ratio"] = (df.get("current_assets", 0) - df.get("inventory", 0)) / df.get("current_liabilities", 0).replace(0, np.nan)
        df["profit_margin"] = df.get("net_income", 0) / df.get("revenue", 0).replace(0, np.nan)
        df["roa"] = df.get("net_income", 0) / ta
        df["roe"] = df.get("net_income", 0) / eq
        df["interest_coverage"] = df.get("ebit", 0) / df.get("interest_expense", 1).replace(0, 1)

        df["log_assets"] = np.log(df.get("total_assets", 0) + 1.0)
        df["capex_to_sales"] = df.get("capex", 0) / df.get("revenue", 0).replace(0, np.nan)
        df["ocf_to_debt"] = df.get("operating_cashflow", 0) / td

        # Altman Z (简化版，适用于非金融企业；对金融业仅作参考)
        wc = df.get("current_assets", 0) - df.get("current_liabilities", 0)
        df["altman_z"] = (
            1.2 * (wc / ta)
            + 1.4 * (df.get("retained_earnings", 0) / ta)
            + 3.3 * (df.get("ebit", 0) / ta)
            + 0.6 * (eq / td.replace(0, np.nan))
            + 1.0 * (df.get("revenue", 0) / ta)
        )

        # revenue growth（若同表中包含 revenue_prev）
        if "revenue_prev" in df.columns:
            df["revenue_growth"] = (df["revenue"] - df["revenue_prev"]) / df["revenue_prev"].replace(0, np.nan)
        return _nan_to_zero(df)

    def fetch_training_dataset(self, tickers: Optional[Iterable[str]] = None) -> pd.DataFrame:
        tickers = list(tickers) if tickers is not None else list(self.cfg.training_tickers)
        rows: List[Dict[str, float]] = []
        for t in tickers:
            LOGGER.info("Fetching %s ...", t)
            rec = self.get_company_financials(t)
            if rec:
                rows.append(rec)
            else:
                LOGGER.warning("Failed to fetch %s", t)

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("No training data fetched from yfinance.")
        df = self.add_financial_ratios(df)
        df["credit_spread"] = df.apply(lambda r: self.estimate_spread_heuristic(r, self.cfg.risk_free_rate), axis=1)
        return df

    @staticmethod
    def estimate_spread_heuristic(row: pd.Series, r: float) -> float:
        z = float(row.get("altman_z", 2))
        lev = float(row.get("leverage_ratio", 0.5))
        cov = float(row.get("interest_coverage", 5))
        vol = float(row.get("equity_volatility", 0.3))

        if z > 3.0:
            base = 0.008
        elif z > 2.5:
            base = 0.012
        elif z > 2.0:
            base = 0.018
        elif z > 1.5:
            base = 0.030
        elif z > 1.0:
            base = 0.050
        else:
            base = 0.080

        base *= 1 + 0.5 * max(0.0, lev - 0.3)
        base *= 1 + 0.2 * max(0.0, 4.0 - cov)
        base *= 1 + 0.3 * max(0.0, vol - 0.25)
        return max(0.0, base)
