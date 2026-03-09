# =========================
# Velez-Pareja 风格预测
# =========================

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class VPConfig:
    kd: float = 0.05
    st_return: float = 0.03
    min_cash: float = 1000.0


class VelezParejaModel:
    def __init__(self, cfg: Optional[VPConfig] = None):
        self.cfg = cfg or VPConfig()
        self.kd = self.cfg.kd
        self.st_return = self.cfg.st_return
        self.min_cash = self.cfg.min_cash

    def forecast(
        self,
        equity0: float,
        years: int,
        ebit_list: List[float],
        depr_list: List[float],
        nfa0: float,
        capex_list: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        这是你原版 VP forecast 的“整理版”：
        - 用 dict-of-arrays 方式存储
        - 最后重建 total_assets 与 total_liab_equity，并给出 balance_check
        """
        capex_list = capex_list or [0.0] * (years + 1)

        n = int(years)
        r = {k: np.zeros(n + 1, dtype=float) for k in [
            "year", "ebit", "depr", "ebitda", "cash",
            "st_inv", "st_loan", "lt_loan", "st_interest", "lt_interest",
            "st_principal", "lt_principal", "net_income", "dividends",
            "nfa", "total_assets", "total_liab_equity", "balance_check"
        ]}

        r["year"] = np.arange(n + 1)

        r["cash"][0] = max(self.min_cash, 0.0)
        r["st_inv"][0] = 0.0
        r["st_loan"][0] = 0.0
        r["lt_loan"][0] = 0.0
        r["nfa"][0] = max(float(nfa0), 0.0)

        eq = float(equity0)

        # 输入序列补齐长度
        def _get(seq: List[float], idx: int) -> float:
            if idx < len(seq):
                return float(seq[idx])
            return float(seq[-1]) if seq else 0.0

        for y in range(1, n + 1):
            r["ebit"][y] = _get(ebit_list, y)
            r["depr"][y] = _get(depr_list, y)
            r["ebitda"][y] = r["ebit"][y] + r["depr"][y]

            # 假设：经营现金净额近似 EBITDA（你原版里叫 op_ncb）
            op_ncb = r["ebitda"][y]

            st_beg = r["st_loan"][y - 1]
            r["st_interest"][y] = st_beg * self.kd
            r["st_principal"][y] = st_beg  # 简化：一年内还清

            lt_beg = r["lt_loan"][y - 1]
            r["lt_interest"][y] = lt_beg * self.kd
            # 简化：等额还本（若有）
            r["lt_principal"][y] = lt_beg / max(n - (y - 1), 1)

            total_debt_pmt = r["st_interest"][y] + r["st_principal"][y] + r["lt_interest"][y] + r["lt_principal"][y]

            st_inv_return = r["st_inv"][y - 1] * self.st_return
            r["net_income"][y] = r["ebit"][y] + st_inv_return - r["st_interest"][y] - r["lt_interest"][y]

            capex_y = _get(capex_list, y)
            # 现金流：经营 - capex - 债务偿付 + 上期短投回收
            ncb_after_inv = op_ncb - capex_y
            ncb_after_debt = ncb_after_inv - total_debt_pmt + r["st_inv"][y - 1]

            # 分红：取上一期净利润的正部分
            if y > 1:
                r["dividends"][y] = max(0.0, r["net_income"][y - 1])
            else:
                r["dividends"][y] = 0.0

            ncb_after_div = ncb_after_debt - r["dividends"][y]

            # 保持最小现金：不足就借短贷；有富余就做短投
            if ncb_after_div + r["cash"][y - 1] < self.min_cash:
                r["st_loan"][y] = self.min_cash - (ncb_after_div + r["cash"][y - 1])
                r["st_inv"][y] = 0.0
                r["cash"][y] = self.min_cash
            else:
                r["st_loan"][y] = 0.0
                r["cash"][y] = r["cash"][y - 1] + ncb_after_div
                r["st_inv"][y] = max(0.0, r["cash"][y] - self.min_cash)
                r["cash"][y] = max(self.min_cash, r["cash"][y] - r["st_inv"][y])

            # NFA 演进
            r["nfa"][y] = max(0.0, r["nfa"][y - 1] + capex_y - r["depr"][y])

            # Equity（留存收益简单累积）
            eq = eq + r["net_income"][y] - r["dividends"][y]

            # Total assets vs (liab+equity) 检查
            r["total_assets"][y] = r["cash"][y] + r["st_inv"][y] + r["nfa"][y]
            r["total_liab_equity"][y] = r["st_loan"][y] + r["lt_loan"][y] + eq
            r["balance_check"][y] = r["total_assets"][y] - r["total_liab_equity"][y]

        out = pd.DataFrame(r)
        return out
