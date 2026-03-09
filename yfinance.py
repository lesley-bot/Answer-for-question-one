# =========================
# yfinance 财务科目同义项映射
# =========================

from typing import Dict, List
import pandas as pd

_FIN_KEYS: Dict[str, List[str]] = {
    "total_assets": ["Total Assets"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest", "Total Liab"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "shareholders_equity": ["Stockholders Equity", "Total Stockholders Equity", "Total Equity Gross Minority Interest"],
    "retained_earnings": ["Retained Earnings"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"],
    "inventory": ["Inventory"],
    "accounts_receivable": ["Accounts Receivable", "Net Receivables"],
    "accounts_payable": ["Accounts Payable"],
    "ppe_net": ["Net PPE", "Property Plant Equipment Net", "Property Plant And Equipment Net"],
    "total_debt": ["Total Debt"],
    "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "short_term_debt": ["Current Debt", "Current Debt And Capital Lease Obligation"],

    "revenue": ["Total Revenue"],
    "cogs": ["Cost Of Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income"],
    "ebit": ["EBIT", "Operating Income"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "interest_expense": ["Interest Expense"],
    "depreciation": ["Depreciation And Amortization"],

    "operating_cashflow": ["Operating Cash Flow", "Total Cash From Operating Activities"],
    "capex": ["Capital Expenditure"],
    "dividends_paid": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
}


def _get_line_item(df: pd.DataFrame, keys: List[str], col: int = 0, default: float = 0.0) -> float:
    if df is None or df.empty:
        return default
    for k in keys:
        if k in df.index:
            try:
                val = df.loc[k].iloc[col]
                if pd.notna(val):
                    return float(val)
            except Exception:
                continue
    return default
