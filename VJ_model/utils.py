# =========================
# Logging
# =========================

import logging
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("k1")
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=_LOG_FORMAT)


# =========================
# Utilities
# =========================

def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        if isinstance(x, (float, int)):
            return float(x)
        if isinstance(x, str):
            x = x.replace(",", "").strip()
            return float(x) if x else default
        return float(x)
    except Exception:
        return default


def _clip_nonneg(x: float) -> float:
    return float(x) if x > 0 else 0.0


def _nan_to_zero(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)