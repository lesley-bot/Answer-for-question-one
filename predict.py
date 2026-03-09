# =========================
# TensorFlow：资产负债表预测（会计恒等式严格满足）
# =========================

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from .yfinance import _FIN_KEYS, _get_line_item
from .utils import setup_logging, _safe_float, _clip_nonneg, _nan_to_zero, LOGGER


class BalanceSheetForecasterTF:
    """
    目标：学习 y_{t+1} = f_theta(x_t, y_t)，并且严格满足：
        TotalAssets_{t+1} == TotalLiabilities_{t+1} + Equity_{t+1}

    做法（强烈推荐的“结构化参数化”）：
    - 模型只输出 6 个非负“自由分量”：
        cash, other_current_assets, ppe, other_assets, current_liab, long_term_liab
    - 由这些分量重建：
        current_assets = cash + other_current_assets
        total_assets   = current_assets + ppe + other_assets
        total_liab     = current_liab + long_term_liab
        equity         = total_assets - total_liab
      => 恒等式在 forward 中天然成立（严格成立）

    备注：equity 可能为负（资不抵债也现实存在）。若你想强制 equity>=0，可在 loss 中加惩罚。
    """

    def __init__(self, hidden: Tuple[int, ...] = (64, 64), lr: float = 1e-3, seed: int = 42):
        self.hidden = hidden
        self.lr = lr
        self.seed = seed
        self._model = None
        self._feature_cols: List[str] = []

    @staticmethod
    def _require_tf():
        try:
            import tensorflow as tf  # noqa
            return tf
        except Exception as e:
            raise RuntimeError(
                "TensorFlow 未安装或不可用。请先安装 tensorflow，然后再运行 --train-tf-bs。\n"
                f"原始错误: {e}"
            )

    @staticmethod
    def _make_targets(df: pd.DataFrame) -> pd.DataFrame:
        """将下一期的真实资产负债表拆成 6 个分量目标（并做非负裁剪）。"""
        out = pd.DataFrame(index=df.index)
        cash = df["cash"].astype(float)
        current_assets = df["current_assets"].astype(float)
        ppe = df.get("ppe_net", 0).astype(float)
        total_assets = df["total_assets"].astype(float)
        current_liab = df["current_liabilities"].astype(float)
        total_liab = df["total_liabilities"].astype(float)

        out["cash"] = cash.clip(lower=0)
        out["other_current_assets"] = (current_assets - cash).clip(lower=0)
        out["ppe"] = ppe.clip(lower=0)
        out["other_assets"] = (total_assets - current_assets - ppe).clip(lower=0)

        out["current_liab"] = current_liab.clip(lower=0)
        out["long_term_liab"] = (total_liab - current_liab).clip(lower=0)
        return out

    @staticmethod
    def build_supervised_pairs(panel: pd.DataFrame, group_col: str = "ticker") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        输入：panel 每行一条（company, year）记录，按时间升序排列。
        输出：X（t期特征）、Y（t+1 的 6 分量）
        """
        # 确保排序
        panel = panel.sort_values([group_col, "year"]).reset_index(drop=True)
        X_rows = []
        Y_rows = []
        for _, g in panel.groupby(group_col, sort=False):
            if len(g) < 2:
                continue
            g = g.reset_index(drop=True)
            x_t = g.iloc[:-1].copy()
            y_t1 = g.iloc[1:].copy()
            X_rows.append(x_t)
            Y_rows.append(BalanceSheetForecasterTF._make_targets(y_t1))

        if not X_rows:
            raise ValueError("Not enough time-series rows to build supervised pairs.")
        X = pd.concat(X_rows, axis=0).reset_index(drop=True)
        Y = pd.concat(Y_rows, axis=0).reset_index(drop=True)
        return X, Y

    def build_model(self, n_features: int):
        tf = self._require_tf()
        tf.random.set_seed(self.seed)

        inp = tf.keras.Input(shape=(n_features,), name="features")
        x = inp
        for i, h in enumerate(self.hidden):
            x = tf.keras.layers.Dense(h, activation="relu", name=f"dense_{i}")(x)

        # 输出 6 个“自由分量”的 raw 值，然后 softplus → 非负
        raw = tf.keras.layers.Dense(6, activation=None, name="raw_out")(x)
        comps = tf.keras.layers.Activation(tf.nn.softplus, name="nonneg_comps")(raw)

        self._model = tf.keras.Model(inputs=inp, outputs=comps, name="bs_forecaster")
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss="mse",
            metrics=[tf.keras.metrics.MeanAbsolutePercentageError(name="mape")],
        )
        return self._model

    def fit(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        feature_cols: List[str],
        epochs: int = 200,
        batch_size: int = 32,
        val_split: float = 0.2,
        verbose: int = 0,
    ) -> Dict[str, float]:
        tf = self._require_tf()

        self._feature_cols = list(feature_cols)
        Xn = _nan_to_zero(X[self._feature_cols]).astype("float32").values
        Yn = _nan_to_zero(Y).astype("float32").values  # 6 columns

        if self._model is None:
            self.build_model(n_features=Xn.shape[1])

        hist = self._model.fit(
            Xn, Yn,
            epochs=int(epochs),
            batch_size=int(batch_size),
            validation_split=float(val_split),
            verbose=int(verbose),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            ],
        )
        last = {k: float(v[-1]) for k, v in hist.history.items()}
        return last

    def predict_components(self, X: pd.DataFrame) -> pd.DataFrame:
        tf = self._require_tf()
        if self._model is None:
            raise RuntimeError("Model not fit. Call fit() first.")
        Xn = _nan_to_zero(X[self._feature_cols]).astype("float32").values
        comps = self._model.predict(Xn, verbose=0)
        cols = ["cash", "other_current_assets", "ppe", "other_assets", "current_liab", "long_term_liab"]
        return pd.DataFrame(comps, columns=cols)

    @staticmethod
    def reconstruct_balance_sheet(components: pd.DataFrame) -> pd.DataFrame:
        """由 6 分量重建 total assets / total liabilities / equity，并返回全表。"""
        df = components.copy()
        df["current_assets"] = df["cash"] + df["other_current_assets"]
        df["total_assets"] = df["current_assets"] + df["ppe"] + df["other_assets"]
        df["total_liabilities"] = df["current_liab"] + df["long_term_liab"]
        df["shareholders_equity"] = df["total_assets"] - df["total_liabilities"]
        df["balance_check"] = df["total_assets"] - (df["total_liabilities"] + df["shareholders_equity"])
        # balance_check 理论上恒为 0（数值误差级别）
        return df

    def predict_balance_sheet(self, X: pd.DataFrame) -> pd.DataFrame:
        comps = self.predict_components(X)
        return self.reconstruct_balance_sheet(comps)