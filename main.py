"""

依赖：
- numpy, pandas, scipy, scikit-learn, yfinance
- tensorflow (用于 BalanceSheetForecasterTF；未安装时会给出清晰报错)

用法（示例）：
    python main.py --demo-costco
    python main.py --train-nonlisted
    python main.py --train-tf-bs

注意：yfinance 财报字段会随公司/行业变化，下面的字段映射已尽量覆盖常见情况，但仍建议你在报告中说明“数据源字段不一致”的处理策略。
"""
import argparse
from .train import train_nonlisted, train_tf_bs, build_panel_from_yfinance, demo_costco
from .utils import setup_logging, _safe_float, _clip_nonneg, _nan_to_zero, LOGGER


def main():
    parser = argparse.ArgumentParser(description="k1 optimized (loan pricing + balance sheet forecasting with TF constraint).")
    parser.add_argument("--demo-costco", action="store_true", help="run a small demo on COST (Costco)")
    parser.add_argument("--train-nonlisted", action="store_true", help="train the non-listed spread ML model (scikit-learn)")
    parser.add_argument("--train-tf-bs", action="store_true", help="train the TensorFlow balance sheet forecaster (Part 1 requirement)")
    parser.add_argument("--log-level", default="INFO", help="logging level (DEBUG/INFO/WARNING/ERROR)")
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.demo_costco:
        demo_costco()
    if args.train_nonlisted:
        train_nonlisted()
    if args.train_tf_bs:
        train_tf_bs()

    if not (args.demo_costco or args.train_nonlisted or args.train_tf_bs):
        parser.print_help()


if __name__ == "__main__":
    main()
