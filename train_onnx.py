"""
训练 ONNX 模型（一键流程）：
1. 聚合 data/*/arknights.csv → 根目录 arknights.csv
2. 运行 train.py 训练 PyTorch 模型
3. 转换最佳模型 → models/best_model_full.onnx
4. 兼容 predict_onnx.py

用法: uv run python train_onnx.py
"""
import sys
import os
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
AGGREGATED_CSV = BASE_DIR / "arknights.csv"


def aggregate_data():
    """聚合所有 data/*/arknights.csv 到根目录"""
    if not DATA_DIR.exists():
        logger.error("data/ 目录不存在，请先收集数据")
        return False

    csv_files = list(DATA_DIR.rglob("arknights.csv"))
    if not csv_files:
        logger.error("data/ 下没有找到 CSV 数据文件")
        return False

    import pandas as pd
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, header=0)
            if len(df) > 0:
                dfs.append(df)
                logger.info(f"  + {f} ({len(df)} 条)")
        except Exception as e:
            logger.warning(f"  跳过 {f}: {e}")

    if not dfs:
        logger.error("没有有效的数据文件")
        return False

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(AGGREGATED_CSV, index=False)
    logger.info(f"✅ 聚合完成: 共 {len(combined)} 条数据 → {AGGREGATED_CSV}")
    return True


def run_training():
    """运行 train.py 训练"""
    logger.info("=" * 50)
    logger.info("开始训练 PyTorch 模型...")
    logger.info("=" * 50)

    # 确保 torch 可用
    try:
        import torch
    except ImportError:
        logger.error("需要 PyTorch。请运行: uv sync --extra cpu")
        return False

    # train.py 会读取根目录的 arknights.csv
    sys.path.insert(0, str(BASE_DIR))
    from train import main as train_main
    train_main()
    return True


def find_latest_pth():
    """找到最近训练出的 .pth 模型"""
    pth_files = list(MODELS_DIR.glob("best_model_full*.pth"))
    if not pth_files:
        logger.error(f"{MODELS_DIR}/ 下未找到 best_model_full*.pth 文件")
        return None
    latest = max(pth_files, key=os.path.getmtime)
    logger.info(f"找到最新模型: {latest}")
    return str(latest)


def convert_to_onnx(pth_path):
    """转换 .pth → .onnx (2输入兼容版)"""
    logger.info(f"正在将 {pth_path} 转换为 ONNX...")
    from predict import CannotModel

    model = CannotModel(pth_path)
    model.load_model()

    output_onnx = str(MODELS_DIR / "best_model_full.onnx")
    model.export_onnx_v2(output_onnx)

    logger.info(f"✅ ONNX 模型已生成: {output_onnx}")
    return output_onnx


def cleanup():
    """清理根目录的临时 csv"""
    if AGGREGATED_CSV.exists():
        AGGREGATED_CSV.unlink()
        logger.info(f"已清理临时文件: {AGGREGATED_CSV}")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not aggregate_data():
        sys.exit(1)

    if not run_training():
        sys.exit(1)

    latest_pth = find_latest_pth()
    if not latest_pth:
        sys.exit(1)

    onnx_path = convert_to_onnx(latest_pth)

    cleanup()

    # 验证 ONNX 模型
    logger.info("=" * 50)
    logger.info("验证 ONNX 模型")
    logger.info("=" * 50)
    import predict_onnx
    import numpy as np
    from recognize import MONSTER_COUNT

    onnx_model = predict_onnx.CannotModel(model_path=onnx_path)
    if onnx_model.is_model_loaded:
        left = np.zeros(MONSTER_COUNT, dtype=np.int64)
        right = np.zeros(MONSTER_COUNT, dtype=np.int64)
        left[28] = 16
        right[30] = 22
        pred = onnx_model.get_prediction(left, right)
        logger.info(f"验证预测: {pred:.4f}")
        logger.info("✅ 训练+转换完成，ONNX 模型就绪！")
    else:
        logger.error("❌ ONNX 模型验证失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
