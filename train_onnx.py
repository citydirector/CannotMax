"""
训练 ONNX 模型（一键流程）：
1. 读取 data/<session>/arknights.csv → 根目录 arknights.csv
2. 运行 train.py 训练 PyTorch 模型
3. 转换最佳模型 → models/<session>_best_model_full.onnx
4. 兼容 predict_onnx.py

用法: uv run python train_onnx.py --session greenvine
"""
import sys
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
AGGREGATED_CSV = BASE_DIR / "arknights.csv"


def aggregate_data(session_name=""):
    """聚合 session 对应的 CSV 到根目录"""
    if not DATA_DIR.exists():
        logger.error("data/ 目录不存在，请先收集数据")
        return False

    if session_name:
        target_path = DATA_DIR / session_name / "arknights.csv"
        if not target_path.exists():
            logger.error(f"数据文件不存在: {target_path}")
            return False
        csv_files = [target_path]
    else:
        csv_files = list(DATA_DIR.rglob("arknights.csv"))

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


def find_best_pretrained(session_name):
    """查找同 session 的已有模型用于微调"""
    pth_files = list(MODELS_DIR.glob(f"{session_name}_best_model_full_*.pth"))
    if not pth_files:
        return ""
    latest = max(pth_files, key=os.path.getmtime)
    logger.info(f"找到同会话已有模型（将用于微调）: {latest}")
    return str(latest)


def run_training(session_name="", pretrained_path="", device_type=""):
    """运行 train.py 训练"""
    logger.info("=" * 50)
    logger.info("开始训练 PyTorch 模型...")
    if device_type:
        logger.info(f"设备: {device_type}")
    if pretrained_path:
        logger.info(f"微调模式: 从 {pretrained_path} 继续训练")
    logger.info("=" * 50)

    try:
        import torch
    except ImportError:
        logger.error("需要 PyTorch。请运行: uv sync --extra cpu")
        return False

    sys.path.insert(0, str(BASE_DIR))
    from train import main as train_main
    train_main(session_name=session_name, pretrained_path=pretrained_path, device_type=device_type)
    return True


def find_latest_pth(session_name=""):
    """找到最近训练出的 .pth 模型"""
    prefix = f"{session_name}_" if session_name else ""
    pth_files = list(MODELS_DIR.glob(f"{prefix}best_model_full_*.pth"))
    if not pth_files:
        logger.error(f"{MODELS_DIR}/ 下未找到 {prefix}best_model_full_*.pth 文件")
        return None
    latest = max(pth_files, key=os.path.getmtime)
    logger.info(f"找到最新模型: {latest}")
    return str(latest)


def convert_to_onnx(pth_path, session_name=""):
    """转换 .pth → .onnx (2输入兼容版)"""
    logger.info(f"正在将 {pth_path} 转换为 ONNX...")
    from predict import CannotModel

    model = CannotModel(pth_path)
    model.load_model()

    prefix = f"{session_name}_" if session_name else ""
    onnx_name = f"{prefix}best_model_full.onnx"
    output_onnx = str(MODELS_DIR / onnx_name)

    # 删除旧文件，避免 .onnx.data 残留导致导出失败
    for p in [Path(output_onnx), Path(output_onnx + ".data")]:
        if p.exists():
            p.unlink()
            logger.info(f"已删除旧文件: {p}")

    model.export_onnx_v2(output_onnx)

    logger.info(f"✅ ONNX 模型已生成: {output_onnx}")
    return output_onnx


def cleanup():
    """清理根目录的临时 csv"""
    if AGGREGATED_CSV.exists():
        AGGREGATED_CSV.unlink()
        logger.info(f"已清理临时文件: {AGGREGATED_CSV}")


def cleanup_models(session_name=""):
    """训练后清理 models 目录，只保留 live ONNX + 最新一组存档 pth"""
    prefix = f"{session_name}_" if session_name else ""

    # 1. 删除临时的 .pth 文件（归档前的文件名，非存档）
    for name in ["best_model_acc.pth", "best_model_loss.pth", "best_model_full.pth"]:
        p = MODELS_DIR / f"{prefix}{name}"
        if p.exists():
            p.unlink()
            logger.info(f"已删除临时模型: {p}")

    # 2. 清理旧存档 pth，只保留每组最新的 1 个
    for suffix in ("acc", "loss", "full"):
        archived = sorted(
            MODELS_DIR.glob(f"{prefix}best_model_{suffix}_*.pth"),
            key=os.path.getmtime, reverse=True,
        )
        for f in archived[1:]:  # 保留最新，删除更早的
            f.unlink()
            logger.info(f"已清理旧存档: {f.name}")

    # 3. 清理无会话前缀的旧 onnx 文件（不再使用）
    for p in [MODELS_DIR / "best_model_full.onnx", MODELS_DIR / "best_model_full.onnx.data"]:
        if p.exists():
            p.unlink()
            logger.info(f"已清理旧默认模型: {p}")

    logger.info("✅ 模型目录清理完成")


def main(session_name="", pretrained_path="", device_type=""):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not aggregate_data(session_name):
        sys.exit(1)

    if not run_training(session_name, pretrained_path, device_type):
        sys.exit(1)

    latest_pth = find_latest_pth(session_name)
    if not latest_pth:
        sys.exit(1)

    onnx_path = convert_to_onnx(latest_pth, session_name)

    cleanup_models(session_name)

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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=str, default="", help="会话名称")
    ap.add_argument("--pretrained", type=str, default="", help="预训练模型路径（可选，自动查同名会话）")
    ap.add_argument("--device", type=str, default="", help="训练设备: cpu 或 cuda")
    args = ap.parse_args()

    pretrained_path = args.pretrained
    if args.session and not pretrained_path:
        pretrained_path = find_best_pretrained(args.session)

    main(session_name=args.session, pretrained_path=pretrained_path, device_type=args.device)
