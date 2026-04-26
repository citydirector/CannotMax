"""
合成数据生成器（用于测试训练流程）
功能：生成符合60怪物格式的合成数据，用于验证训练流程
注意：这仅用于测试，实际训练应使用真实游戏数据
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(num_samples=100, output_file="synthetic_training_data.csv"):
    """
    生成合成训练数据

    Args:
        num_samples: 生成的样本数量
        output_file: 输出文件名
    """
    logger.info(f"开始生成 {num_samples} 条合成数据...")

    # 创建列名
    left_cols = [f"{i}L" for i in range(1, 61)]
    right_cols = [f"{i}R" for i in range(1, 61)]
    columns = left_cols + right_cols + ["Result"]

    data = []
    for _ in range(num_samples):
        row = {}

        # 左侧怪物：随机选择1-3种怪物，每种1-5个
        num_left_types = np.random.randint(1, 4)
        left_monsters = np.random.choice(range(60), num_left_types, replace=False)
        for monster_id in left_monsters:
            col_name = f"{monster_id + 1}L"
            row[col_name] = np.random.randint(1, 6)

        # 右侧怪物：随机选择1-3种怪物，每种1-5个
        num_right_types = np.random.randint(1, 4)
        right_monsters = np.random.choice(range(60), num_right_types, replace=False)
        for monster_id in right_monsters:
            col_name = f"{monster_id + 1}R"
            row[col_name] = np.random.randint(1, 6)

        # 填充未选择的怪物为0
        for col in left_cols + right_cols:
            if col not in row:
                row[col] = 0

        # 随机生成胜负结果（可以根据需要调整概率）
        row["Result"] = np.random.choice(["L", "R"], p=[0.5, 0.5])

        data.append(row)

    df = pd.DataFrame(data, columns=columns)

    # 保存到文件
    df.to_csv(output_file, index=False)
    logger.info(f"✓ 合成数据已保存到: {output_file}")
    logger.info(f"  数据维度: {df.shape}")

    # 统计信息
    left_wins = (df["Result"] == "L").sum()
    right_wins = (df["Result"] == "R").sum()
    logger.info(f"  左胜: {left_wins}, 右胜: {right_wins}")

    return output_file


def merge_with_real_data(synthetic_file, real_data_dir="data", output_file="merged_training_data.csv"):
    """
    将合成数据与真实数据合并（用于测试）

    Args:
        synthetic_file: 合成数据文件路径
        real_data_dir: 真实数据目录
        output_file: 输出文件路径
    """
    logger.info("合并合成数据与真实数据...")

    # 读取合成数据
    synthetic_df = pd.read_csv(synthetic_file)
    logger.info(f"  合成数据: {len(synthetic_df)} 条")

    # 收集真实数据
    from pathlib import Path
    csv_files = list(Path(real_data_dir).rglob("arknights.csv"))
    real_dfs = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                real_dfs.append(df)
        except Exception as e:
            logger.warning(f"读取 {csv_file} 失败: {e}")

    if real_dfs:
        real_df = pd.concat(real_dfs, ignore_index=True)
        logger.info(f"  真实数据: {len(real_df)} 条")

        # 合并
        merged_df = pd.concat([real_df, synthetic_df], ignore_index=True)
        logger.info(f"  合并后: {len(merged_df)} 条")

        # 保存
        merged_df.to_csv(output_file, index=False)
        logger.info(f"✓ 合并数据已保存到: {output_file}")

        return output_file
    else:
        logger.warning("没有找到真实数据，只使用合成数据")
        return synthetic_file


def main():
    logger.info("=" * 60)
    logger.info("合成数据生成器（测试用）")
    logger.info("=" * 60)
    logger.info("\n⚠ 警告: 此工具仅用于测试训练流程")
    logger.info("   实际训练应使用真实游戏数据\n")

    # 生成合成数据
    num_samples = 200  # 可以调整这个值
    synthetic_file = generate_synthetic_data(num_samples)

    logger.info(f"\n生成的文件可用于:")
    logger.info(f"  1. 测试训练流程: uv run python train_onnx.py")
    logger.info(f"  2. 验证模型结构是否正确")
    logger.info(f"  3. 检查训练脚本是否有bug")

    logger.info(f"\n注意:")
    logger.info(f"  - 合成数据的预测能力很差，不应用于实际游戏")
    logger.info(f"  - 请继续收集真实数据，达到500+条后再重新训练")


if __name__ == "__main__":
    main()
