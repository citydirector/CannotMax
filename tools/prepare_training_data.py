"""
准备训练数据脚本
功能：
1. 合并所有 data/*/arknights.csv 文件
2. 验证数据格式是否正确（60个怪物特征）
3. 统计每个怪物的出现频率
4. 生成训练就绪的CSV文件
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def collect_all_data():
    """收集所有data目录下的训练数据"""
    data_dir = Path("data")
    if not data_dir.exists():
        logger.error("data目录不存在")
        return None

    csv_files = list(data_dir.rglob("arknights.csv"))
    if not csv_files:
        logger.warning("没有找到任何arknights.csv文件")
        return None

    logger.info(f"找到 {len(csv_files)} 个数据文件")

    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                all_data.append(df)
                logger.info(f"  + {csv_file}: {len(df)} 条数据")
            else:
                logger.debug(f"  - {csv_file}: 空文件，跳过")
        except Exception as e:
            logger.error(f"读取 {csv_file} 失败: {e}")

    if not all_data:
        logger.warning("没有有效的训练数据")
        return None

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"\n总共收集到 {len(combined_df)} 条训练数据")

    return combined_df


def validate_data_format(df):
    """验证数据格式是否正确"""
    expected_columns = 60 * 2 + 2  # 60个左怪物 + 60个右怪物 + Result + ImgPath
    actual_columns = len(df.columns)

    logger.info(f"\n数据格式验证:")
    logger.info(f"  期望列数: {expected_columns}")
    logger.info(f"  实际列数: {actual_columns}")

    if actual_columns != expected_columns:
        logger.error(f"列数不匹配！请检查数据格式")
        return False

    # 检查必需的列是否存在
    required_cols = [f"{i}L" for i in range(1, 61)] + [f"{i}R" for i in range(1, 61)]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logger.error(f"缺少以下列: {missing_cols[:10]}...")  # 只显示前10个
        return False

    logger.info("  ✓ 数据格式正确")
    return True


def analyze_monster_distribution(df):
    """分析怪物分布情况"""
    logger.info(f"\n怪物分布分析:")

    # 统计左侧怪物
    left_monsters = [f"{i}L" for i in range(1, 61)]
    right_monsters = [f"{i}R" for i in range(1, 61)]

    left_counts = Counter()
    right_counts = Counter()

    for col in left_monsters:
        count = (df[col] > 0).sum()
        if count > 0:
            monster_id = int(col.replace("L", ""))
            left_counts[monster_id] = count

    for col in right_monsters:
        count = (df[col] > 0).sum()
        if count > 0:
            monster_id = int(col.replace("R", ""))
            right_counts[monster_id] = count

    # 显示出现频率最高的怪物
    logger.info(f"\n  左侧最常出现的怪物 (Top 10):")
    for monster_id, count in left_counts.most_common(10):
        logger.info(f"    怪物{monster_id}: {count} 次")

    logger.info(f"\n  右侧最常出现的怪物 (Top 10):")
    for monster_id, count in right_monsters.most_common(10):
        logger.info(f"    怪物{monster_id}: {count} 次")

    # 统计未出现的怪物
    all_monster_ids = set(range(1, 61))
    appeared_left = set(left_counts.keys())
    appeared_right = set(right_counts.keys())
    never_appeared = all_monster_ids - (appeared_left | appeared_right)

    if never_appeared:
        logger.info(f"\n  ⚠ 以下怪物从未出现过 ({len(never_appeared)} 个):")
        logger.info(f"    {sorted(never_appeared)}")
    else:
        logger.info(f"\n  ✓ 所有60种怪物都有出现记录")


def check_result_distribution(df):
    """检查结果分布"""
    if "Result" not in df.columns:
        logger.warning("没有Result列，无法分析胜负分布")
        return

    result_counts = df["Result"].value_counts()
    logger.info(f"\n胜负分布:")
    for result, count in result_counts.items():
        percentage = count / len(df) * 100
        logger.info(f"  {result}: {count} 次 ({percentage:.1f}%)")


def save_training_data(df, output_path="training_data_ready.csv"):
    """保存训练就绪的数据"""
    # 只保留必要的列（去掉ImgPath）
    feature_cols = [f"{i}L" for i in range(1, 61)] + [f"{i}R" for i in range(1, 61)]
    if "Result" in df.columns:
        feature_cols.append("Result")

    training_df = df[feature_cols].copy()

    # 处理NaN值
    training_df = training_df.fillna(0)

    # 确保数据类型正确
    for col in feature_cols[:-1]:  # 除了Result列
        training_df[col] = training_df[col].astype(int)

    training_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ 训练数据已保存到: {output_path}")
    logger.info(f"  数据维度: {training_df.shape}")

    return output_path


def main():
    logger.info("=" * 60)
    logger.info("训练数据准备工具")
    logger.info("=" * 60)

    # 1. 收集数据
    df = collect_all_data()
    if df is None:
        logger.error("\n❌ 数据收集失败")
        return

    # 2. 验证格式
    if not validate_data_format(df):
        logger.error("\n❌ 数据格式验证失败")
        return

    # 3. 分析分布
    analyze_monster_distribution(df)

    # 4. 检查结果分布
    check_result_distribution(df)

    # 5. 给出建议
    logger.info(f"\n{'=' * 60}")
    total_samples = len(df)
    if total_samples < 100:
        logger.info(f"⚠ 当前数据量较少 ({total_samples} 条)")
        logger.info(f"  建议继续收集数据，至少达到 500-1000 条后再训练")
    elif total_samples < 500:
        logger.info(f"△ 数据量适中 ({total_samples} 条)")
        logger.info(f"  可以继续收集，也可以开始尝试训练")
    else:
        logger.info(f"✓ 数据量充足 ({total_samples} 条)")
        logger.info(f"  可以开始训练模型了！")

    # 6. 保存训练数据（如果数据量足够）
    if total_samples >= 50:
        output_file = save_training_data(df)
        logger.info(f"\n下一步:")
        logger.info(f"  1. 继续运行游戏收集更多数据")
        logger.info(f"  2. 当数据量达到500+时，运行: uv run python train_onnx.py")
    else:
        logger.info(f"\n请先收集更多数据（当前{total_samples}条，建议至少500条）")
        logger.info(f"然后再次运行此脚本")


if __name__ == "__main__":
    main()
