"""
将训练好的 PyTorch 模型 (.pth) 转换为 ONNX 格式。
用法: uv run tools/convert_model.py [--input path/to/model.pth]
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import predict
import predict_onnx
import numpy as np
from recognize import MONSTER_COUNT


def replace_suffix(s):
    idx = s.rfind('.')
    if idx == -1:
        return s + '.onnx'
    else:
        return s[:idx] + '.onnx'


def main():
    parser = argparse.ArgumentParser(description="Convert .pth to .onnx")
    parser.add_argument("--input", default="models/best_model_full.pth",
                        help="Input .pth model file path")
    parser.add_argument("--output", default=None,
                        help="Output .onnx file path (default: same name with .onnx)")
    args = parser.parse_args()

    model_path = args.input
    if not Path(model_path).exists():
        print(f"错误: 未找到模型文件 {model_path}")
        sys.exit(1)

    output_path = args.output or replace_suffix(model_path)
    print(f"输入: {model_path}")
    print(f"输出: {output_path}")

    # 加载 PyTorch 模型
    model = predict.CannotModel(model_path)
    model.load_model()
    
    # 导出为兼容的 2 输入 ONNX
    model.export_onnx_v2(output_path)
    print(f"✅ ONNX 模型已保存: {output_path}")

    # 验证导出结果
    print("\n验证推理结果...")
    data_array = np.zeros(MONSTER_COUNT, dtype=np.int64)
    data_array[28] = 16
    left = data_array
    right = np.zeros(MONSTER_COUNT, dtype=np.int64)
    right[30] = 22

    onnx_model = predict_onnx.CannotModel(model_path=output_path)
    prediction = onnx_model.get_prediction(left, right)
    print(f"验证预测结果: {prediction:.4f}")
    print("✅ 验证通过")


if __name__ == "__main__":
    main()
