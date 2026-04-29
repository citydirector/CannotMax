# CannotMax-Greenvine

一个基于深度学习的明日方舟游戏辅助工具，自动识别战斗画面、预测胜率、收集数据、训练模型。

## 功能特点

- **多模式画面捕获**：
  - **ADB 模式**：适配雷电、MuMu、蓝叠等主流模拟器
  - **PC 模式**：适配明日方舟官方 PC 客户端，支持 MaaFramework 后台截图/点击（游戏可最小化）
  - **WIN 模式**：基于 WinRT 的高性能窗口/屏幕截取
- **会话名机制**：数据/模型按名称分组隔离（如 `greenvine`），同名会话自动追加数据、基于旧模型微调
- **全自动化流程**：一键收集数据 → 定时停止 → 备份旧模型 → 基于历史模型微调 → 导出新 ONNX
- **投资策略可选**：跟随 GUI "投资" 复选框，可启用模型预测后投资（有偏数据），或禁用预测固定观望（纯净数据）
- **深度学习预测**：支持 PyTorch (CUDA 13.0 加速) 训练 + ONNX Runtime 推理，自动选择 CUDA / DirectML / CPU 执行提供器
- **NPU/GPU 加速推理**：通过 ONNX Runtime DirectML 提供器，支持 Intel/AMD NPU 及任意 GPU
- **历史匹配**：与历史战斗记录进行相似度匹配，自动左右镜像对比
- **自动登录**：断线重连、自动登录游戏
- **多开管理**：支持多个实例并行运行
- **暗色主题**：适配暗色模式的 UI 样式

## 手动安装步骤

1. **安装 uv**：参考 [uv 官方文档](https://docs.astral.sh/uv/getting-started/installation)

2. **配置环境**：
   ```bash
   # 安装基础依赖（不含 PyTorch）
   uv sync
   # (可选) CUDA 13.0 加速训练
   uv sync --extra cu130
   # (可选) 仅 CPU 运行
   uv sync --extra cpu
   ```

3. **运行主程序**：
   ```bash
   uv run main.py
   ```

## 使用指南

### 1. 捕获模式选择
- **ADB**：输入或选择模拟器序列号（如 `127.0.0.1:5555`）后连接
- **PC**：连接明日方舟 PC 客户端，自动启用 MaaFramework 后台操作
- **WIN**：点击"选择窗口"按钮，通过 WinRT 捕获指定窗口

### 2. 核心操作
- **自动获取数据**：按会话名追加到 `data/<会话名>/arknights.csv`，同会话名累积复用
- **训练设备选择**：自动检测 / CUDA / NPU (DirectML) / CPU
- **一键收集并训练**：收集数据 → 备份旧模型 → 微调训练 → 导出 ONNX，全自动
- **预测/识别**：手动分析当前画面，基于当前识别到的单位进行胜率预测

### 3. 数据收集
1. 模拟器中打开争锋频道页面
2. 填写会话名（如 `greenvine`），勾选是否投资
3. 点击"自动获取数据"或"一键收集并训练"
4. 到达设定时长后自动停止，或手动点击"停止"（可选择"仅停止"或"停止并训练"）

### 4. 模型训练
- 建议收集 100+ 条数据后再训练
- 一键流程会自动使用同会话名历史 `.pth` 做 warm-start 微调
- 也可通过"训练 ONNX 模型"按钮或 `uv run python train_onnx.py --session greenvine` 手动触发

## 注意事项

- **分辨率**：模拟器/客户端建议 `1920x1080`
- **Python 版本**：需要 `>=3.11`（onnxruntime 1.24+ 不再支持 3.10）
- **MaaFramework**：PC 后台模式需要 `uv sync` 自动安装 `maafw` 包
- **依赖冲突**：如遇到 OpenCV 报错，删除 `opencv-python-headless`，保留 `opencv-python`

## 主要文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | 主程序 GUI |
| `auto_fetch.py` | 自动数据采集引擎（状态机） |
| `auto_collect_and_train.py` | 一键收集+训练工作流 |
| `train.py` | PyTorch 模型训练 |
| `train_onnx.py` | 训练 + ONNX 导出全流程 |
| `predict.py` | PyTorch 模型预测 |
| `predict_onnx.py` | ONNX Runtime 推理（支持 CUDA/DirectML/CPU） |
| `login.py` | 自动登录模块 |
| `multi_instance.py` | 多开管理器 |
| `recognize.py` | 图像识别与 OCR |
| `loadData.py` | 连接器抽象（ADB/PC/WIN） |
| `maa_adb_connector.py` | MaaFramework ADB 连接器 |
| `data_package.py` | 数据打包工具 |
| `dark_mode_style_fix.py` | 暗色模式 UI 样式 |

---

基于 [Ancientea/CannotMax](https://github.com/Ancientea/CannotMax) 上游开发
