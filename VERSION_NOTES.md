# 本地版本控制说明

## 当前版本

**Tag**: `v0.2-local-60monsters`  
**Commit**: `798e3a5`  
**日期**: 2026-04-26

## 版本特性

### 核心功能
1. ✅ **60怪物配置支持**
   - 基于 `monster_greenvine.csv`
   - 临时维度适配（60→77填充）
   - 所有60种怪物图片齐全

2. ✅ **自动数据收集和训练**
   - 一键完成：忽略旧模型 → 收集数据 → 训练新模型
   - 固定选左策略，确保数据一致性
   - 实时剩余时间显示（每秒更新）
   - 支持中途停止并保留数据

3. ✅ **GUI增强**
   - 新增绿色"从0开始收集数据并训练"按钮
   - 新增红色"停止"按钮
   - 状态栏实时更新

### 工具脚本
- `tools/prepare_training_data.py` - 数据准备和验证
- `tools/generate_synthetic_data.py` - 合成数据生成器（测试用）
- `test_auto_collect.py` - 自动化测试

### 文档
- `AUTO_COLLECT_GUIDE.md` - 详细使用指南
- `QUICK_START.md` - 快速开始
- `IMPLEMENTATION_SUMMARY.md` - 实施总结
- `FIXES_APPLIED.md` - 修复说明
- `README_NEW_FEATURE.md` - 新功能说明

## Git历史

```
* 798e3a5 (HEAD -> main, tag: v0.2-local-60monsters) feat: 添加60怪物配置支持和自动数据收集训练功能
| * 5c29c79 (origin/main) Update release version
| * 11e6903 Modify unpackage data path
| * 46bc3c5 Optimize the package size
| *   e2e8f95 Merge remote-tracking branch 'origin/main'
| |\  
| | * 413af3d Add data_train path and update uv package
| * | 2755a5a 支持pc端后台模式
| |/  
| * bd9af13 支持pc端前台模式
|/  
* 642af3f Add capture mode setting and adb select
```

## 与上游的关系

- **本地分支**: `main` (领先上游7个commit的并行开发)
- **上游分支**: `origin/main` (有7个新commit未合并)
- **状态**: 分叉开发，尚未合并

### 上游未合并的功能
1. PC端支持（前台/后台模式）
2. 连接器抽象化（`adb_connector` → `connector`）
3. 模型路径自动查找
4. 颜色检测算法优化
5. 数据打包改进

## 使用建议

### 日常使用
当前版本完全可用，专注于60怪物配置的数据收集和训练。

### 何时考虑合并上游
- 需要PC端支持时
- 当前功能稳定运行一段时间后
- 准备好进行充分测试时

### 如何回滚
如果需要回到某个版本：
```bash
# 查看历史
git log --oneline

# 回退到指定commit
git reset --hard <commit-hash>

# 或者切换到标签
git checkout v0.2-local-60monsters
```

## 备份建议

重要数据目录（不在Git中）：
- `data/` - 收集的对战数据
- `models/` - 训练的模型文件
- `images/` - 怪物图片

建议定期手动备份这些目录。

## 下一步计划

1. 使用当前版本收集足够数据（500+条）
2. 训练出高质量的60维模型
3. 测试验证模型效果
4. 考虑是否合并上游的PC端支持

---

**注意**: 这是本地开发版本，用于个人使用。如需与团队协作或发布，需要考虑与上游合并。
