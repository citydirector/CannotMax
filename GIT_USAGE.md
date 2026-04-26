# 本地Git使用指南

## 当前状态

✅ **已启用本地Git版本控制**
- 工作目录干净，所有修改已提交
- 创建了版本标签 `v0.2-local-60monsters`
- 与上游仓库分叉开发（未合并）

## 常用Git命令

### 查看状态
```bash
# 查看当前状态
git status

# 查看提交历史
git log --oneline -10

# 查看图形化历史
git log --graph --oneline --all -10
```

### 提交修改
```bash
# 添加所有修改
git add -A

# 提交并写消息
git commit -m "描述你的修改"

# 或者使用多行消息
git commit -m "标题

详细说明..."
```

### 版本管理
```bash
# 创建标签
git tag -a v0.3-new-feature -m "描述"

# 查看所有标签
git tag -l

# 切换到某个标签（只读）
git checkout v0.2-local-60monsters

# 返回最新提交
git checkout main
```

### 回滚操作
```bash
# 撤销未提交的修改
git restore <file>

# 撤销所有未提交修改（危险！）
git restore .

# 回退到上一个commit
git reset --hard HEAD~1

# 回退到指定commit
git reset --hard <commit-hash>
```

### 分支操作
```bash
# 创建新分支
git branch feature-xxx

# 切换分支
git checkout feature-xxx

# 创建并切换分支
git checkout -b feature-xxx

# 查看分支
git branch
```

## 当前分支结构

```
* 036e05a (HEAD -> main) docs: 添加本地版本控制说明文档
* 798e3a5 (tag: v0.2-local-60monsters) feat: 添加60怪物配置支持
| * 5c29c79 (origin/main) Update release version
| * 11e6903 Modify unpackage data path
| ...
|/  
* 642af3f Add capture mode setting
```

- **本地main**: 你的开发版本（包含60怪物功能）
- **origin/main**: 上游官方版本（包含PC端支持）
- **两者分叉**: 从 `642af3f` 开始分别开发

## 重要提示

### ⚠️ 不要随意执行
```bash
# ❌ 不要这样做（会覆盖你的本地修改）
git pull origin main

# ❌ 不要强制推送
git push -f origin main
```

### ✅ 推荐做法
```bash
# ✅ 定期提交本地修改
git add -A && git commit -m "保存进度"

# ✅ 创建标签标记重要版本
git tag -a v0.x-description -m "说明"

# ✅ 备份重要数据（不在Git中）
# - data/ 目录
# - models/ 目录
```

## 如果需要合并上游

当你准备好合并上游的PC端支持时：

```bash
# 1. 先确保本地修改已提交
git status

# 2. 创建备份分支
git branch backup-before-merge

# 3. 尝试合并（可能有冲突）
git merge origin/main

# 4. 解决冲突后提交
git add -A
git commit -m "Merge upstream with PC support"

# 5. 测试所有功能
# ...

# 6. 如果有问题，可以回滚
git reset --hard backup-before-merge
```

## 数据备份

以下目录**不在Git中**，需要手动备份：

```bash
# 推荐的备份方式
# 1. 压缩data目录
tar -czf data_backup_20260426.tar.gz data/

# 2. 复制models目录
cp -r models/ models_backup_20260426/

# 3. 使用Windows资源管理器手动复制
```

## 版本历史

| 版本 | 标签 | 说明 |
|------|------|------|
| v0.2 | `v0.2-local-60monsters` | 60怪物配置 + 自动训练 |
| v0.1 | - | 初始版本（上游） |

---

**记住**: Git是你的安全网，经常提交可以避免丢失工作成果！
