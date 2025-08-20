# 🔐 GitHub仓库权限设置指南

## 方法1：给Cursor GitHub App添加权限（推荐）

### 步骤：

1. **进入每个新仓库的设置页面**
   ```
   https://github.com/zhihaozhao/WiFi-CSI-Sensing-Results/settings
   https://github.com/zhihaozhao/WiFi-CSI-Journal-Paper/settings  
   https://github.com/zhihaozhao/WiFi-CSI-PhD-Thesis/settings
   ```

2. **在左侧菜单点击 "Collaborators and teams"**

3. **点击 "Add people" 按钮**

4. **搜索并添加以下账号之一：**
   - `cursor[bot]` (Cursor的GitHub Bot账号)
   - `github-actions[bot]` (GitHub Actions Bot)
   - 或者输入邮箱：`cursor-support@cursor.sh`

5. **设置权限级别为 "Maintain" 或 "Admin"**
   - **Admin**: 完全控制权限（推荐）
   - **Maintain**: 管理仓库但不能删除

6. **对所有3个新仓库重复上述步骤**

## 方法2：使用GitHub Personal Access Token

### 步骤：

1. **生成新的Personal Access Token**
   - 访问：https://github.com/settings/tokens
   - 点击 "Generate new token" → "Generate new token (classic)"
   - 设置名称：`WiFi-CSI-Multi-Repo-Access`
   - 选择权限：
     - ✅ `repo` (Full control of private repositories)
     - ✅ `workflow` (Update GitHub Action workflows)

2. **复制生成的token（只显示一次）**

3. **告诉我这个token，我用它来推送**

## 方法3：邀请协作者（备选）

如果上述方法不行，你可以：

1. **邀请一个GitHub账号作为协作者**
2. **我使用该账号的凭据进行推送**

---

## 🚀 推荐操作流程

### 最简单的方法：

1. **现在立即执行方法1**（给cursor[bot]添加权限）
2. **我验证权限是否生效**
3. **我直接推送所有3个子仓库**

### 如果方法1不行：

1. **执行方法2**（生成Personal Access Token）
2. **私密方式告诉我token**
3. **我使用token推送**

---

## ⏱️ 预计时间

- **设置权限**：5分钟
- **我推送所有仓库**：2分钟
- **验证和测试**：3分钟

**总计：10分钟内完成！**

---

你想先尝试哪种方法？我建议从**方法1**开始，这是最标准的做法。