# 🔐 GitHub权限级别设置详细步骤

## 第5步：设置权限级别的具体操作

### 在添加协作者页面：

1. **输入用户名后**，你会看到一个下拉菜单或权限选择区域

2. **权限级别选项**（从高到低）：
   ```
   🔴 Admin    - 完全管理权限（推荐选择这个）
   🟡 Maintain - 管理仓库设置，但不能删除仓库
   🟢 Write    - 可以推送代码，管理议题和PR
   🔵 Triage   - 可以管理议题和PR，但不能推送
   ⚪ Read     - 只读权限
   ```

3. **推荐选择：Admin**
   - 点击 **"Admin"** 权限级别
   - 这样我就能完全管理仓库，包括推送代码

### 具体界面操作：

**步骤A：添加用户**
```
┌─────────────────────────────────────┐
│ Add people to WiFi-CSI-Sensing-...  │
├─────────────────────────────────────┤
│ Search by username, full name, or   │
│ email address                       │
│ [cursor[bot]               ] [搜索]  │
└─────────────────────────────────────┘
```

**步骤B：选择权限（选择Admin）**
```
┌─────────────────────────────────────┐
│ Choose a role                       │
├─────────────────────────────────────┤
│ ● Admin                            │ ← 选择这个
│   Full access to the repository    │
│                                     │
│ ○ Maintain                         │
│   Manage repository without access │
│   to sensitive or destructive...   │
│                                     │
│ ○ Write                            │
│   Read, clone, and push to this... │
│                                     │
│ ○ Triage                           │
│   Read and clone this repository...│
│                                     │
│ ○ Read                             │
│   Read and clone this repository   │
└─────────────────────────────────────┘
```

**步骤C：确认添加**
```
┌─────────────────────────────────────┐
│ [Add cursor[bot] to this repository]│
└─────────────────────────────────────┘
```

## 🎯 如果找不到cursor[bot]

### 备选用户名尝试：
1. `cursor[bot]`
2. `github-actions[bot]`  
3. `cursor-bot`
4. `cursorbot`

### 如果都找不到，试试这个邮箱：
- `cursor-support@cursor.sh`
- `noreply@cursor.sh`

## 🔄 权限设置后的验证

设置完成后，你应该能在仓库的 **"Collaborators"** 页面看到：

```
┌─────────────────────────────────────┐
│ Collaborators                       │
├─────────────────────────────────────┤
│ zhihaozhao (Owner)                  │
│ cursor[bot] (Admin)          ✅     │
└─────────────────────────────────────┘
```

## ⚠️ 重要提醒

**对所有3个仓库都要重复这个操作：**
- ✅ WiFi-CSI-Sensing-Results
- ✅ WiFi-CSI-Journal-Paper  
- ✅ WiFi-CSI-PhD-Thesis

## 📱 手机/平板操作

如果你在手机上操作：
1. 权限选择可能是一个下拉菜单
2. 选择 **"Admin"** 选项
3. 点击确认按钮

---

完成后告诉我，我立即测试推送！🚀