# 📥 从服务器下载文件的方法

## 🎯 目标文件
- **服务器路径**：`/workspace/wifi-csi-repos-backup.tar.gz`
- **文件大小**：4.5MB
- **下载到**：`D:\workspace_AI\WiFi-CSI-Project\`

## 方法1：SCP命令（如果你有SSH访问）

### 在Windows PowerShell中：
```powershell
# 进入目标目录
cd D:\workspace_AI\WiFi-CSI-Project

# 使用SCP下载（替换YOUR_SERVER为你的服务器地址）
scp your_username@YOUR_SERVER:/workspace/wifi-csi-repos-backup.tar.gz .

# 或者如果有SSH密钥：
scp -i path\to\your\key.pem username@server:/workspace/wifi-csi-repos-backup.tar.gz .
```

## 方法2：SFTP工具

### 使用WinSCP（推荐）
1. **下载WinSCP**：https://winscp.net/
2. **连接到服务器**
3. **导航到** `/workspace/`
4. **下载** `wifi-csi-repos-backup.tar.gz`

### 使用FileZilla
1. **下载FileZilla**：https://filezilla-project.org/
2. **设置SFTP连接**
3. **下载文件**

## 方法3：wget/curl（如果Windows有这些工具）

```powershell
# 如果服务器支持HTTP下载
wget http://your-server/path/wifi-csi-repos-backup.tar.gz
# 或
curl -O http://your-server/path/wifi-csi-repos-backup.tar.gz
```

## 方法4：VSCode/Cursor远程功能

如果你在用Cursor/VSCode连接服务器：
1. **在远程文件浏览器中找到文件**
2. **右键点击文件**
3. **选择"Download"**

## 方法5：通过Web界面（如果有）

如果服务器有Web文件管理界面：
1. **访问服务器Web管理面板**
2. **导航到** `/workspace/`
3. **下载文件**

## 方法6：我创建HTTP下载链接

我可以在服务器上创建一个临时HTTP服务：

```bash
# 在服务器上执行
cd /workspace
python3 -m http.server 8000

# 然后你可以通过浏览器访问：
# http://your-server-ip:8000/wifi-csi-repos-backup.tar.gz
```