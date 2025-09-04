# 🚀 Xavier Git Clone 加速方案

## 方案1：使用国内镜像（最快）

### GitHub镜像
```bash
# 原地址：https://github.com/username/repo.git
# 改为以下镜像之一：

# 1. ghproxy.com (推荐)
git clone https://ghproxy.com/https://github.com/zhihaozhao/paperA.git

# 2. gitclone.com
git clone https://gitclone.com/github.com/zhihaozhao/paperA.git

# 3. fastgit.org
git clone https://hub.fastgit.xyz/zhihaozhao/paperA.git

# 4. cnpmjs.org
git clone https://github.com.cnpmjs.org/zhihaozhao/paperA.git
```

## 方案2：使用代理

### 设置HTTP代理
```bash
# 如果有代理服务器
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

# 取消代理
git config --global --unset http.proxy
git config --global --unset https.proxy
```

### 只对GitHub使用代理
```bash
git config --global http.https://github.com.proxy socks5://127.0.0.1:7890
git config --global https.https://github.com.proxy socks5://127.0.0.1:7890
```

## 方案3：浅克隆（减少下载量）

```bash
# 只克隆最新版本，不要历史
git clone --depth=1 https://github.com/zhihaozhao/paperA.git

# 只克隆特定分支
git clone --depth=1 --branch feat/enhanced-model-and-sweep https://github.com/zhihaozhao/paperA.git
```

## 方案4：直接传输文件（最简单）

### 从您的电脑直接传到Xavier

```bash
# 1. 在您的电脑上，只打包需要的文件
cd /workspace
tar czf experiments.tar.gz paper/paper2_pase_net/experiments/ src/models.py

# 2. 用SCP传到Xavier（局域网内会很快）
scp experiments.tar.gz xavier@<xavier-ip>:~/

# 3. 在Xavier上解压
ssh xavier@<xavier-ip>
tar xzf experiments.tar.gz
```

## 方案5：只复制必要文件（推荐）

### 创建最小实验包
```bash
# 在您的电脑上创建最小包
mkdir -p xavier_minimal
cp paper/paper2_pase_net/experiments/xavier_simple_test.py xavier_minimal/
cp src/models.py xavier_minimal/  # 如果需要

# 压缩
tar czf xavier_minimal.tar.gz xavier_minimal/

# 传输（文件很小，几秒完成）
scp xavier_minimal.tar.gz xavier@<xavier-ip>:~/
```

## 🎯 最快方案：直接运行独立脚本

由于`xavier_simple_test.py`是独立的，不需要整个代码库：

```bash
# 方法1：直接复制脚本内容
# 在Xavier上创建文件
cat > xavier_test.py << 'EOF'
[粘贴 xavier_simple_test.py 的内容]
EOF

# 方法2：使用wget从GitHub获取单个文件
wget https://raw.githubusercontent.com/zhihaozhao/paperA/feat/enhanced-model-and-sweep/paper/paper2_pase_net/experiments/xavier_simple_test.py

# 或使用镜像
wget https://raw.fastgit.org/zhihaozhao/paperA/feat/enhanced-model-and-sweep/paper/paper2_pase_net/experiments/xavier_simple_test.py
```

## 🔧 网络优化建议

### 1. 修改DNS
```bash
# 使用阿里DNS
sudo bash -c 'echo "nameserver 223.5.5.5" > /etc/resolv.conf'
sudo bash -c 'echo "nameserver 223.6.6.6" >> /etc/resolv.conf'

# 或使用腾讯DNS
sudo bash -c 'echo "nameserver 119.29.29.29" > /etc/resolv.conf'
```

### 2. 修改hosts文件
```bash
# 添加GitHub加速
sudo bash -c 'cat >> /etc/hosts << EOF
140.82.114.4 github.com
185.199.108.153 assets-cdn.github.com
199.232.69.194 github.global.ssl.fastly.net
EOF'
```

### 3. 使用aria2多线程下载
```bash
# 安装aria2
sudo apt-get install aria2

# 使用aria2下载
aria2c -x 16 -s 16 https://github.com/zhihaozhao/paperA/archive/refs/heads/feat/enhanced-model-and-sweep.zip
```

## ✅ 推荐方案总结

**如果Xavier和您的电脑在同一局域网：**
```bash
# 最快！直接SCP传输
scp xavier_simple_test.py xavier@192.168.1.x:~/
```

**如果需要从互联网下载：**
```bash
# 使用镜像站
git clone --depth=1 https://ghproxy.com/https://github.com/zhihaozhao/paperA.git
```

**最简单方案：**
1. 我把脚本内容直接给您
2. 您复制粘贴到Xavier上
3. 立即运行，无需下载

需要我把完整的`xavier_simple_test.py`内容整理成可以直接复制的格式吗？