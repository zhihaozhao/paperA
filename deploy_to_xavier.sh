#!/bin/bash
# Xavier设备一键部署脚本
# 使用方法: ./deploy_to_xavier.sh

set -e

XAVIER_IP="192.168.2.36"
XAVIER_USER="nvidia"
XAVIER_PATH="~/workspace_PHD/paperA"

echo "🚀 Xavier设备部署脚本"
echo "📱 目标设备: $XAVIER_USER@$XAVIER_IP"
echo "📂 目标路径: $XAVIER_PATH"
echo ""

# 检查必要文件
echo "🔍 检查必要文件..."
required_files=(
    "measure_all_models_xavier.py"
    "measure_conformer_lite_xavier.py" 
    "run_conformer_lite_xavier.sh"
    "src/models.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 缺少文件: $file"
        exit 1
    fi
    echo "✅ 找到文件: $file"
done

# 创建部署包
echo ""
echo "📦 创建部署包..."
tar -czf xavier_deployment.tar.gz \
    measure_all_models_xavier.py \
    measure_conformer_lite_xavier.py \
    run_conformer_lite_xavier.sh \
    src/models.py \
    XAVIER_SCRIPT_UPDATE_SUMMARY.md \
    DEPLOY_TO_XAVIER_GUIDE.md

echo "✅ 部署包创建完成: xavier_deployment.tar.gz"

# 传输文件
echo ""
echo "📤 传输文件到Xavier设备..."
echo "请输入Xavier设备的密码:"
scp xavier_deployment.tar.gz $XAVIER_USER@$XAVIER_IP:$XAVIER_PATH/

if [ $? -eq 0 ]; then
    echo "✅ 文件传输成功"
else
    echo "❌ 文件传输失败"
    exit 1
fi

# 在Xavier上执行部署
echo ""
echo "🔧 在Xavier设备上执行部署..."
ssh $XAVIER_USER@$XAVIER_IP << 'EOF'
cd ~/workspace_PHD/paperA

echo "📦 解压部署包..."
tar -xzf xavier_deployment.tar.gz

echo "🔧 设置权限..."
chmod +x run_conformer_lite_xavier.sh
chmod +x measure_all_models_xavier.py
chmod +x measure_conformer_lite_xavier.py

echo "🔍 检查环境..."
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo "🎮 检查GPU状态..."
nvidia-smi

echo "📁 检查文件..."
ls -la measure_*_xavier.py run_conformer_lite_xavier.sh

echo "✅ Xavier设备部署完成！"
echo ""
echo "🚀 现在可以运行以下命令进行实验："
echo "   python3 measure_all_models_xavier.py --device cuda"
echo "   python3 measure_conformer_lite_xavier.py --device cuda"
echo "   ./run_conformer_lite_xavier.sh"
EOF

echo ""
echo "🎉 部署完成！"
echo ""
echo "📋 下一步操作："
echo "1. SSH连接到Xavier: ssh $XAVIER_USER@$XAVIER_IP"
echo "2. 进入工作目录: cd ~/workspace_PHD/paperA"
echo "3. 运行实验: ./run_conformer_lite_xavier.sh"
echo "4. 查看结果: ls -la xavier_*.json results_gpu/"
echo "5. 提交到Git: git add . && git commit -m 'Xavier D1 experiment results'"
echo ""
echo "📖 详细说明请查看: DEPLOY_TO_XAVIER_GUIDE.md"