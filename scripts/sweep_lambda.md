使用示例与参数讲解
- 全量跑，断点续跑（已完成的跳过）
  - python scripts/sweep_lambda.py --resume
- 强制重跑所有（忽略已有 JSON）
  - python scripts/sweep_lambda.py --force
- 只跑 mid 难度、seed=0 的 λ∈{0.0,0.05,0.12}，采用 learnable 温度
  - python scripts/sweep_lambda.py --only mid:0 --lambdas 0.0,0.05,0.12 --temp_mode learnable --resume
- 用区间语法扫描 λ 从 0.0 到 0.2 步长 0.02，难度 easy,mid，seeds 0,1
  - python scripts/sweep_lambda.py --difficulties easy,mid --seeds 0,1 --lambdas 0.0:0.2:0.02 --resume
- 指定自定义训练脚本路径（相对项目根）
  - python scripts/sweep_lambda.py --train src/train_eval.py --resume

参数要点
- --lambdas
  - 逗号分隔或区间步进（start:end:step），会去重并排序
- --resume / --force
  - 二选一。--resume 会跳过“已存在且可解析”的 JSON；--force 则无视已有文件
- --only
  - 形如 mid:0 或 easy:all;mid:1，多段用分号分隔
- --difficulties, --seeds
  - 不使用 --only 时，这两个控制全量组合
- 输出
  - 成功结果写入 results/synth_lambda/metrics_lambda_full.csv
  - 控制台打印每个 (difficulty, seed) 的最佳 λ（基于成功结果）