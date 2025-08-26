Parameter and Metrics Template

Capture from code (do not infer from paper unless explicitly reported):

- Model: <name> (citation key)
- Params (M): <compute via torchinfo/model.numel>
- FLOPs / Throughput: <ptflops/fvcore or timed inference>
- Input (T x F x C): <shape>
- Attention: <type if any>
- Dataset/Protocol: <dataset + LOSO/LORO/ID>
- Metric: <macro F1 / acc / ECE / NLL>
- Source: <code commit / paper reported>
- Notes: <deviations, preprocessing>
