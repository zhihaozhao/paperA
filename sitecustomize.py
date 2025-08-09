import os, sys

# 当前文件的绝对路径：.../paperA/sitecustomize.py
_this_file = os.path.abspath(__file__)

# 项目根目录：.../paperA
project_root = os.path.dirname(_this_file)

# 确保项目根在 sys.path 最前，Python 能找到 src/ 包
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# import os, sys
# project_root = os.path.dirname(os.path.abspath(file))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)