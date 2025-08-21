#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
root=Path(__file__).resolve().parent
m= pd.read_csv(root/"master.csv")
for c in ["map50","small_ap","fps","success_rate","cycle_time_s","damage_rate"]:
    if c in m.columns:
        m[c]=pd.to_numeric(m[c], errors='coerce')
summary = m.groupby(['task','method']).agg({'map50':'mean','fps':'mean','success_rate':'mean','cycle_time_s':'mean','damage_rate':'mean'}).round(3)
summary.to_csv(root/"summary_by_task_method.csv")
print("Wrote summary_by_task_method.csv")
