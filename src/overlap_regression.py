import numpy as np
from scipy import stats

def regression_slope_pval(x, y):
    slope, intercept, r, p, std = stats.linregress(x, y)
    return dict(slope=float(slope), r=float(r), p=float(p))