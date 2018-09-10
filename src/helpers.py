# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 20:58:38 2018

@author: Saurabh
"""

import pandas as pd
import numpy as np
from arch import arch_model
import os

os.chdir("C:\\Users\\Saurabh\\Documents\\Python Scripts\\Fall\\230T\\Project")

implied_vol = pd.read_csv(".//data//implied_vol.csv")

# garch_model = arch_model(y=implied_vol["impl_volatility"],
#                         mean="Constant", p=1, q=1)
# model_result = garch_model.fit()


def fit_garch_model(ts=implied_vol["impl_volatility"], p=1, q=1):
    ''' Takes in the time series returns the parameters. Default params are p=1
    and q=1 '''
    garch_model = arch_model(y=ts, mean="Constant",
                             p=1, q=1)
    model_result = garch_model.fit()
    # params = model_result.params
    return(model_result)
