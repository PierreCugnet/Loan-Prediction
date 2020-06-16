# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 09:48:41 2020

@author: Pierre Cugnet

Project Description: Personnal project
"""

import pandas as pd
import time
import re
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("loan_dataset.csv")

#A bit of data exploration
df.describe()
df.head()
df.info()

df=df.drop('Loan_ID',axis=1)
df

