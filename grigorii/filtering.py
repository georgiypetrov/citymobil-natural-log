import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder

if not os.path.exists('data/train.csv'):
    print(f'Error! Cannot find path: `data/train.csv`')
    exit()

df = pd.read_csv('data/train.csv')

