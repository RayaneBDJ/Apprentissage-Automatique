from typing import Tuple
from sklearn.preprocessing import minmax_scale
import numpy as np
import pandas as pd
import torch

# Pour pytorch
class SwissmetroDataSet(torch.utils.data.Dataset):
    def __init__(self, path: str):
        df = pd.read_csv(path, sep='\t')
        df.drop(df[((df['PURPOSE'] != 1) & (df['PURPOSE'] != 3)) | (df['CHOICE'] == 0)].index, inplace=True) 
        df['SM_CO'] *= (df['GA'] == 0)
        df['TRAIN_CO'] *= (df['GA'] == 0)
        self.X = minmax_scale(df.drop('CHOICE', axis=1).values.astype(np.float32))
        # Décalage de 1 pour avoir les classes 0, 1 et 2
        self.y = (df['CHOICE'] - 1).values
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, i) -> Tuple[np.array, np.array]:
        return self.X[i], self.y[i]
