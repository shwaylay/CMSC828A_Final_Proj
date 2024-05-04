import os 
import glob
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, folder, src_col, target_col=None):
        self.folder = folder
        self.csvs = glob.glob(folder + "/*.csv")
        self.src_col = src_col
        self.target_col = target_col

    def __len__(self):
        return len(self.csvs)
    
    def __getitem__(self, idx):
        df = pd.read_csv(self.csvs[idx], index_col=0, parse_dates=True).sort_index()
        if self.target_col == None:
            return df[self.src_col]
        else:
            return df[self.src_col], df[self.target_col]

