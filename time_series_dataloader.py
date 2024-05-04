import os 
import glob
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, folder, col):
        self.folder = folder
        self.csvs = glob.glob(folder + "/*.csv")
        self.col = col

    def __len__(self):
        return len(self.csvs)
    
    def __getitem__(self, idx):
        df = pd.read_csv(self.csvs[idx], index_col=0, parse_dates=True).sort_index()
        return df[self.col]

