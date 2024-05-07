import os 
import glob
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch

class TimeSeriesDataset(Dataset):
    def __init__(self, folder, src_col):
        folder = folder
        csvs = glob.glob(folder + "/*.csv")
        X_train = []

        for csv in csvs:
            df = pd.read_csv(csv, index_col=0, parse_dates=True).sort_index()
            X_train.append(df[src_col].values)

        X_train = torch.from_numpy(np.array(X_train))

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape[1], X_train.shape[2])) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        self.x_data = X_train
        self.num_channels = X_train.shape[1]
        self.len = X_train.shape[0]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.x_data[idx].float()

