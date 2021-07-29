import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class NCFDataset(Dataset):

    def __init__(self, df: pd.DataFrame, user_col, item_col, label_col):
        self.df = df
        self.users = np.array(self.df[user_col], dtype=np.int64)
        self.items = np.array(self.df[item_col], dtype=np.int64)
        self.labels = np.array(self.df[label_col], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]