import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class NCFDataset(Dataset):

    def __init__(self, df: pd.DataFrame, user_col, item_col, label_col):
        """
        NCF Dataset class that helps pytorch handle epochs and batch size when training the model. Therefore,
        a dataframe is "casted" to NCFDataset
        :param df:
        :param user_col:
        :param item_col:
        :param label_col:
        """
        self.df = df
        self.users = np.array(self.df[user_col], dtype=np.int64)
        self.items = np.array(self.df[item_col], dtype=np.int64)
        self.labels = np.array(self.df[label_col], dtype=np.float32)

    def __len__(self):
        """
        Function that returns the len of the NCFDataset
        :return: len of NCFDataset
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Function that sets how pytorch loads items from the dataframe
        :param idx: integer or list of ids from which the data from the set will be loaded
        :return: user, item and label from ids
        """
        return self.users[idx], self.items[idx], self.labels[idx]