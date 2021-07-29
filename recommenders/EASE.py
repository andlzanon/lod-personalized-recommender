import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from recommenders.base_recommender import BaseRecommender


class EASE(BaseRecommender):
    def __init__(self, folder_path: str, output_filename: str, rank_size: int, cols_used: list, col_names: list,
                 lambda_: float):
        """

        :param folder_path:
        :param output_filename:
        :param rank_size:
        :param cols_used:
        :param col_names:
        :param lambda_:
        """
        super().__init__(folder_path, output_filename, rank_size, cols_used, col_names)
        self.lambda_ = lambda_

    def train(self):
        """

        :return:
        """
        df = self.train_set.reset_index()
        user_item = df.pivot(index=self.col_names[0], columns=self.col_names[1], values=self.col_names[2])
        user_item = user_item.fillna(0)

        x = csr_matrix(user_item.values)
        g = x.T.dot(x).toarray()
        diag_indices = np.diag_indices(g.shape[0])
        g[diag_indices] += self.lambda_
        p = np.linalg.inv(g)
        b = p / (-np.diag(p))
        b[diag_indices] = 0

        self.pred = x.dot(b)
        self.pred = pd.DataFrame(self.pred)
        self.pred.index = user_item.index
        self.pred.columns = user_item.columns

    def predict(self, user: int):
        """

        :param user:
        :return:
        """

        historic = self.train_set.loc[user][self.col_names[1]]
        if type(historic) != np.int64:
            historic = historic.tolist()
        else:
            historic = [historic]

        predictions = self.pred.loc[user]
        predictions = predictions.sort_values(ascending=False)

        top_n = []
        n = 0
        for item, score in predictions.iteritems():
            if item not in historic:
                top_n.append((user, item, score))
                n = n + 1
                if n == self.rank_size:
                    break
        return top_n

    def run(self):
        print(self.output_filename)
        cols = ['user', 'item', 'score']
        results = pd.DataFrame(columns=cols)
        users = self.test_set.index.unique()

        for u in users:
            ranked_items = self.predict(u)
            results = pd.concat([results, pd.DataFrame(ranked_items, columns=cols)], ignore_index=True)
            print("User: " + str(u))

        results.to_csv(self.output_path, mode='w', header=False, index=False)
