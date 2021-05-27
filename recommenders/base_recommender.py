import pandas as pd
from caserec.evaluation.item_recommendation import ItemRecommendationEvaluation

class BaseRecommender(object):
    def __init__(self, folder_path: str, output_filename: str, rank_size: int, col_names: list):
        """
         Recommender algorithm base class.
         :param folder_path: folder of the test and train files
         :param output_filename: name of the output file
         :param rank_size: number of recommended items to a user in the test set
         :param col_names: name of the columns of test and train set where index 0 is user_id, 1 is movie_id and
          2 is interaction
         """
        self.folder_path = folder_path
        self.test_path = self.folder_path + "/test.dat"
        self.train_path = self.folder_path + "/train.dat"
        self.output_filename = output_filename
        self.output_path = self.folder_path + "/outputs/" + output_filename
        self.col_names = col_names
        self.rank_size = rank_size

        # create pandas test set and train set and set their index by the user id column
        self.test_set = pd.read_csv(self.test_path, header=None)
        self.test_set.columns = self.col_names
        self.test_set = self.test_set.set_index(self.test_set.columns[0])

        self.train_set = pd.read_csv(self.train_path, header=None)
        self.train_set.columns = self.col_names
        self.train_set = self.train_set.set_index(self.train_set.columns[0])

    def train(self):
        pass

    def predict(self, user: int):
        pass

    def run(self):
        pass