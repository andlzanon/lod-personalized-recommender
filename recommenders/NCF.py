import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch import nn
from preprocessing.NCFDataset import NCFDataset
from recommenders.base_recommender import BaseRecommender
from recommenders.NCFModule import NCFRecommender


class NCF(BaseRecommender):

    def __init__(self, folder_path: str, output_filename: str, rank_size: int, factors: int, layers: list,
                 epochs: int, neg_smp_train: int, neg_smp_test: int, p=0.4, batch_size=256, cols_used=None,
                 col_names=None, model_path=None, seed=42, model_disk=None):
        """
        Neural Collaborative Filtering Recommender constructor. This algorithm is proposed on the paper https://doi.org/10.1145/3038912.3052569
        :param folder_path: folder of the test and train files
        :param output_filename: name of the output file
        :param rank_size: number of recommended items to a user in the test set
        :param factors: number of factors
        :param layers: list wit number of neurons per layer
        :param epochs: number of epochs
        :param neg_smp_train: number of negative samples per positive sample
        :param neg_smp_test: number of negative samples on test set per user.
        :param p: dropout probability value
        :param batch_size: batch size for training
        :param cols_used: columns that the recommender algorithm will use from the original dataset
        :param col_names: name of the columns of test and train set
        :param model_path: path to trained model
        :param model_disk: 'r' to read model from model_path and 'w' to write model
        """

        if cols_used is None:
            cols_used = [0, 1, 2]

        super().__init__(folder_path, output_filename, rank_size, cols_used, col_names)

        self.factors = factors
        self.epochs = epochs
        self.layers = layers
        self.p = p
        self.batch_size = batch_size
        self.model_path = model_path
        self.model_disk = model_disk
        self.seed = seed
        self.neg_smp_train = neg_smp_train
        self.neg_smp_test = neg_smp_test
        random.seed(self.seed)

        print("Params: epochs: " + str(self.epochs) + ", batch size: " + str(self.batch_size))

        self.all_items = list(set(list(self.test_set[self.col_names[1]].unique()) +
                                  list(self.train_set[self.col_names[1]].unique())))
        self.all_users = list(set(list(self.test_set.index.unique()) + list(self.train_set.index.unique())))

        self.model = NCFRecommender(max(self.all_users) + 1, max(self.all_items) + 1, factors=self.factors,
                                    layers=self.layers, p=self.p)

        # check to load model from disk
        if self.model_disk == 'r' and self.model_path is not None:
            self.model.load_state_dict(torch.load(model_path))

        # try to read train and test sets, if error, generates the files
        try:
            self.train_neg = pd.read_csv(self.folder_path + "/train_neg.csv", header=None)
            self.train_neg.columns = self.col_names
            self.train_neg = self.train_neg.set_index(self.col_names[0])
            self.train_neg = self.train_neg.sample(frac=1, random_state=self.seed)

            self.test_neg = pd.read_csv(self.folder_path + "/test_neg.csv", header=None)
            self.test_neg.columns = self.col_names
            self.test_neg = self.test_neg.set_index(self.col_names[0])
            print("--- TRAIN FILE AND TEST FILE READ SUCCESSFULLY ---")
        except FileNotFoundError:
            self.train_neg, self.test_neg = self.leave_one_out()

    def leave_one_out(self):
        """
        Function that changes the the split of the folder from 0.9 to train and 0.1 to test to leave only the last
        interaction of the test to the test set along with n_neg_test random samples. The train set "receives" the other
        interactions of the test and for every positive interaction n_neg_train for every positive interaction is added
        :return: new train and test sets
        """
        train_neg = self.train_set.copy().reset_index()
        test_neg = pd.DataFrame(columns=self.col_names)

        # pass all the interactions for the test set excluding the last for each user, only if there is more thant one
        for user in self.test_set.index.unique():
            t_user = self.test_set.loc[user]
            if type(t_user) != pd.Series:
                train_neg = pd.concat([train_neg, t_user.reset_index().iloc[:-1]], ignore_index=True)
                test_neg = test_neg.append(pd.Series(t_user.reset_index().iloc[-1].to_dict()), ignore_index=True)
            else:
                test_neg = test_neg.append(t_user.append(pd.Series({self.col_names[0]: user})), ignore_index=True)

        # add negative samples to test and train
        test_neg = self.add_negative_smp_test(train_neg.set_index(self.col_names[0]),
                                              test_neg.set_index(self.col_names[0]))
        train_neg = self.add_negative_smp_train(train_neg.set_index(self.col_names[0]),
                                                test_neg)

        return train_neg, test_neg

    def add_negative_smp_train(self, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        """
        Function that adds n_neg_train negative samples per positive interaction on the train set
        without adding negative samples from test positive interactions
        :param train: train dataframe with interactions
        :param test: test dataframe with only one interaction for every user
        :returns train set shuffled and with negative samples
        """

        print("----- ADDING NEGATIVE SAMPLES TO TRAIN -----")
        if self.neg_smp_train == 0:
            return train.sample(frac=1, random_state=self.seed)

        output = train.copy()

        item_name = self.col_names[1]
        interaction_name = self.col_names[2]
        for u in train.index.unique():
            print(u)

            seen = train.loc[u][item_name]
            if not isinstance(seen, (int, np.integer)):
                seen = seen.tolist()
            else:
                seen = [seen]

            try:
                valid = test.loc[u][item_name]
                if not isinstance(valid, (int, np.integer)):
                    valid = valid.tolist()
                else:
                    valid = [valid]
            except KeyError:
                valid = []

            neg_smp_usr = self.neg_smp_train * len(seen)
            seen = seen + valid
            n = 0
            neg_sample = random.choice(self.all_items)
            while n < neg_smp_usr:
                if neg_sample not in seen:
                    output = output.append(pd.Series({item_name: neg_sample, interaction_name: 0}, name=u))
                    n = n + 1
                neg_sample = random.choice(self.all_items)

        output = output.reset_index().sort_values(by=[self.col_names[0]])
        output.to_csv(self.folder_path + "/train_neg.csv", mode='w', header=False, index=False)
        output = output.set_index(self.col_names[0])
        output = output.sample(frac=1, random_state=self.seed)
        return output

    def add_negative_smp_test(self, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        """
        Function that adds n_neg_test negative samples per user on the test set
        without adding negative samples from train positive interactions
        :param train: train dataframe with interactions
        :param test: test dataframe with only one interaction for every user
        :returns test set with negative samples
        """

        print("----- ADDING NEGATIVE SAMPLES TO TEST -----")
        if self.neg_smp_test == 0:
            return test

        output = test.copy()
        item_name = self.col_names[1]
        interaction_name = self.col_names[2]
        for u in test.index.unique():
            print(u)

            seen = test.loc[u][item_name]
            if not isinstance(seen, (int, np.integer)):
                seen = seen.tolist()
            else:
                seen = [seen]

            try:
                valid = train.loc[u][item_name]
                if not isinstance(valid, (int, np.integer)):
                    valid = valid.tolist()
                else:
                    valid = [valid]
            except KeyError:
                valid = []

            neg_smp_usr = self.neg_smp_test * len(seen)
            seen = seen + valid
            n = 0
            neg_sample = random.choice(self.all_items)
            while n < neg_smp_usr:
                if neg_sample not in seen:
                    output = output.append(pd.Series({item_name: neg_sample, interaction_name: 0}, name=u))
                    n = n + 1
                neg_sample = random.choice(self.all_items)

        output = output.reset_index().sort_values(by=[self.col_names[0]])
        output.to_csv(self.folder_path + "/test_neg.csv", mode='w', header=False, index=False)
        output = output.set_index(self.col_names[0])
        return output

    def train(self):
        """
        Function that trains the NCF model
        :return: model trained
        """

        if self.model_disk == 'r' and self.model_path is not None:
            print("Model loaded. Already trained")
            return

        neg_train = self.train_neg.reset_index()
        dataset = NCFDataset(neg_train, self.col_names[0], self.col_names[1], self.col_names[2])
        train_loader = data.DataLoader(dataset, batch_size=self.batch_size)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        print("----- START TRAINING MODEL -----")
        for i in range(self.epochs):
            for user, item, label in train_loader:
                y_pred = self.model(user, item)
                loss = criterion(y_pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Epoch: " + str(i + 1) + " Loss: " + str(loss.item()))

        if self.model_disk == 'w' and self.model_path is not None:
            print("----- SAVING MODEL -----")
            torch.save(self.model.state_dict(), self.model_path)

        print("----- END TRAINING MODEL -----")

    def predict(self, user: int):
        """
        Function that makes the predictions for the user based on the positive and negative interactions on the
        test set. The objective is that the only positive interaction on the test appears on the top N ranked items
        :param user: user to make predictions to
        :return: list of tuples with user, item and score to add to ranked dataframe with all users predictions
        """
        dic = {}
        historic = self.train_neg.loc[user][self.col_names[1]]

        if not isinstance(historic, (int, np.integer)):
            historic = historic.tolist()
        else:
            historic = [historic]

        items_eval = self.test_neg.loc[user][self.col_names[1]]
        if not isinstance(items_eval, (int, np.integer)):
            items_eval = items_eval.tolist()
        else:
            items_eval = [items_eval]

        for item in items_eval:
            if item not in historic:
                with torch.no_grad():
                    y_val = self.model(torch.LongTensor([user]), torch.LongTensor([item]))
                    dic[item] = y_val.item()

        dic_sort = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))
        top_n = []
        i = 0
        while i < self.rank_size:
            key = list(dic_sort.keys())[i]
            top_n.append((user, int(key), dic_sort[key]))
            i = i + 1

        return top_n

    def run(self):
        """
        Run the model and make the predictions for all users
        :return: file with ranked items to all users
        """
        print(self.output_filename)
        cols = ['user', 'item', 'score']
        results = pd.DataFrame(columns=cols)
        users = self.test_set.index.unique()
        self.model.eval()

        for u in users:
            ranked_items = self.predict(u)
            results = pd.concat([results, pd.DataFrame(ranked_items, columns=cols)], ignore_index=True)
            print("User: " + str(u))

        results.to_csv(self.output_path, mode='w', header=False, index=False)
