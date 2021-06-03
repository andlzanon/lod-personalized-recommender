import random
import numpy as np
import pandas as pd
import networkx as nx
from recommenders.lod_reordering import LODPersonalizedReordering


class PathReordering(LODPersonalizedReordering):
    def __init__(self, train_file: str, output_rec_file: str, prop_path: str, prop_cols: list, cols_used: list,
                 policy: str, p_items: float, hybrid=False, n_sentences=3):
        """
        Path Reordering class: this algorithm will reorder the output of other recommendation algorithm based on the
        best path from an historic item and a recommended one. The best paths are extracted based on the value for each
        object of the LOD with the semantic profile
        :param train_file: train file in which the recommendations of where computed
        :param output_rec_file: output file of the recommendation algorithm
        :param prop_path: path to the properties on dbpedia or wikidata
        :param prop_cols: columns of the property set
        :param cols_used: columns used from the test and train set
        :param policy: the policy to get the historic items to get the best paths. Possible values: 'all' for all items
                       'last' for the last interacted items, 'first' for the first interacted items and 'random' for
                       the random interacted items
        :param p_items: percentage from 0 to 1 of items to consider in the policy. E.g. policy last and p_items = 0.1,
                        then the paths will consider only the last 0.1 * len(items inteacted)
        :param hybrid: if the reorder of the recommendations should [True] or not consider the score from the recommender
        :param n_sentences: number of paths to generate the sentence of explanation
        """

        self.policy = policy
        self.p_items = p_items
        self.output_name = 'path_' + str(policy) + str(p_items).replace('.', '')

        if hybrid:
            self.output_name = self.output_name + "_hybrid"

        super().__init__(train_file, output_rec_file, self.output_name, prop_path, prop_cols, cols_used,
                         hybrid, n_sentences)

    def reorder(self):
        """
        Function that reorders the recommendations made by the recommendation algorithm based on an adapted TF-IDF to
        the LOD, where the words of a document are the values of properties of the items the user iteracted and all the
        documents are all items properties
        :return: file with recommendations for every user reordered
        """
        reorder = pd.DataFrame({'user_id': pd.Series([], dtype=int),
                                'item_id': pd.Series([], dtype=int),
                                'score': pd.Series([], dtype=float)})

        for u in self.output_rec_set.index.unique():
            print("User: " + str(u))
            # get items that the user interacted and recommended by an algorithm
            items_historic = self.train_set.loc[u].sort_values(by=self.cols_used[-1], ascending=False)
            try:
                items_historic = items_historic[self.cols_used[1]].to_list()
            except AttributeError:
                items_historic = list(self.train_set.loc[u][self.cols_used[1]])[:-1]

            items_recommended = list(self.output_rec_set.loc[u][self.output_cols[1]])

            # get semantic profile and extract the best paths from the suggested item to the recommended
            user_semantic_profile = self.user_semantic_profile(items_historic)
            items_historic_cutout = self.__items_by_policy(items_historic)

            # new items interacted based on policy and percentage
            sem_dist = self.__semantic_path_distance(items_historic_cutout, items_recommended, user_semantic_profile)

            # create column with the sum of paths, pivot to create a matrix with interacted items by recommended
            # and reorder the recommended items by the sum of the columns
            sem_dist['score'] = pd.DataFrame(sem_dist['path'].to_list()).mean(1)
            sem_dist_matrix = sem_dist.pivot(index='historic', columns='recommended', values='score')
            reordered_items = pd.DataFrame(sem_dist_matrix.mean().sort_values(ascending=False))
            reordered_items = reordered_items.reset_index()
            reordered_items['user_id'] = u
            reordered_items.columns = ['item_id', 'score', 'user_id']

            if self.hybrid:
                output_rec = self.output_rec_set.loc[u].set_index('item_id')
                for i in items_recommended[:10]:
                    curr_score = float(reordered_items.loc[(reordered_items['item_id']) == i, 'score'])
                    rec_score = float(output_rec.loc[i])
                    reordered_items.loc[(reordered_items['item_id']) == i, 'score'] = curr_score * rec_score

                reordered_items = reordered_items.fillna(0)
                reordered_items = reordered_items.sort_values(by='score', ascending=False)

            reorder = pd.concat([reorder, reordered_items], ignore_index=True)

        reorder.to_csv(self.output_path, mode='w', header=False, index=False)

    def __semantic_path_distance(self, historic: list, recommeded: list, semantic_profile: dict) -> pd.DataFrame:
        """
        Get the best path based on the semantic profile from all the historic items to the recommended ones
        :param historic: list of items that the user interacted
        :param recommeded: recommended items by the user
        :param semantic_profile: semantic profile of the user
        :return: data frame with historic item, the recommended and path
        """
        sem_path_dist = pd.DataFrame(columns=['historic', 'recommended', 'path'])
        historic_codes = ['I' + str(i) for i in historic]
        recommeded = recommeded[:10]
        recommeded_codes = ['I' + str(i) for i in recommeded]
        historic_props = list(set(self.prop_set.loc[self.prop_set.index.isin(historic)]['obj']))
        subgraph = self.graph.subgraph(historic_codes + recommeded_codes + historic_props)

        for hm in historic:
            hm_node = 'I' + str(hm)
            for rm in recommeded:
                rm_name = 'I' + str(rm)
                try:
                    paths = nx.all_shortest_paths(subgraph, source=hm_node, target=rm_name)
                    paths = [list(map(semantic_profile.get, p[1::2])) for p in paths]
                    values = [sum(values) / len(values) for values in paths]
                    sem_path_dist = sem_path_dist.append(
                        {'historic': hm, 'recommended': rm, 'path': paths[np.argmax(values)]},
                        ignore_index=True)
                except nx.exception.NetworkXNoPath:
                    sem_path_dist = sem_path_dist.append({'historic': hm, 'recommended': rm, 'path': []},
                                                         ignore_index=True)
                print("Historic: " + str(hm) + " Recommended: " + str(rm))

        return sem_path_dist

    def __items_by_policy(self, historic: list):
        """
        Function that returns items the p_items percentage of the total number of historic items based
        :param historic: items in the historic ordered by the last column of the dataset
        :return: cutout list of the historic based on the policy and the percentage of items to consider
        """
        n = int(self.p_items * len(historic))
        if n < 10:
            n = 10

        if self.policy == 'all':
            return historic
        if self.policy == 'last':
            return historic[:n]
        if self.policy == 'first':
            return historic[(-1 * n):]
        if self.policy == 'random':
            return random.sample(historic, n)
