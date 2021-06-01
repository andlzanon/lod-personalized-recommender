import random
import numpy as np
import pandas as pd
import networkx as nx


class LODPersonalizedReordering(object):

    def __init__(self, train_file: str, output_rec_file: str, prop_path: str, prop_cols: list, cols_used: list,
                 policy: str, p_items: float, hybrid=False, n_sentences=3):
        """
        LOD personalized reordering class does not extend the base recommender because it does not recommend items.
        Instead it reorders them based on the best explanations to the item. Therefore, initially a semantic profile
        (favorite properties) of the user is obtained based on a policy that will use 'all' items of the historic,
        'random' items 'last' items or 'first' n_moives percentage of total from the user.

        :param train_file: train file in which the recommendations of where computed
        :param output_rec_file: output file of the recommendation algorithm
        :param prop_path: path to the properties on dbpedia or wikidata
        :param prop_cols: columns of the property set
        :param policy: the policy to get the historic items to get the best paths. Possible values: 'all' for all items
        'last' for the last interacted items, 'first' for the first interacted items and 'random' for the random
        interacted items
        :param p_items: percentage from 0 to 1 of items to consider in the policy. E.g. policy last and p_items = 0.1,
        then the paths will consider only the last 0.1 * len(items inteacted)
        :param hybrid: if the reorder of the recommendations should [True] or not consider the score from the recommender
        :param n_sentences: number of paths to generate the sentence of explanation
        """
        self.cols_used = cols_used

        self.train_file = train_file
        self.train_set = pd.read_csv(self.train_file, header=None)
        self.train_set.columns = self.cols_used
        self.train_set = self.train_set.set_index(self.cols_used[0])

        self.output_cols = ['user_id', 'item_id', 'score']
        self.output_rec_file = output_rec_file
        self.output_rec_set = pd.read_csv(self.output_rec_file, header=None)
        self.output_rec_set.columns = self.output_cols
        self.output_rec_set = self.output_rec_set.set_index(self.cols_used[0])

        self.prop_path = prop_path
        self.prop_cols = prop_cols
        self.prop_set = pd.read_csv(self.prop_path, usecols=self.prop_cols)
        self.prop_set = self.prop_set.set_index(self.prop_cols[0])

        self.hybrid = hybrid
        self.n_sentences = n_sentences
        self.policy = policy
        self.p_items = p_items

        s = output_rec_file.split(".")
        self.output_path = "." + str(s[1]) + "_lodreorder_" + self.policy + str(self.p_items) + ".csv"

        self.graph = self.__build_graph()

    def __build_graph(self) -> nx.Graph:
        """
        Build a graph with the information from the test set and the wikidata or dbpedia set to create
        a graph with users, items and property nodes e.g.: user 1 interacted the item Inception with Di Caprio as an actor
        therefore, on the graph there is a three nodes, one for each entity (user, item and actor/property) and three
        edges, one connecting user to item and other item to property: user 1 --> Inception --> Di Caprio
        :return: networkx graph with users, items and properties from the dbpedia or wikidata
        """

        user_item_set = self.train_set.copy()
        edgelist = pd.DataFrame(columns=['origin', 'destination'])

        user_item_set['origin'] = ['U' + x for x in user_item_set.index.astype(str)]
        user_item_set['destination'] = ['I' + x for x in user_item_set[user_item_set.columns[0]].astype(str)]

        edgelist = pd.concat([edgelist, user_item_set[['origin', 'destination']]], ignore_index=True)

        item_prop_copy = self.prop_set.copy()
        item_prop_copy['origin'] = ['I' + x for x in item_prop_copy.index.astype(str)]
        item_prop_copy['destination'] = item_prop_copy['obj']

        edgelist = pd.concat([edgelist, item_prop_copy[['origin', 'destination']]], ignore_index=True)

        G = nx.from_pandas_edgelist(edgelist, 'origin', 'destination')

        return G

    def reorder(self):
        """
        Function that reorders the recommendations made by the recommendation algorithm based on an adapted TF-IDF to
        the LOD, where the words of a document are the values of properties of the items the user iteracted and all the
        documents are all items properties
        :return: file with recommendations for every user reordered
        """
        reorder = pd.DataFrame(columns=['user_id', 'item_id', 'score'])

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
            user_semantic_profile = self.__user_semantic_profile(items_historic)
            items_historic_cutout = self.__items_by_policy(items_historic)

            # new items interacted based on policy and percentage
            sem_dist = self.__semantic_path_distance(items_historic_cutout, items_recommended, user_semantic_profile)

            # create column with the sum of paths, pivot to create a matrix with interacted items by recommended
            # and reorder the recommended items by the sum of the columns
            sem_dist['score'] = pd.DataFrame(sem_dist['path'].to_list()).mean(1)
            sem_dist_matrix = sem_dist.pivot(index='historic', columns='recommended', values='score')
            reordered_items = pd.DataFrame(sem_dist_matrix.sum().sort_values(ascending=False))
            reordered_items = reordered_items.reset_index()
            reordered_items['user_id'] = u
            reordered_items.columns = ['item_id', 'score', 'user_id']
            reorder = pd.concat([reorder, reordered_items], ignore_index=True)

        reorder.to_csv(self.output_path, mode='w', header=False, index=False)

    def __user_semantic_profile(self, historic: list) -> dict:
        """
        Generate the user semantic profile, where all the values of properties (e.g.: George Lucas, action films, etc)
        are ordered by a score that is calculated as:
            score = (npi/i) * log(N/dft)
        where npi are the number of edges to a value, i the number of interacted items,
        N the total number of items and dft the number of items with the value
        :param historic: list of the items interacted by a user
        :return: dictionary with properties' values as keys and scores as values
        """

        # create npi, i and n columns
        interacted_props = self.prop_set.loc[self.prop_set.index.isin(historic)].copy()
        interacted_props['npi'] = interacted_props.groupby('obj')['obj'].transform('count')
        interacted_props['i'] = len(historic)
        interacted_props['n'] = len(self.prop_set.index.unique())

        # get items per property on full dbpedia/wikidata by dropping the duplicates with same item id and prop value
        # therefore, a value that repeats in the same item is ignored
        items_per_obj = self.prop_set.reset_index().drop_duplicates(subset=[self.prop_set.columns[0], 'obj']).set_index('obj')
        df_dict = items_per_obj.index.value_counts()

        # generate the dft column based on items per property and score column base on all new created columns
        interacted_props['dft'] = interacted_props.apply(lambda x: df_dict[x['obj']], axis=1)

        interacted_props['score'] = (interacted_props['npi'] / interacted_props['i']) * (np.log(interacted_props['n'] / interacted_props['dft']))

        # generate the dict
        interacted_props.reset_index(inplace=True)
        interacted_props = interacted_props.set_index('obj')
        fav_prop = interacted_props['score'].to_dict()

        return fav_prop

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
                    sem_path_dist = sem_path_dist.append({'historic': hm, 'recommended': rm, 'path': paths[np.argmax(values)]},
                                                ignore_index=True)
                except nx.exception.NetworkXNoPath:
                    sem_path_dist = sem_path_dist.append({'historic': hm, 'recommended': rm, 'path': []},
                                                ignore_index=True)
                # print("Historic: " + str(hm) + " Recommended: " + str(rm))

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