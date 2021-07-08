import pandas as pd
import networkx as nx
import re
from recommenders.base_recommender import BaseRecommender


class PageRankRecommnder(BaseRecommender):

    def __init__(self, folder_path: str, output_filename: str, rank_size: int, prop_set_path: str, prop_cols: list,
                 cols_used=None, col_names=None, node_weighs=None):
        """
        Page Rank Recommender constructor
        :param folder_path: folder of the test and train files
        :param output_filename: name of the output file
        :param rank_size: number of recommended items to a user in the test set
        :param cols_used: columns that the recommender algorithm will use from the original dataset
        :param col_names: name of the columns of test and train set
        :param prop_set_path: item property dbpedia or wikidata set, on the file the first row is the item id and the
            property column is the last column. Therefore, the dataset must in the format item_id, prop, value
        :param prop_cols: columns used in the property set
        :param node_weighs: list of weights on the personalization for the Personalized Page Rank, the order is item
            weight, properties related to item weight and then all the other node weights
        """

        if cols_used is None:
            cols_used = [0, 1, 2]
        if node_weighs is None:
            node_weighs = [0.8, 0, 0.2]

        super().__init__(folder_path, output_filename, rank_size, cols_used, col_names)

        self.prop_set_path = prop_set_path
        self.prop_cols = prop_cols
        self.prop_set = pd.read_csv(self.prop_set_path, usecols=self.prop_cols)
        self.prop_set = self.prop_set.set_index(self.prop_set.columns[0])
        self.node_weights = node_weighs

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
        item_prop_copy['destination'] = item_prop_copy[self.prop_set.columns[-1]]

        edgelist = pd.concat([edgelist, item_prop_copy[['origin', 'destination']]], ignore_index=True)

        G = nx.from_pandas_edgelist(edgelist, 'origin', 'destination')

        return G

    def predict(self, user: int):
        """
        Predict top items for user
        :param user: user id of the user to recommend to
        :return: top n items of the recommendation algorithm
        """

        # get historic items and transform it into the node name
        try:
            historic = self.train_set.loc[user][self.col_names[1]].to_list()
        except AttributeError:
            historic = list(self.train_set.loc[user])[:-1]

        items_codes = ['I' + str(x) for x in historic]
        n_items = len(items_codes)
        n_all = self.graph.number_of_nodes()

        # generate personalization dict to pass as parameter on page rank by dividing the 1 total into
        # the items the user iteracted, properties related to theses items and all the other nodes
        personalization = {}

        for node in self.graph.nodes:
            if node in items_codes:
                personalization[node] = self.node_weights[0] / n_items
            else:
                personalization[node] = self.node_weights[2] / (n_all - n_items)

        if self.node_weights[1] > 0:
            props = []
            for i in historic:
                try:
                    props = props + self.prop_set.loc[i][self.prop_set.columns[-1]].to_list()
                except AttributeError:
                    props = props + [self.prop_set.loc[i][self.prop_set.columns[-1]]]
                except KeyError:
                    continue

            props = set(props)
            n_props = len(props)
            for p in props:
                personalization[p] = self.node_weights[1] / n_props

        # run page rank
        page_rank = nx.pagerank_scipy(self.graph, personalization=personalization, max_iter=1000)

        # order by page rank score
        ordered_nodes = dict(sorted(page_rank.items(), key=lambda item: item[1], reverse=True))
        top_n = []
        n = 0
        i = 0
        while n < self.rank_size:
            key = list(ordered_nodes.keys())[i]
            if re.match("I(\d+)+$", str(key)) and key not in items_codes:
                top_n.append((user, int(key[1:]), page_rank[key]))
                n = n + 1
            i = i + 1

        return top_n

    def run(self):
        """
        Predict items for all users on test set and save on file
        :return: file with columns user, item and score
        """
        print(self.output_filename)
        cols = ['user', 'item', 'score']
        results = pd.DataFrame(columns=cols)
        users = self.test_set.index.unique()

        for u in users:
            ranked_items = self.predict(u)
            results = pd.concat([results, pd.DataFrame(ranked_items, columns=cols)], ignore_index=True)
            print("User: " + str(u))

        results.to_csv(self.output_path, mode='w', header=False, index=False)