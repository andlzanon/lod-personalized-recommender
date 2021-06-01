import numpy as np
import pandas as pd
import networkx as nx


class LODPersonalizedReordering(object):

    def __init__(self, train_file: str, output_rec_file: str, prop_path: str, prop_cols: list, hybrid=False, n_sentences=1):
        """
        LOD personalized reordering class does not extend the base recommender because it does not recommend items but
        reorders them based on the best explanations to the item
        :param train_file: train file in which the recommendations of where computed
        :param output_rec_file: output file of the recommendation algorithm
        :param prop_path: path to the properties on dbpedia or wikidata
        :param prop_cols: columns of the property set
        :param hybrid: if the reorder of the recommendations should [True] or not consider the score from the recommender
        :param n_sentences: number of paths to generate the sentence of explanation
        """
        self.train_file = train_file
        self.train_set = pd.read_csv(self.train_file, header=None)
        self.train_set.columns = ['user_id', 'movie_id', 'interaction']
        self.train_set = self.train_set.set_index('user_id')

        self.output_rec_file = output_rec_file
        self.output_rec_set = pd.read_csv(self.output_rec_file, header=None)
        self.output_rec_set.columns = ['user_id', 'movie_id', 'score']
        self.output_rec_set = self.output_rec_set.set_index('user_id')

        self.prop_path = prop_path
        self.prop_cols = prop_cols
        self.prop_set = pd.read_csv(self.prop_path, usecols=self.prop_cols)
        self.prop_set = self.prop_set.set_index(self.prop_cols[0])

        s = output_rec_file.split(".")
        self.output_path = str(s[0]) + str(s[1]) + "_lodreorder.csv"
        self.hybrid = hybrid
        self.n_sentences = n_sentences
        self.graph = self.__build_graph()

    def __build_graph(self) -> nx.Graph:
        """
        Build a graph with the information from the test set and the wikidata or dbpedia set to create
        a graph with users, movies and property nodes e.g.: user 1 watched the movie Inception with Di Caprio as an actor
        therefore, on the graph there is a three nodes, one for each entity (user, movie and actor/property) and three
        edges, one connecting user to movie and other movie to property: user 1 --> Inception --> Di Caprio
        :return: networkx graph with users, movies and properties from the dbpedia or wikidata
        """

        user_item_set = self.train_set.copy()
        edgelist = pd.DataFrame(columns=['origin', 'destination'])

        user_item_set['origin'] = ['U' + x for x in user_item_set.index.astype(str)]
        user_item_set['destination'] = ['M' + x for x in user_item_set[user_item_set.columns[0]].astype(str)]

        edgelist = pd.concat([edgelist, user_item_set[['origin', 'destination']]], ignore_index=True)

        item_prop_copy = self.prop_set.copy()
        item_prop_copy['origin'] = ['M' + x for x in item_prop_copy.index.astype(str)]
        item_prop_copy['destination'] = item_prop_copy['obj']

        edgelist = pd.concat([edgelist, item_prop_copy[['origin', 'destination']]], ignore_index=True)

        G = nx.from_pandas_edgelist(edgelist, 'origin', 'destination')

        return G

    def reorder(self):
        """
        Function that reorders the recommendations made by the recommendation algorithm based on an adapted TF-IDF to
        the LOD, where the words of a document are the values of properties of the movies the user watched and all the
        documents are all movies properties
        :return: file with recommendations for every user reordered
        """
        reorder = pd.DataFrame(columns=['user_id', 'movie_id', 'score'])

        for u in self.output_rec_set.index.unique():
            # get movies that the user watched and recommended by an algorithm
            movies_watched = list(self.train_set.loc[u]['movie_id'])
            movies_recommended = list(self.output_rec_set.loc[u]['movie_id'])

            # reorder rec
            user_semantic_profile = self.__user_semantic_profile(movies_watched)
            sem_dist = self.__semantic_path_distance(movies_watched, movies_recommended, user_semantic_profile)
            sem_dist['score'] = pd.DataFrame(sem_dist['path'].to_list()).sum(1)
            sem_dist_matrix = sem_dist.pivot(index='historic', columns='recommended', values='score')
            reordered_movies = pd.DataFrame(sem_dist_matrix.sum().sort_values(ascending=False))
            reordered_movies = reordered_movies.reset_index()
            reordered_movies['user_id'] = u
            reordered_movies.columns = ['movie_id', 'score', 'user_id']
            reorder = pd.concat([reorder, reordered_movies], ignore_index=True)

            print("end")

        reorder.to_csv(self.output_path, mode='w', header=False, index=False)

    def __user_semantic_profile(self, watched: list) -> dict:
        """
        Generate the user semantic profile, where all the values of properties (e.g.: George Lucas, action films, etc)
        are ordered by a score that is calculated as:
        score = (npi/i) * log(N/dft), where npi are the number of edges to a value, i the number of watched movies,
        N the total number of movies and dft the number of movies with the value
        :param watched: list of the movies watched by a user
        :return: dictionary with properties' values as keys and scores as values
        """

        # create npi, i and n columns
        watched_props = self.prop_set.loc[self.prop_set.index.isin(watched)].copy()
        watched_props['npi'] = watched_props.groupby('obj')['obj'].transform('count')
        watched_props['i'] = len(watched)
        watched_props['n'] = len(self.prop_set.index.unique())

        # get items per property on full dbpedia/wikidata by dropping the duplicates with same item id and prop value
        # therefore, a value that repeats in the same item is ignored
        movies_per_obj = self.prop_set.reset_index().drop_duplicates(subset=[self.prop_set.columns[0], 'obj']).set_index('obj')
        df_dict = movies_per_obj.index.value_counts()

        # generate the dft column based on movies per property and score column base on all new created columns
        watched_props['dft'] = watched_props.apply(lambda x: df_dict[x['obj']], axis=1)

        watched_props['score'] = (watched_props['npi'] / watched_props['i']) * (np.log(watched_props['n'] / watched_props['dft']))

        # generate the dict
        watched_props.reset_index(inplace=True)
        watched_props = watched_props.set_index('obj')
        fav_prop = watched_props['score'].to_dict()

        return fav_prop

    def __semantic_path_distance(self, historic: list, recommeded: list, semantic_profile: dict) -> pd.DataFrame:
        sem_path_dist = pd.DataFrame(columns=['historic', 'recommended', 'path'])
        historic_codes = ['M' + str(m) for m in historic]
        recommeded_codes = ['M' + str(m) for m in recommeded]
        watched_props = list(set(self.prop_set.loc[self.prop_set.index.isin(historic)]['obj']))
        subgraph = self.graph.subgraph(historic_codes + recommeded_codes + watched_props)

        for hm in historic:
            hm_node = 'M' + str(hm)
            for rm in recommeded:
                rm_name = 'M' + str(rm)
                paths = nx.all_shortest_paths(subgraph, source=hm_node, target=rm_name)
                paths = [list(map(semantic_profile.get, p[1::2])) for p in paths]
                values = [sum(values) / len(values) for values in paths]
                sem_path_dist = sem_path_dist.append({'historic': hm, 'recommended': rm, 'path': paths[np.argmax(values)]},
                                                ignore_index=True)
                print("Historic: " + str(hm) + " Recommended: " + str(rm))

        return sem_path_dist