import _io
import random
import numpy as np
import pandas as pd
import networkx as nx

from recommenders.lod_reordering import LODPersonalizedReordering
from scipy.stats import entropy
import evaluation_utils as eval


class PathReordering(LODPersonalizedReordering):
    def __init__(self, train_file: str, output_rec_file: str, prop_path: str, prop_cols: list, cols_used: list,
                 n_reorder: int,
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
                        then the paths will consider only the last 0.1 * len(items inteacted). If p_items is bigger than 1
                        it will use the policy of the p_items of the user historic
        :param hybrid: if the reorder of the recommendations should [True] or not consider the score from the recommender
        :param n_sentences: number of paths to generate the sentence of explanation
        """

        self.policy = policy
        self.p_items = p_items
        self.output_name = 'path[policy=' + str(policy) + "_items=" + str(p_items).replace('.', '') + "_reorder=" + str(
            n_reorder) + "]"

        if self.policy == 'random':
            random.seed(42)

        if hybrid:
            self.output_name = self.output_name[:-1] + "_hybrid]"

        super().__init__(train_file, output_rec_file, self.output_name, prop_path, prop_cols, cols_used, n_reorder,
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

        print(self.output_name)

        for u in self.output_rec_set.index.unique():
            print("User: " + str(u))
            # get items that the user interacted and recommended by an algorithm
            items_historic = self.train_set.loc[u].sort_values(by=self.cols_used[-1], ascending=False)

            try:
                items_historic = items_historic[self.cols_used[1]].to_list()
            except AttributeError:
                items_historic = list(self.train_set.loc[u][self.cols_used[1]])[:-1]

            items_recommended = list(self.output_rec_set.loc[u][self.output_cols[1]])[:self.n_reorder]

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
                for i in items_recommended:
                    curr_score = float(reordered_items.loc[(reordered_items['item_id']) == i, 'score'])
                    rec_score = float(output_rec.loc[i])
                    reordered_items.loc[(reordered_items['item_id']) == i, 'score'] = curr_score * rec_score

                reordered_items = reordered_items.fillna(0)
                reordered_items = reordered_items.sort_values(by='score', ascending=False)

            reorder = pd.concat([reorder, reordered_items], ignore_index=True)

        reorder.to_csv(self.output_path, mode='w', header=False, index=False)

    def reorder_with_path(self, fold: str, h_min: int, h_max: int, max_users: int, expl_alg: str, reordered: int):
        """
        Function that reorders the recommendations made by a recommendation algorithm based on an adapted TF-IDF to
        the LOD and generate the explanation paths. There are two approaches: the max that always return the best path
        or the diverse that also considers the diversity of properties conecting recommended and suggested items. The
        final explanation approach is  the ExpLOD framework (https://dl.acm.org/doi/abs/10.1145/2959100.2959173)
        :param: h_min: minimum number of users' historic items to generate the recommendations and explanations to, if a
        user has a smaller number of interacted items than this parameter the algorithm will not generate explanations
        :param: h_max: maximum number of users' historic items to generate the recommendations and explanations to, if a
        user has a bigger number of interacted items than this parameter the algorithm will not generate explanations
        :param: max_users: maximum number of user to generate explanations to
        :return: file with recommendations for every user reordered
        """

        # create result and output file names
        results_file_name = fold + "/results/explanations/" + self.output_path.split("/")[-1]
        results_title_l = results_file_name.split("/")
        results_title = '/'.join(results_title_l[:-1])
        results_title = results_title + "/reordered_recs=" + str(reordered) + "_" + results_title_l[-1]

        output_file_name = fold + "/outputs/explanations/" + self.output_path.split("/")[-1]
        output_title_l = output_file_name.split("/")
        output_title = '/'.join(output_title_l[:-1])
        output_title = output_title + "/reordered_recs=" + str(reordered) + "_" + output_title_l[-1]
        f = open(output_title, mode="w", encoding='utf-8')
        f.write(output_title + "\n")

        print(output_title)
        n_users = 0
        m_items = []
        m_props = []
        total_items = {}
        total_props = {}
        for u in self.output_rec_set.index.unique():
            # get items that the user interacted and recommended by an algorithm
            items_historic = self.train_set.loc[u].sort_values(by=self.cols_used[-1], ascending=False)

            if h_min >= 0 and h_max > 0 and max_users > 0:
                if len(items_historic) <= h_min or len(items_historic) >= h_max:
                    continue
                if n_users == max_users:
                    break
                n_users = n_users + 1

            try:
                items_historic = items_historic[self.cols_used[1]].to_list()
            except AttributeError:
                items_historic = list(self.train_set.loc[u][self.cols_used[1]])[:-1]

            print("\nUser: " + str(u))
            f.write("\nUser: " + str(u) + "\n")
            print("Items interacted by the user")
            f.write("Items interacted by the user\n")
            for i in items_historic:
                try:
                    movie_name = self.prop_set.loc[i].iloc[0][0]
                except pd.IndexingError:
                    movie_name = self.prop_set.loc[i].iloc[0]
                print("Item id: " + str(i) + " Name: " + movie_name)
                f.write("Item id: " + str(i) + " Name: " + movie_name + "\n")
            items_recommended = list(self.output_rec_set.loc[u][self.output_cols[1]])[:self.n_reorder]

            print("\nTop " + str(self.n_reorder) + " recommended")
            for i in items_recommended:
                movie_name = self.prop_set.loc[i].iloc[0, 0]
                print("Item id: " + str(i) + " Name: " + movie_name)
                f.write("Item id: " + str(i) + " Name: " + movie_name + "\n")

            # get semantic profile and extract the best paths from the suggested item to the recommended
            user_semantic_profile = self.user_semantic_profile(items_historic)

            print("\nUsers favorites attributes on the kG")
            f.write("\nUsers favorites attributes on the kG\n")
            s_user_sem_pro = dict(sorted(user_semantic_profile.items(), key=lambda item: item[1], reverse=True))
            n = 0
            for k in s_user_sem_pro.keys():
                if n < 5:
                    print(k)
                    f.write(k + "\n")
                    n = n + 1
                else:
                    break

            items_historic_cutout = self.__items_by_policy(items_historic)
            # new items interacted based on policy and percentage
            sem_dist = self.__semantic_path_distance(items_historic_cutout, items_recommended, user_semantic_profile)

            # create column with the sum of paths
            sem_dist['score'] = pd.DataFrame(sem_dist['path'].to_list()).mean(1)
            if reordered:
                # pivot to create a matrix with interacted items by recommended
                # and reorder the recommended items by the sum of the columns
                sem_dist_matrix = sem_dist.pivot(index='historic', columns='recommended', values='score')
                reordered_items = pd.DataFrame(sem_dist_matrix.mean().sort_values(ascending=False))
                reordered_items = reordered_items.reset_index()
                reordered_items['user_id'] = u
                reordered_items.columns = ['item_id', 'score', 'user_id']

                if self.hybrid:
                    output_rec = self.output_rec_set.loc[u].set_index('item_id')
                    for i in items_recommended:
                        curr_score = float(reordered_items.loc[(reordered_items['item_id']) == i, 'score'])
                        rec_score = float(output_rec.loc[i])
                        reordered_items.loc[(reordered_items['item_id']) == i, 'score'] = curr_score * rec_score

                    reordered_items = reordered_items.fillna(0)
                    reordered_items = reordered_items.sort_values(by='score', ascending=False)

                print("\nReordered Recommendations")
                f.write("\nReordered Recommendations\n")
                item_rank = list(reordered_items['item_id'])
                for i in item_rank:
                    movie_name = self.prop_set.loc[i].iloc[0, 0]
                    print("Item id: " + str(i) + " Name: " + movie_name)
                    f.write("Item id: " + str(i) + " Name: " + movie_name + "\n")

            else:
                item_rank = pd.read_csv(self.output_path, header=None)
                item_rank.columns = ["user_id", "item", "score"]
                item_rank = item_rank.set_index("user_id")
                item_rank = list(item_rank.loc[u]["item"])

                print("\nRecommendations")
                f.write("\nRecommendations\n")
                for i in item_rank:
                    movie_name = self.prop_set.loc[i].iloc[0, 0]
                    print("Item id: " + str(i) + " Name: " + movie_name)
                    f.write("Item id: " + str(i) + " Name: " + movie_name + "\n")

            sem_dist = sem_dist.set_index('recommended')
            sem_dist = sem_dist.fillna(0)
            items, props = [], []
            if expl_alg == 'max':
                items, props = self.__best_ranked_paths(item_rank, sem_dist, user_semantic_profile, f)
            elif expl_alg == 'diverse':
                items, props = self.__diverse_ranked_paths(item_rank, sem_dist, user_semantic_profile, f)
            elif expl_alg == 'explod':
                items, props = self.__explod_ranked_paths(item_rank, items_historic, user_semantic_profile, f)
            f.write("\n")

            total_items = dict(list(total_items.items()) + list(items.items()))
            m_items.append(len(items))
            total_props = dict(list(total_props.items()) + list(props.items()))
            m_props.append(len(props))

        f.close()
        eval.evaluate_explanations(results_title, m_items, m_props, total_items, total_props, self.train_set,
                                   self.prop_set)

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
                    values = [sum(values) / len(values) for values in paths if len(values) > 0 or values is None]
                    sem_path_dist = sem_path_dist.append(
                        {'historic': hm, 'recommended': rm, 'path': paths[np.argmax(values)]},
                        ignore_index=True)
                except (nx.exception.NetworkXNoPath, ValueError):
                    sem_path_dist = sem_path_dist.append({'historic': hm, 'recommended': rm, 'path': []},
                                                         ignore_index=True)
                # print("Historic: " + str(hm) + " Recommended: " + str(rm))

        return sem_path_dist

    def __items_by_policy(self, historic: list):
        """
        Function that returns items the p_items percentage or quntity of the total number of historic items based
        :param historic: items in the historic ordered by the last column of the dataset
        :return: cutout list of the historic based on the policy and the percentage of items to consider
        """

        if self.p_items > 1:
            n = self.p_items
        else:
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
            try:
                items = random.sample(historic, n)
            except ValueError:
                items = random.sample(historic, len(historic))
            return items

    def __best_ranked_paths(self, ranked_items: list, semantic_distance: pd.DataFrame, semantic_profile: dict, file: _io.TextIOWrapper):
        """
        Explanation method that obtains the most relevant path based on the property value and show to user
        :param ranked_items: list of the recommended items
        :param semantic_distance: dataframe with paths from historic to recommended items
        :param semantic_profile: dictionary with property as key and score as value
        :param file: file to write explanations
        :return: historic items and properties used in explanations
        """
        hist_items = {}
        nodes = {}
        for i in ranked_items:
            print("\nPaths for the Recommended Item: " + str(i))
            file.write("\nPaths for the Recommended Item: " + str(i) + "\n")
            paths_rec = semantic_distance.loc[i].sort_values(by='score', ascending=False)
            k = 0
            # obtain best path
            hist_item = paths_rec.iloc[k, 0]
            node_value = paths_rec.iloc[k, 1]
            node_name = []
            for v in node_value:
                node_name.append(
                    list(semantic_profile.keys())[list(semantic_profile.values()).index(v)])

            # create sentence
            item = self.prop_set.loc[hist_item].iloc[0, 0]
            origin = "origin: \"" + item + "\""
            hist_items = self.__add_dict(hist_items, item)
            path_sentence = " nodes: "
            for n in node_name:
                path_sentence = path_sentence + "\"" + n + "\" "
                nodes = self.__add_dict(nodes, n)
            destination = "destination: \"" + self.prop_set.loc[i].iloc[0, 0] + "\""
            print(origin + path_sentence + destination)
            file.write(origin + path_sentence + destination)

            # check for others with same value
            k = k + 1
            value = paths_rec.iloc[k, 2]
            while paths_rec.iloc[k, 2] == value:
                hist_item = paths_rec.iloc[k, 0]
                node_value = paths_rec.iloc[k, 1]
                node_name = []
                for v in node_value:
                    node_name.append(
                        list(semantic_profile.keys())[list(semantic_profile.values()).index(v)])

                item = self.prop_set.loc[hist_item].iloc[0, 0]
                origin = "origin: \"" + item + "\""
                hist_items = self.__add_dict(hist_items, item)

                path_sentence = " nodes: "
                for n in node_name:
                    path_sentence = path_sentence + "\"" + n + "\""
                    nodes = self.__add_dict(nodes, n)
                destination = " destination: \"" + self.prop_set.loc[i].iloc[0, 0] + "\""
                print(origin + path_sentence + destination)
                file.write(origin + path_sentence + destination)
                k = k + 1

        return hist_items, nodes

    def __diverse_ranked_paths(self, ranked_items: list, semantic_distance: pd.DataFrame, semantic_profile: dict, file: _io.TextIOWrapper):
        """
        Generate explanations to recommendations considering the max value varying the properties shown to the user
        the logic to this explanation is to order all paths to recommended items and resolve the conflicts (when there
        is a repetition of properties) from biggest value to lowest value recursively until all items have different
        properties. If there is a conflict with the lowest prop value, then we repeat the highest values
        :param ranked_items: list of the recommended items
        :param semantic_distance: dataframe with paths from historic to recommended items
        :param semantic_profile: dictionary with property as key and score as value
        :param file: file to write explanations
        :return: historic items and properties used in explanations
        """
        hist_items = {}
        nodes = {}
        high_values = self.__diverse_ordered_properties(ranked_items, semantic_distance)

        # display the explanation path for every recommendation
        for i in high_values.index:
            node_value = high_values.loc[i][1]
            print("\nPaths for the Recommended Item: " + str(i))
            file.write("\nPaths for the Recommended Item: " + str(i) + "\n")
            node_name = []
            for v in node_value:
                node_name.append(
                    list(semantic_profile.keys())[list(semantic_profile.values()).index(v)])

            origin = ""
            hscore = high_values.loc[i][2]
            rec_rows = semantic_distance.loc[i]
            hscore_rows = rec_rows[rec_rows['score'] == hscore]
            for r in hscore_rows.iterrows():
                hist_item = r[1][0]
                item = self.prop_set.loc[hist_item].iloc[0, 0]
                origin = origin + "\"" + item + "\"; "
                hist_items = self.__add_dict(hist_items, item)
            origin = origin[:-2]

            path_sentence = " nodes: "
            for n in node_name:
                path_sentence = path_sentence + "\"" + n + "\" "
                nodes = self.__add_dict(nodes, n)
            destination = "destination: \"" + self.prop_set.loc[i].iloc[0, 0] + "\""
            print(origin + path_sentence + destination)
            file.write(origin + path_sentence + destination)

        return hist_items, nodes

    def __diverse_ordered_properties(self, ranked_items: list, semantic_distance: pd.DataFrame):
        """
        Order the explanation paths in order to maximize value, without or repeating only a few times properites
        :param ranked_items: list of recommended items
        :param semantic_distance: dataframe with paths from historic to recommended items
        :return: pandas df with the paths of best unrepeated items
        """
        # order all the paths based on the max value of the props on the path
        high_values = pd.DataFrame()
        for i in ranked_items:
            sem_dist = semantic_distance.loc[i].sort_values(by='score', kind="quicksort", ascending=False)
            row = sem_dist.iloc[0].copy()
            row["rec"] = i
            high_values = high_values.append(row.to_dict(), ignore_index=True)

        # create counter that represents the next index to check for multiple values and df with the max value
        high_values = high_values.set_index('rec')
        count = 0
        max_size = len(ranked_items)
        maximum = high_values['score'].sort_values(ascending=False).iloc[count]
        df_max = high_values[high_values['score'] == maximum]
        # while the count is not the last row, resolve conflicts
        while count < max_size - 1:
            # if there is only one row on max value there is not conflict, base case
            if len(list(df_max['score'])) == 1:
                order_values = high_values['score'].sort_values(ascending=False)
                count = list(order_values).index(maximum) + 1
                maximum = order_values.iloc[count]
                df_max = high_values[high_values['score'] == maximum]
                continue
            # create df of next high value for every item with the same current maximum value
            second_high = pd.DataFrame()
            for i in df_max.index:
                sem_dist = semantic_distance.loc[i].sort_values(by='score', kind="quicksort", ascending=False)
                scores = list(sem_dist['score'].unique())
                index = scores.index(maximum) + 1
                if index < len(scores):
                    next_value = scores[index]
                    row = sem_dist[sem_dist['score'] == next_value].iloc[0].copy()
                    row["rec"] = i
                    second_high = second_high.append(row.to_dict(), ignore_index=True)

            # substitute for every item except the last (otherwise the explanation value would not be the max possible)
            # the next biggest value with the current max that has conflicts
            try:
                second_high = second_high.sort_values(by="score", kind="quicksort", ascending=False)
                second_high = second_high.set_index('rec')
                sub_list = second_high.index[:-1]
                # if there is only one value to substitute, substitute this value
                if len(second_high) == 1:
                    sub_list = second_high.index

                for i in sub_list:
                    high_values.loc[i] = second_high.loc[i]

                # get next max to check for conflicts
                order_values = high_values['score'].sort_values(ascending=False)
                count = list(order_values).index(maximum) + 1
                maximum = order_values.iloc[count]
                df_max = high_values[high_values['score'] == maximum]
            # if there the conflicts was not resolved (lowest value is a tie) then recursively repeat the best
            # properties only for the items with tie
            except KeyError:
                if len(second_high) == 0:
                    break
                second_high = self.__diverse_ordered_properties(list(df_max.index), semantic_distance)
                for i in second_high.index:
                    high_values.loc[i] = second_high.loc[i]
                break

        return high_values

    def __explod_ranked_paths(self, ranked_items: list, items_historic: list, semantic_profile: dict, file: _io.TextIOWrapper):
        """
        Build explanation to recommendations based on the ExpLOD, method, explained in https://dl.acm.org/doi/abs/10.1145/2959100.2959173
        :param ranked_items: list of the recommended items
        :param items_historic: list of historic items
        :param semantic_profile: dictionary with property as key and score as value
        :param file: file to write explanations
        :return: historic items and properties used in explanations
        """

        # get properties from historic and recommended items
        hist_props = self.prop_set.loc[items_historic]
        hist_items = {}
        nodes = {}

        for r in ranked_items:
            rec_props = self.prop_set.loc[r]

            # check properties on both sets
            intersection = pd.Series(list(set(hist_props['obj']).intersection(set(rec_props['obj']))))

            # get properties with max value
            max = -1
            max_props = []
            for pi in intersection:
                value = semantic_profile[pi]
                if value > max:
                    max = value
                    max_props.clear()
                    max_props.append(pi)
                elif value == max:
                    max_props.append(pi)

            # build sentence
            hist_names = hist_props[hist_props['obj'].isin(max_props)]['title'].unique()
            rec_name = rec_props[rec_props['obj'].isin(max_props)]['title'].unique()[0]

            print("\nPaths for the Recommended Item: " + str(r))
            file.write("\nPaths for the Recommended Item: " + str(i) + "\n")
            origin = ""
            # check for others with same value
            for i in hist_names:
                origin = origin + "\"" + i + "\"; "
                hist_items = self.__add_dict(hist_items, i)
            origin = origin[:-2]

            path_sentence = " nodes: "
            for n in max_props:
                path_sentence = path_sentence + "\"" + n + "\" "
                nodes = self.__add_dict(nodes, n)
            destination = "destination: \"" + rec_name + "\""
            print(origin + path_sentence + destination)
            file.write(origin + path_sentence + destination)

        return hist_items, nodes

    def __add_dict(self, d: dict, key) -> dict:
        """
        Function to increment one in the key
        :param d: dictionary
        :param key: key to increment value
        :return: new dictionary
        """
        try:
            d[key] = d[key] + 1
        except KeyError:
            d[key] = 1

        return d
