import _io
import os
import random
from collections import Counter
import time
import openai
import re
import inspect

import networkx as nx
import numpy as np
import pandas as pd
from pandas.core.indexing import IndexingError
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from gensim.models.keyedvectors import KeyedVectors
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from groq import Groq, RateLimitError

import evaluation_utils as eval
from recommenders.lod_reordering import LODPersonalizedReordering
import torch


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

    def reorder_with_path(self, fold: str, h_min: int, h_max: int, max_users: int, expl_alg: str, reordered: int,
                          n_explain: int):
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

        semantic_pro = True
        if expl_alg in ["rotate", "word2vec", "explod_v2", "pem"] or expl_alg.startswith("webmedia") \
                or expl_alg.startswith("llm"):
            semantic_pro = False

        dataset = "ml"
        if self.prop_path.split("/")[3] == "last-fm":
            dataset = "last-fm"

        # create result and output file names
        if reordered:
            results_file_name = fold + "/results/explanations/" + self.output_path.split("/")[-1]
        else:
            results_file_name = fold + "/results/explanations/" + self.output_rec_file.split("/")[-1]

        results_title_l = results_file_name.split("/")
        results_title = '/'.join(results_title_l[:-1])
        if n_explain != 5:
            results_title = results_title + "/reordered_recs=" + str(reordered) + "_expl_alg=" + expl_alg + "_" + \
                            "n_explain=" + str(n_explain) + "_" + results_title_l[-1]
        else:
            results_title = results_title + "/reordered_recs=" + str(reordered) + "_expl_alg=" + expl_alg + "_" \
                            + results_title_l[-1]

        if reordered:
            output_file_name = fold + "/outputs/explanations/" + self.output_path.split("/")[-1]
        else:
            output_file_name = fold + "/outputs/explanations/" + self.output_rec_file.split("/")[-1]

        output_title_l = output_file_name.split("/")
        output_title = '/'.join(output_title_l[:-1])
        output_title = output_title + "/reordered_recs=" + str(reordered) + "_expl_alg=" + expl_alg + "_" + \
                       "n_explain=" + str(n_explain) + "_" + output_title_l[-1]

        if expl_alg.startswith("webmedia"):
            output_title = output_title.replace("/explanations/", "/explanations/webmedia/")
            results_title = results_title.replace("/explanations/", "/explanations/webmedia/")
            print(output_title)
            print(results_title)

        if expl_alg.startswith("llm"):
            load_dotenv()
            output_title = output_title.replace("/explanations/", "/explanations/llm/")
            results_title = results_title.replace("/explanations/", "/explanations/llm/")
            print(output_title)
            print(results_title)

        f = open(output_title, mode="w", encoding='utf-8')
        f.write(output_title + "\n")
        print(output_title)
        n_users = 0
        m_items = []
        m_props = []
        total_items = {}
        total_props = {}
        total_lir = []
        total_etd = []
        total_sep = []
        memo_sep = {}
        ulir = -1
        usep = -1
        uetd = -1
        for u in self.output_rec_set.index.unique():
            # get items that the user interacted and recommended by an algorithm
            items_historic = self.train_set.loc[u].sort_values(by=self.cols_used[-1], ascending=False)

            if h_min >= 0 and h_max > 0 and max_users > 0:
                if len(items_historic) <= h_min or len(items_historic) >= h_max:
                    continue
                if n_users == max_users:
                    break
                n_users = n_users + 1
            elif h_min == -1 and h_max == -1 and max_users > 0:
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
            if semantic_pro:
                sem_dist = self.__semantic_path_distance(items_historic_cutout, items_recommended,
                                                         user_semantic_profile)
                # create column with the sum of paths
                sem_dist['score'] = pd.DataFrame(sem_dist['path'].to_list()).mean(1)

            if reordered and semantic_pro:
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
                item_rank = list(reordered_items['item_id'])[:n_explain]
                for i in item_rank:
                    movie_name = self.prop_set.loc[i].iloc[0, 0]
                    print("Item id: " + str(i) + " Name: " + movie_name)
                    f.write("Item id: " + str(i) + " Name: " + movie_name + "\n")

            else:
                item_rank = self.output_rec_set
                item_rank = list(item_rank.loc[u][:n_explain]["item_id"])

                print("\nRecommendations")
                f.write("\nRecommendations\n")
                for i in item_rank:
                    try:
                        movie_name = self.prop_set.loc[i].iloc[0, 0]
                    except IndexingError:
                        movie_name = self.prop_set.loc[i].iloc[0]
                    print("Item id: " + str(i) + " Name: " + movie_name)
                    f.write("Item id: " + str(i) + " Name: " + movie_name + "\n")

            if semantic_pro:
                sem_dist = sem_dist.set_index('recommended')
                sem_dist = sem_dist.fillna(0)

            items, props = [], []
            if expl_alg == 'diverse':
                items, props, (ulir, usep, uetd) = self.__diverse_ranked_paths(item_rank, sem_dist,
                                                                               user_semantic_profile, u,
                                                                               items_historic_cutout, f, memo_sep)
            elif expl_alg == 'explod':
                items, props, (ulir, usep, uetd) = self.__explod_ranked_paths(item_rank, items_historic,
                                                                              user_semantic_profile, u, f, memo_sep)

            elif expl_alg == 'pem':
                items, props, (ulir, usep, uetd) = self.__pem_ranked_paths(item_rank, items_historic, u, dataset,
                                                                           f, memo_sep)

            elif expl_alg == 'explod_v2':
                items, props, (ulir, usep, uetd) = self.__explod_ranked_paths_v2(item_rank, items_historic, u, dataset,
                                                                                 f, memo_sep)

            elif expl_alg == "word2vec":
                items, props, (ulir, usep, uetd) = self.__word2vec_embeedings(item_rank, items_historic, u, f, memo_sep)

            elif expl_alg == "rotate":
                items, props, (ulir, usep, uetd) = self.__rotate_embedding(item_rank, items_historic, u, f, memo_sep)

            elif expl_alg.startswith("webmedia"):
                model_name = expl_alg.split("_")[-1]
                items, props, (ulir, usep, uetd) = self.__webmedia_embedding(model_name, item_rank, items_historic, u,
                                                                             f, memo_sep)

            elif expl_alg.startswith("llm"):
                model_name = expl_alg.split("_")[-1]
                items, props, (ulir, usep, uetd) = self.__llm_paths(model_name, item_rank, items_historic, u,
                                                                    f, memo_sep)
            f.write("\n")

            total_items = dict(Counter(total_items) + Counter(items))
            m_items.append(len(items))
            total_props = dict(Counter(total_props) + Counter(props))
            m_props.append(len(props))
            total_lir.append(ulir)
            total_sep.append(usep)
            total_etd.append(uetd)

        f.close()
        eval.evaluate_explanations(results_title, m_items, m_props, total_items, total_props, self.train_set,
                                   self.prop_set, total_lir, total_sep, total_etd)

    def __semantic_path_distance(self, historic: list, recommeded: list, semantic_profile: dict) -> pd.DataFrame:
        """
        Get the best path based on the semantic profile from all the historic items to the recommended ones
        :param historic: list of items that the user interacted
        :param recommeded: recommended items by the user
        :param semantic_profile: semantic profile of the user
        :return: data frame with historic item, the recommended and path
        """
        sem_path_dist = pd.DataFrame(columns=['historic', 'recommended', 'path', 'path_s'])
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
                    paths_s = [p for p in paths]
                    paths = [list(map(semantic_profile.get, p[1::2])) for p in paths_s]
                    values = [sum(values) / len(values) for values in paths if len(values) > 0 or values is None]
                    sem_path_dist = sem_path_dist.append(
                        {'historic': hm, 'recommended': rm, 'path': paths[np.argmax(values)],
                         'path_s': paths_s[np.argmax(values)]},
                        ignore_index=True)
                except (nx.exception.NetworkXNoPath, ValueError):
                    sem_path_dist = sem_path_dist.append({'historic': hm, 'recommended': rm, 'path': [], 'path_s': []},
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

    def __diverse_ranked_paths(self, ranked_items: list, semantic_distance: pd.DataFrame, semantic_profile: dict,
                               user: int, historic_items: list, file: _io.TextIOWrapper, memo_sep: dict):
        """
        Generate explanations to recommendations considering the max value varying the properties shown to the user
        the logic to this explanation is to order all paths to recommended items and resolve the conflicts (when there
        is a repetition of properties) from biggest value to lowest value recursively until all items have different
        properties. If there is a conflict with the lowest prop value, then we repeat the highest values
        :param ranked_items: list of the recommended items
        :param semantic_distance: dataframe with paths from historic to recommended items
        :param semantic_profile: dictionary with property as key and score as value
        :param user: user id of user to show explanations to
        :param file: file to write explanations
        :return: historic items and properties used in explanations
        """
        hist_items = {}
        nodes = {}

        high_values = self.__diverse_ordered_properties(ranked_items, semantic_distance)

        subgraph = self.graph.subgraph(["I" + str(int(h)) for h in historic_items] +
                                       ["I" + str(int(h)) for h in list(high_values.index)] +
                                       list(self.prop_set.loc[historic_items]['obj']))
        hist_lists = []
        prop_lists = []
        # display the explanation path for every recommendation
        for i in high_values.index:
            paths = []
            path_set = {}

            # obtain all paths
            for j in list(high_values['historic'].unique()):
                try:
                    paths = paths + [p for p in nx.all_shortest_paths(subgraph, source="I" + str(int(j)),
                                                                      target="I" + str(int(i)))]
                except nx.exception.NetworkXNoPath:
                    pass

            # obtain paths with the properties selected by the diverse_ordered_properties method explanations
            for p in paths:
                path_values = [set(map(semantic_profile.get, p[1::2]))]
                for value in path_values:
                    if value == set(high_values.loc[i]['path']):
                        for k in range(0, len(p) - 1, 2):
                            n = p[k]
                            if n.startswith("I"):
                                item_id = int(n[1:])
                                if item_id != i:
                                    try:
                                        path_set[p[k + 1]].add(item_id)
                                    except KeyError:
                                        path_set[p[k + 1]] = {item_id}

            # get items names from ids do generate explanations
            for k in path_set.keys():
                s_items = self.train_set.loc[user].sort_values(by=self.train_set.columns[-1], kind="quicksort",
                                                               ascending=False)
                s_items = s_items[s_items[self.train_set.columns[0]].isin(list(path_set[k]))]
                item_names = []
                for h in s_items[s_items.columns[0]].to_list():
                    item_names.append(list(self.prop_set.loc[h][self.prop_set.columns[0]])[0])
                path_set[k] = item_names

            # generate sentence using a top of 3 items
            origin = ""
            path_sentence = " nodes: "
            n = 0
            ind = [0 for _ in path_set.keys()]
            k_ind = 0
            keys = list(path_set.keys())
            used_items = []
            used_props = []
            hist_ids = []
            end_flag = sum([len(path_set[keys[h]]) <= ind[h] for h in range(0, len(keys))]) == len(keys)
            while n < 3 and not end_flag:
                key = keys[k_ind]
                try:
                    ori = path_set[key][ind[k_ind]]
                    if ori not in used_items:
                        origin = origin + "\"" + ori + "\"; "
                        hist_items = self.__add_dict(hist_items, ori)
                        used_items.append(ori)
                        ids = self.prop_set[self.prop_set[self.prop_set.columns[0]] == ori].index.values

                        found = False
                        ids = list(set(ids))
                        j = 0
                        while j < len(ids) and not found:
                            if ids[j] in historic_items:
                                hist_ids.append(ids[j])
                                found = True
                            j = j + 1

                        if not found:
                            hist_ids.append(ids[0])

                    if key not in used_props:
                        path_sentence = path_sentence + "\"" + key + "\" "
                        nodes = self.__add_dict(nodes, key)
                        used_props.append(key)

                    n = n + 1
                except IndexError:
                    pass
                ind[k_ind] = ind[k_ind] + 1
                k_ind = k_ind + 1
                if k_ind == len(keys):
                    k_ind = 0
                end_flag = sum([len(path_set[keys[h]]) <= ind[h] for h in range(0, len(keys))]) == len(keys)

            hist_lists.append(hist_ids)
            prop_lists.append(used_props)
            print("\nPaths for the Recommended Item: " + str(i))
            file.write("\nPaths for the Recommended Item: " + str(i) + "\n")

            origin = origin[:-2]
            destination = "destination: \"" + self.prop_set.loc[i].iloc[0, 0] + "\""
            print(origin + path_sentence + destination)
            file.write(origin + path_sentence + destination)

        lir = eval.lir_metric(0.3, user, hist_lists, self.train_set)
        sep = eval.sep_metric(0.3, prop_lists, self.prop_set, memo_sep)
        etd = eval.etd_metric(list(nodes.keys()), len(ranked_items), len(self.prop_set['obj'].unique()))
        return hist_items, nodes, (lir, sep, etd)

    def __diverse_ordered_properties(self, ranked_items: list, semantic_distance: pd.DataFrame, lowest_flag=False):
        """
        Order the explanation paths in order to maximize value, without or repeating only a few times properites
        :param ranked_items: list of recommended items
        :param semantic_distance: dataframe with paths from historic to recommended items
        :param lowest_flag: boolean flag that, if True consider on the reordering the lowest valued property
            as explanation when there is a conflict. If False, replace the lowest value for repetitions
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
            if len(list(df_max['score'].index)) == 1:
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

                # verify if exist
                if self.__empty_paths(second_high) == 1:
                    raise KeyError("There are not any other values for lowest indexes")

                # if there is only one value to substitute, substitute this value
                if len(second_high) == 1:
                    sub_list = second_high.index

                for i in sub_list:
                    l = second_high.loc[i]
                    if len(l) > 0:
                        high_values.loc[i] = second_high.loc[i]

                # get next max to check for conflicts
                order_values = high_values['score'].sort_values(ascending=False)
                count = list(order_values).index(maximum) + 1
                maximum = order_values.iloc[count]
                df_max = high_values[high_values['score'] == maximum]
            # if there the conflicts was not resolved (lowest value is a tie) then recursively repeat the best
            # properties only for the items with tie
            except KeyError:
                if len(df_max.index) > 1 and len(df_max.index) != len(high_values.index) and \
                        not self.__only_one_value(df_max, semantic_distance):
                    second_high = self.__diverse_ordered_properties(list(df_max.index), semantic_distance)
                    if lowest_flag:
                        second_high = second_high.sort_values(by="score", kind="quicksort", ascending=False)
                        lowest_index = second_high.index[-1]
                        second_high.loc[lowest_index] = df_max.loc[lowest_index]
                if self.__empty_paths(second_high) == 2:
                    for i in second_high.index:
                        high_values.loc[i] = second_high.loc[i]
                break

        return high_values

    def __only_one_value(self, df_max: pd.DataFrame, semantic_distance: pd.DataFrame):
        """
        Check if the items to be recommended have only one path to the historical items
        :param df_max: dataframe with items with explanation conflict
        :param semantic_distance: dataframe with the semantic distances between historical and recommended items
        :return: True if there is only one path for all items with explanation conflict, false otherwise
        """
        count = 0
        rec_items = list(df_max.index)
        for rec_item in rec_items:
            if len(semantic_distance.loc[rec_item]['score'].unique()) == 1:
                count = count + 1

        if count == len(rec_items):
            return True
        else:
            return False

    def __empty_paths(self, paths_df: pd.DataFrame):
        """
        Function that determines if a pandas dataframe with origin, path and destination has empty paths
        :param paths_df: pandas dataframe with with origin, path and destination as columns
        :return: 1 if all paths are empty, 2 if all paths are filled, 0 if there are some empty paths
        """
        c = 0
        for i, row in paths_df.iterrows():
            if len(row[1]) == 0:
                c = c + 1
        if c == len(paths_df.index):
            return 1
        elif c == 0:
            return 2
        else:
            return 0

    def __explod_ranked_paths(self, ranked_items: list, items_historic: list, semantic_profile: dict,
                              user: int, file: _io.TextIOWrapper, memo_sep: dict):
        """
        Build explanation to recommendations based on the ExpLOD, method, explained in https://dl.acm.org/doi/abs/10.1145/2959100.2959173
        :param ranked_items: list of the recommended items
        :param items_historic: list of historic items
        :param semantic_profile: dictionary with property as key and score as value
        :param user: user id of user to show explanations to
        :param file: file to write explanations
        :return: historic items and properties used in explanations
        """

        # get properties from historic and recommended items
        hist_props = self.prop_set.loc[items_historic]
        hist_items = {}
        nodes = {}
        hist_lists = []
        prop_lists = []
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
            user_df = self.train_set.loc[user]
            user_item = user_df[
                user_df[user_df.columns[0]].isin(list(hist_props[hist_props['obj'].isin(max_props)].index.unique()))]
            hist_ids = list(user_item.sort_values(by=user_item.columns[-1], ascending=False)[:3][user_item.columns[0]])
            hist_lists.append(hist_ids)
            hist_names = hist_props.loc[hist_ids][self.prop_cols[1]].unique()
            try:
                rec_name = self.prop_set.loc[r][self.prop_cols[1]].unique()[0]
            except AttributeError:
                rec_name = self.prop_set.loc[r][self.prop_cols[1]]

            print("\nPaths for the Recommended Item: " + str(r))
            file.write("\nPaths for the Recommended Item: " + str(r) + "\n")
            origin = ""
            # check for others with same value
            for i in hist_names:
                origin = origin + "\"" + i + "\"; "
                hist_items = self.__add_dict(hist_items, i)
            origin = origin[:-2]

            path_sentence = " nodes: "
            prop_lists.append(max_props)
            for n in max_props:
                path_sentence = path_sentence + "\"" + n + "\" "
                nodes = self.__add_dict(nodes, n)
            destination = "destination: \"" + rec_name + "\""
            print(origin + path_sentence + destination)
            file.write(origin + path_sentence + destination)

        lir = eval.lir_metric(0.3, user, hist_lists, self.train_set)
        sep = eval.sep_metric(0.3, prop_lists, self.prop_set, memo_sep)
        etd = eval.etd_metric(list(nodes.keys()), len(ranked_items), len(self.prop_set['obj'].unique()))

        return hist_items, nodes, (lir, sep, etd)

    def __explod_ranked_paths_v2(self, ranked_items: list, items_historic: list, user: int, dataset: str,
                                 file: _io.TextIOWrapper, memo_sep: dict):
        """
        Build explanation to recommendations based on the ExpLOD, method, explained in https://dl.acm.org/doi/abs/10.1145/2959100.2959173
        :param ranked_items: list of the recommended items
        :param items_historic: list of historic items
        :param user: user id of user to show explanations to
        :param dataset: string that represent the dataset. It is either 'ml' or 'last-fm'
        :param file: file to write explanations
        :return: historic items and properties used in explanations
        """
        # get properties from historic and recommended items
        hist_props = self.prop_set.loc[items_historic]
        rec_props = self.prop_set.loc[ranked_items]
        union = pd.Series(list(set(hist_props['obj']).intersection(set(rec_props['obj']))))

        # generating rank of properties for the user
        # create npi, i and n columns
        interacted_props = self.prop_set.loc[items_historic + ranked_items].copy()
        interacted_props['npi'] = interacted_props.groupby(self.prop_set.columns[-1])[
            self.prop_set.columns[-1]].transform('count')
        interacted_props['i'] = len(items_historic)
        interacted_props['n'] = len(self.prop_set.index.unique())

        # get items per property on full dbpedia/wikidata by dropping the duplicates with same item id and prop value
        # therefore, a value that repeats in the same item is ignored
        items_per_obj = self.prop_set.reset_index().drop_duplicates(
            subset=[self.prop_set.columns[0], self.prop_set.columns[-1]]).set_index(
            self.prop_set.columns[-1])
        df_dict = items_per_obj.index.value_counts().to_dict()

        # generate the dft column based on items per property and score column base on all new created columns
        interacted_props['dft'] = interacted_props.apply(lambda x: df_dict[x[self.prop_set.columns[-1]]], axis=1)

        interacted_props['r'] = 1
        interacted_props['npr'] = interacted_props.apply(lambda x: 1 if x['obj'] in rec_props else 0, axis=1)

        interacted_props['score'] = ((interacted_props['npi'] / interacted_props['i']) +
                                     (interacted_props['npr'] / interacted_props['r'])) * \
                                    (np.log(interacted_props['n'] / interacted_props['dft']))

        interacted_props = interacted_props[['obj', 'score', 'dft']].drop_duplicates()

        if dataset == "ml":
            hierarchy_df = pd.read_csv("./generated_files/wikidata/props_hierarchy_wikidata_movielens_small.csv")
        elif dataset == "last-fm":
            hierarchy_df = pd.read_csv("./generated_files/wikidata/last-fm/props_hierarchy_wikidata_lastfm_small.csv")

        historic_hierarchy_props_l1 = hierarchy_df[hierarchy_df['obj'].isin(union)]
        historic_hierarchy_props_l1 = historic_hierarchy_props_l1.merge(interacted_props[['obj', 'score', 'dft']],
                                                                        on='obj', how='left').drop_duplicates()

        historic_hierarchy_props_l1 = historic_hierarchy_props_l1.groupby('super_obj').sum()
        historic_hierarchy_props_l1['n'] = len(self.prop_set.index.unique())
        historic_hierarchy_props_l1['score_final'] = historic_hierarchy_props_l1['score'] * \
                                                     (np.log(
                                                         historic_hierarchy_props_l1['n'] / historic_hierarchy_props_l1[
                                                             'dft']))
        historic_hierarchy_props_l1 = historic_hierarchy_props_l1.reset_index()
        historic_hierarchy_props_l1 = historic_hierarchy_props_l1[['super_obj', 'score_final']]. \
            rename({'super_obj': 'obj', 'score_final': 'score'}, axis=1)

        props_score_df = pd.concat([historic_hierarchy_props_l1, interacted_props[['obj', 'score']]], ignore_index=True)
        props_score_df = props_score_df.groupby('obj').sum()

        hist_items = {}
        nodes = {}
        hist_lists = []
        prop_lists = []
        for r in ranked_items:
            rec_props = self.prop_set.loc[r]

            # check properties on both sets
            valid_props = pd.Series(list(set(hist_props['obj']).intersection(set(rec_props['obj']))))
            prop_order = props_score_df.loc[valid_props]

            # get properties with max value
            max = -1
            max_props = []
            for pi in prop_order.index:
                value = prop_order.loc[pi][0]
                if value > max:
                    max = value
                    max_props.clear()
                    max_props.append(pi)
                elif value == max:
                    max_props.append(pi)

            # build sentence
            user_df = self.train_set.loc[user]
            user_item = user_df[
                user_df[user_df.columns[0]].isin(list(hist_props[hist_props['obj'].isin(max_props)].index.unique()))]
            hist_ids = list(user_item.sort_values(by=user_item.columns[-1], ascending=False)[:3][user_item.columns[0]])
            hist_lists.append(hist_ids)
            hist_names = hist_props.loc[hist_ids][self.prop_cols[1]].unique()
            try:
                rec_name = self.prop_set.loc[r][self.prop_cols[1]].unique()[0]
            except AttributeError:
                rec_name = self.prop_set.loc[r][self.prop_cols[1]]

            print("\nPaths for the Recommended Item: " + str(r))
            file.write("\nPaths for the Recommended Item: " + str(r) + "\n")
            origin = ""
            # check for others with same value
            for i in hist_names:
                origin = origin + "\"" + i + "\"; "
                hist_items = self.__add_dict(hist_items, i)
            origin = origin[:-2]

            path_sentence = " nodes: "
            prop_lists.append(max_props)
            for n in max_props:
                path_sentence = path_sentence + "\"" + n + "\" "
                nodes = self.__add_dict(nodes, n)
            destination = "destination: \"" + rec_name + "\""
            print(origin + path_sentence + destination)
            file.write(origin + path_sentence + destination)

        lir = eval.lir_metric(0.3, user, hist_lists, self.train_set)
        sep = eval.sep_metric(0.3, prop_lists, self.prop_set, memo_sep)
        etd = eval.etd_metric(list(nodes.keys()), len(ranked_items), len(self.prop_set['obj'].unique()))
        return hist_items, nodes, (lir, sep, etd)

    def __pem_ranked_paths(self, ranked_items: list, items_historic: list, user: int, dataset: str,
                           file: _io.TextIOWrapper, memo_sep: dict):
        # count historic movies properties considering two levels of hierarchy above
        if dataset == "ml":
            hierarchy_df = pd.read_csv("./generated_files/wikidata/props_hierarchy_wikidata_movielens_small.csv")
        elif dataset == "last-fm":
            hierarchy_df = pd.read_csv("./generated_files/wikidata/last-fm/props_hierarchy_wikidata_lastfm_small.csv")

        hist_props = self.prop_set.loc[items_historic]
        hist_props_l = hist_props['obj'].unique().tolist()

        historic_hierarchy_props_l1 = hierarchy_df[hierarchy_df['obj'].isin(hist_props_l)]
        historic_hierarchy_list_l1 = historic_hierarchy_props_l1['super_obj'].to_list()
        historic_hierarchy_props_l1 = historic_hierarchy_props_l1[
            historic_hierarchy_props_l1.obj != historic_hierarchy_props_l1.super_obj]

        historic_hierarchy_props_l2 = hierarchy_df[hierarchy_df['obj'].isin(historic_hierarchy_list_l1)]
        historic_hierarchy_props_l2 = historic_hierarchy_props_l2[
            historic_hierarchy_props_l2.obj != historic_hierarchy_props_l2.super_obj]

        historic_hierarchy_list_l2 = historic_hierarchy_props_l2['super_obj'].to_list()

        count_hist_df = hist_props[[hist_props.columns[0], hist_props.columns[-1]]].drop_duplicates().groupby(
            'obj').count()
        count_hist_df.columns = ['count']
        count_hist_hierarchy_l1 = count_hist_df.merge(historic_hierarchy_props_l1, on='obj', how='right')
        count_hist_hierarchy_l2 = count_hist_hierarchy_l1.merge(historic_hierarchy_props_l2, left_on='super_obj',
                                                                right_on='obj', how='right')
        count_hist_hierarchy_l2 = count_hist_hierarchy_l2[['super_obj_x', 'count', 'super_obj_y']]
        count_hist_hierarchy_l2.columns = ['obj', 'count', 'super_obj']

        hist_items = {}
        nodes = {}
        hist_lists = []
        prop_lists = []
        for r in ranked_items:
            # count recommended item properties considering two levels of hierarchy above
            rec_props = self.prop_set.loc[r]
            try:
                rec_props_l = list(rec_props['obj'].unique())
            except AttributeError:
                rec_props_l = [rec_props['obj']]
                rec_props = pd.DataFrame(rec_props).T
            rec_hierarchy_props_l1 = hierarchy_df[hierarchy_df['obj'].isin(rec_props_l)]
            rec_hierarchy_props_l1 = rec_hierarchy_props_l1[
                rec_hierarchy_props_l1.obj != rec_hierarchy_props_l1.super_obj]

            rec_hierarchy_list_l1 = rec_hierarchy_props_l1['super_obj'].to_list()

            rec_hierarchy_props_l2 = hierarchy_df[hierarchy_df['obj'].isin(rec_hierarchy_list_l1)]
            rec_hierarchy_props_l2 = rec_hierarchy_props_l2[
                rec_hierarchy_props_l2.obj != rec_hierarchy_props_l2.super_obj]

            rec_hierarchy_list_l2 = rec_hierarchy_props_l2['super_obj'].to_list()

            # obtain all properties from historic hierarchy and recommended hierarchy, then filter to only valids
            all_user_props = set(hist_props_l + historic_hierarchy_list_l1 + historic_hierarchy_list_l2). \
                intersection(rec_props_l + rec_hierarchy_list_l1 + rec_hierarchy_list_l2)

            # valid properties are properties from the hierarchy that are from the user profile or recommended hierarchy
            # and annotate at least one item from the set
            valid_props = set(all_user_props).intersection(set(self.prop_set['obj'].to_list()))

            # count number of items described by each of valid props
            count_rec_df = rec_props[[rec_props.columns[0], rec_props.columns[-1]]].drop_duplicates().groupby(
                'obj').count()
            count_rec_df.columns = ['count']
            count_rec_hierarchy_l1 = count_rec_df.merge(rec_hierarchy_props_l1, on='obj', how='right')
            count_rec_hierarchy_l2 = count_rec_hierarchy_l1.merge(rec_hierarchy_props_l2, left_on='super_obj',
                                                                  right_on='obj', how='right')
            count_rec_hierarchy_l2 = count_rec_hierarchy_l2[['super_obj_x', 'count', 'super_obj_y']]
            count_rec_hierarchy_l2.columns = ['obj', 'count', 'super_obj']

            all_user = pd.concat([count_hist_hierarchy_l1, count_hist_hierarchy_l2,
                                  count_rec_hierarchy_l1, count_rec_hierarchy_l2], ignore_index=True)
            all_user = all_user.groupby('super_obj').count()['count']
            all_user = pd.concat(
                [all_user, count_hist_df[count_hist_df.columns[0]], count_rec_df[count_rec_df.columns[0]]])
            all_user = all_user.groupby(level=0).sum()
            all_user_valid = all_user.loc[list(valid_props)]

            # remove redundant properties, that are super properties that are described by the same amount of user
            # liked items compared to their respective child properties
            node_dicts = {}
            graph = [historic_hierarchy_props_l1, historic_hierarchy_props_l2,
                     rec_hierarchy_props_l1, rec_hierarchy_props_l2]
            for subgraph in graph:
                for index, row in subgraph.iterrows():
                    try:
                        node_dicts[row[1]].add(row[0])
                    except KeyError:
                        node_dicts[row[1]] = {row[0]}

            non_redundant = []
            for obj in all_user_valid.index.to_list():
                try:
                    childs = list(node_dicts[obj])
                except KeyError:
                    non_redundant.append(obj)
                    continue

                obj_count = all_user_valid.loc[obj]
                redundant_f = False
                n_iter = 0
                while len(childs) > 0 and n_iter < 100:
                    super_p = childs.pop()
                    if obj_count == all_user.loc[super_p]:
                        redundant_f = True
                        break

                    try:
                        childs = childs + list(node_dicts[super_p])
                    except KeyError:
                        pass
                    finally:
                        n_iter = n_iter + 1

                if not redundant_f:
                    non_redundant.append(obj)

            # score properties
            props_catalog = self.prop_set[[self.prop_set.columns[0], self.prop_set.columns[-1]]].drop_duplicates() \
                .groupby("obj").count()
            props_catalog = pd.DataFrame(props_catalog[props_catalog.columns[0]])
            props_catalog.columns = ['count']

            catalog_hierarchy_props_l1 = hierarchy_df[hierarchy_df['obj'].isin(all_user.index)]
            catalog_hierarchy_list_l1 = catalog_hierarchy_props_l1['super_obj'].to_list()

            catalog_hierarchy_props_l2 = hierarchy_df[hierarchy_df['obj'].isin(catalog_hierarchy_list_l1)]

            count_catalog_hierarchy_l1 = props_catalog.merge(catalog_hierarchy_props_l1, on='obj', how='right')
            count_catalog_hierarchy_l2 = count_catalog_hierarchy_l1.merge(catalog_hierarchy_props_l2,
                                                                          left_on='super_obj',
                                                                          right_on='obj', how='right')
            count_catalog_hierarchy_l2 = count_catalog_hierarchy_l2[['super_obj_x', 'count', 'super_obj_y']]
            count_catalog_hierarchy_l2.columns = ['obj', 'count', 'super_obj']
            all_catalog = pd.concat([count_catalog_hierarchy_l1, count_catalog_hierarchy_l2], ignore_index=True)
            all_catalog = pd.DataFrame(all_catalog.groupby("super_obj").sum())
            all_catalog = pd.concat([all_catalog, props_catalog])
            all_catalog = all_catalog.reset_index()
            all_catalog = all_catalog.groupby(all_catalog.columns[0]).sum().squeeze()

            c = len(self.prop_set.index.unique())
            iu = len(items_historic)
            props_rank = {}
            for p in non_redundant:
                ipc = all_catalog.loc[p]
                ipu = all_user.loc[p]
                props_rank[p] = np.log10(ipc) * ((ipu / iu) / (ipc / c))

            prop_rank_l = list(dict(sorted(props_rank.items(), key=lambda item: item[1], reverse=True)).keys())

            if len(prop_rank_l) > 0:
                max_prop = [prop_rank_l[0]]
                user_df = self.train_set.loc[user]
                user_item = user_df[
                    user_df[user_df.columns[0]].isin(list(hist_props[hist_props['obj'] == max_prop[0]].index.unique()))]
                hist_ids = list(
                    user_item.sort_values(by=user_item.columns[-1], ascending=False)[:3][user_item.columns[0]])
                sub_p = [prop_rank_l[0]]
                while len(hist_ids) == 0:
                    user_item = user_df[
                        user_df[user_df.columns[0]].isin(
                            list(hist_props[hist_props['obj'].isin(sub_p)].index.unique()))]
                    hist_ids = list(
                        user_item.sort_values(by=user_item.columns[-1], ascending=False)[:3][user_item.columns[0]])
                    new_sub_p = []
                    for p in sub_p:
                        try:
                            new_sub_p = new_sub_p + list(node_dicts[p])
                        except KeyError:
                            continue
                    sub_p = new_sub_p

                hist_lists.append(hist_ids)
                hist_names = hist_props.loc[hist_ids][self.prop_cols[1]].unique()
            else:
                max_prop = []
                hist_ids = self.train_set.loc[user].sort_values(by=self.train_set.columns[-1], ascending=False)[:3][
                    self.train_set.columns[0]]
                hist_names = hist_props.loc[hist_ids][self.prop_cols[1]].unique()
                hist_lists.append(hist_ids)

            try:
                rec_name = self.prop_set.loc[r][self.prop_cols[1]].unique()[0]
            except AttributeError:
                rec_name = self.prop_set.loc[r][self.prop_cols[1]]
            # building explanation
            print("\nPaths for the Recommended Item: " + str(r))
            file.write("\nPaths for the Recommended Item: " + str(r) + "\n")
            origin = ""
            for i in hist_names:
                origin = origin + "\"" + i + "\"; "
                hist_items = self.__add_dict(hist_items, i)
            origin = origin[:-2]

            prop_lists.append(max_prop)
            path_sentence = " nodes: "
            if len(max_prop) > 0:
                n = max_prop[0]
                path_sentence = path_sentence + "\"" + n + "\" "
                nodes = self.__add_dict(nodes, n)

            destination = "destination: \"" + rec_name + "\""
            print(origin + path_sentence + destination)
            file.write(origin + path_sentence + destination)

        lir = eval.lir_metric(0.3, user, hist_lists, self.train_set)
        sep = eval.sep_metric(0.3, prop_lists, self.prop_set, memo_sep)
        etd = eval.etd_metric(list(nodes.keys()), len(ranked_items), len(self.prop_set['obj'].unique()))

        return hist_items, nodes, (lir, sep, etd)

    def __word2vec_embeedings(self, recommeded: list, historic: list, user: int, file: _io.TextIOWrapper,
                              memo_sep: dict,
                              save_model=False):
        model_path = self.train_file.split("/")
        model_path = '/'.join(model_path[:-3])
        model_path = model_path + "/models/word2_vec_model"
        hist_props = self.prop_set.loc[historic]
        user_code = "U" + str(user)
        # obtain user embeeding
        if save_model:
            node2vec = Node2Vec(self.graph, dimensions=64, walk_length=30, num_walks=100, workers=1)
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
            model.save(model_path)
            print("--- Models Saved ---")
        else:
            model = KeyedVectors.load(model_path)

        sem_path_dist = pd.DataFrame(columns=['historic', 'recommended', 'path_s', 'path_v', 'values'])
        historic_codes = ['I' + str(i) for i in historic]
        recommeded_codes = ['I' + str(i) for i in recommeded]
        historic_props = list(set(self.prop_set.loc[self.prop_set.index.isin(historic)]['obj']))

        subgraph = self.graph.subgraph(historic_codes + recommeded_codes + historic_props)

        # obtain paths from historic item to recommended
        for hm in historic:
            hm_node = 'I' + str(hm)
            for rm in recommeded:
                rm_name = 'I' + str(rm)
                try:
                    paths = nx.all_shortest_paths(subgraph, source=hm_node, target=rm_name)
                    paths_s = [p for p in paths if len(p) <= 5]
                    all_path_v = []
                    for p in paths_s:
                        path_v = []
                        for prop in p:
                            path_v.append(model.wv.similarity(user_code, prop))
                        all_path_v.append(path_v)

                    values = [sum(values) / len(values) for values in all_path_v if len(values) > 0 or values is None]
                    if len(values) == 0:
                        values = [0]
                    sem_path_dist = sem_path_dist.append(
                        {'historic': hm, 'recommended': rm, 'path_s': paths_s, 'path_v': all_path_v, 'values': values},
                        ignore_index=True)
                    # obtain sum of similarity from  uer embeeding with properties of the path
                except (nx.exception.NetworkXNoPath, ValueError):
                    sem_path_dist = sem_path_dist.append(
                        {'historic': hm, 'recommended': rm, 'path_s': [-1], 'path_v': [-1], 'values': [-1]},
                        ignore_index=True)

        # choose path with highest mean similarity of sem_path_dist
        sem_path_dist_m = sem_path_dist.pivot(index='historic', columns='recommended', values='values')
        sem_path_dist_m = sem_path_dist_m.applymap(max)
        max_paths = sem_path_dist_m.max().to_frame().T

        hist_items = {}
        nodes = {}
        hist_lists = []
        prop_lists = []
        for rec in recommeded:
            print("\nPaths for the Recommended Item: " + str(rec))
            file.write("\nPaths for the Recommended Item: " + str(rec) + "\n")
            try:
                max_value = max_paths[rec][0]
                if max_value == -1:
                    raise KeyError
                origin = sem_path_dist_m[sem_path_dist_m[rec] == max_value].index[0]
                row = sem_path_dist[(sem_path_dist['historic'] == origin) & (sem_path_dist['recommended'] == rec)]
                path_index = list(row["values"])[0].index(max_value)
                path = list(row["path_s"])[0][path_index]

                origin = ""
                hist_names = hist_props.loc[hist_props.index.isin([int(x[1:]) for x in path[:-1][0::2]])][
                    self.prop_cols[1]].unique()
                hist_lists.append([int(x[1:]) for x in path[:-1][0::2]])
                for i in hist_names:
                    origin = origin + "\"" + i + "\"; "
                    hist_items = self.__add_dict(hist_items, i)
                origin = origin[:-2]

                path_props = [x for x in path[:-1][1::2]]
                prop_lists.append(path_props)
                path_sentence = " nodes: "
                for n in path_props:
                    path_sentence = path_sentence + "\"" + n + "\" "
                    nodes = self.__add_dict(nodes, n)

                rec_name = self.prop_set.loc[rec][self.prop_cols[1]].unique()[0]
                destination = "destination: \"" + rec_name + "\""
            except KeyError:
                origin = ""
                path_sentence = ""
                rec_name = self.prop_set.loc[rec][self.prop_cols[1]].unique()[0]
                destination = "destination: \"" + rec_name + "\""
                hist_lists.append([])
                prop_lists.append([])

            print(origin + path_sentence + destination)
            file.write(origin + path_sentence + destination)

        lir = eval.lir_metric(0.3, user, hist_lists, self.train_set)
        sep = eval.sep_metric(0.3, prop_lists, self.prop_set, memo_sep)
        etd = eval.etd_metric(list(nodes.keys()), len(recommeded), len(self.prop_set['obj'].unique()))

        return hist_items, nodes, (lir, sep, etd)

    def __rotate_embedding(self, recommeded: list, historic: list, user: int, file: _io.TextIOWrapper, memo_sep: dict,
                           save_model=False):
        hist_props = self.prop_set.loc[historic]
        model_path = self.train_file.split("/")
        model_path = '/'.join(model_path[:-3])
        model_path = model_path + "/models/rotate_model"
        triples = self.prop_set.to_numpy()
        tf = TriplesFactory.from_labeled_triples(triples)

        # obtain user embeeding
        if save_model:
            # labels are in model
            training, testing, validation = tf.split(ratios=[0.8, 0.1, 0.1], random_state=64)
            result = pipeline(
                training=training,
                testing=testing,
                validation=validation,
                model="RotatE",
                training_loop='sLCWA',
                loss='NSSALoss',
                optimizer='Adam',
                model_kwargs=dict(embedding_dim=200),
                optimizer_kwargs=dict(lr=1e-3),
                training_kwargs=dict(num_epochs=50, batch_size=256),
                use_testing_data=False,
                evaluation_kwargs=dict(batch_size=64),
                random_seed=64,
            )
            result.save_to_directory(model_path)

        model = torch.load(model_path + "/trained_model.pkl")
        entity_embeds = pd.DataFrame(model.entity_representations[0](indices=None).detach().numpy()).rename(
            tf.entity_id_to_label)
        relation_embeds = pd.DataFrame(model.relation_representations[0](indices=None).detach().numpy()).rename(
            tf.relation_id_to_label)

        # create user embeding as pooling of entity
        user_embed = pd.Series(np.zeros(shape=(300,)))
        for i in historic:
            try:
                title = self.prop_set.loc[i][self.prop_set.columns[0]].unique()[0]
            except AttributeError:
                title = str(self.prop_set.loc[i][self.prop_set.columns[0]])

            item_embed = entity_embeds.loc[title]
            user_embed = user_embed + item_embed

        sem_path_dist = pd.DataFrame(columns=['historic', 'recommended', 'path_s', 'path_v'])
        historic_codes = ['I' + str(i) for i in historic]
        recommeded_codes = ['I' + str(i) for i in recommeded]
        historic_props = list(set(self.prop_set.loc[self.prop_set.index.isin(historic)]['obj']))

        subgraph = self.graph.subgraph(historic_codes + recommeded_codes + historic_props)
        titles = self.prop_set[self.prop_cols[1]].to_dict()
        relations = self.prop_set.set_index([self.prop_cols[1], self.prop_cols[-1]]).to_dict()['prop']
        # obtain paths from historic item to recommended
        for hm in historic:
            hm_node = 'I' + str(hm)
            for rm in recommeded:
                rm_name = 'I' + str(rm)
                try:
                    paths = nx.all_shortest_paths(subgraph, source=hm_node, target=rm_name)
                    max = -1
                    max_path = []
                    for p in paths:
                        if len(p) > 5:
                            continue
                        path_embed = pd.Series(np.zeros(shape=(300,)))
                        uflag = False
                        count = 0
                        buff = [None, None]
                        for prop in p:
                            try:
                                title = titles[int(prop[1:])]
                                path_embed = path_embed + entity_embeds.loc[title]
                                buff[0] = title
                            except (ValueError, KeyError):
                                path_embed = path_embed + entity_embeds.loc[prop]
                                buff[1] = prop
                            count = count + 1

                            if count == 2:
                                count = 1
                                try:
                                    edge = relations[(buff[0], buff[1])]
                                except KeyError:
                                    try:
                                        edge = self.prop_set[self.prop_set[self.prop_set.columns[-1]] == buff[1]][
                                            "prop"].unique()[0]
                                    except AttributeError:
                                        edge = str(self.prop_set[self.prop_set[self.prop_set.columns[-1]] == buff[1]][
                                                       "prop"])
                                    except IndexError:
                                        uflag = True
                                        break

                                path_embed = path_embed + relation_embeds.loc[edge]

                        # calculate similarity and check if is it higher
                        score = cosine_similarity([user_embed.to_numpy()], [path_embed.to_numpy()])[0][0]
                        if score > max and not uflag:
                            max = score
                            max_path = p

                    sem_path_dist = sem_path_dist.append(
                        {'historic': hm, 'recommended': rm, 'path_s': max_path, 'path_v': max}, ignore_index=True)

                except (nx.exception.NetworkXNoPath, ValueError):
                    sem_path_dist = sem_path_dist.append(
                        {'historic': hm, 'recommended': rm, 'path_s': [], 'path_v': -1},
                        ignore_index=True)

        # choose path with highest mean similarity of sem_path_dist
        sem_path_dist_m = sem_path_dist.pivot(index='historic', columns='recommended', values='path_v')
        max_paths = sem_path_dist_m.max().to_frame().T

        hist_items = {}
        nodes = {}
        hist_lists = []
        prop_lists = []
        for rec in recommeded:
            print("\nPaths for the Recommended Item: " + str(rec))
            file.write("\nPaths for the Recommended Item: " + str(rec) + "\n")
            try:
                max_value = max_paths[rec][0]
                if max_value == -1:
                    raise KeyError
                origin = sem_path_dist_m[sem_path_dist_m[rec] == max_value].index[0]
                row = sem_path_dist[(sem_path_dist['historic'] == origin) & (sem_path_dist['recommended'] == rec)]
                path = list(row["path_s"])[0]

                origin = ""
                hist_names = hist_props.loc[hist_props.index.isin([int(x[1:]) for x in path[:-1][0::2]])][
                    self.prop_cols[1]].unique()
                hist_lists.append([int(x[1:]) for x in path[:-1][0::2]])
                for i in hist_names:
                    origin = origin + "\"" + i + "\"; "
                    hist_items = self.__add_dict(hist_items, i)
                origin = origin[:-2]

                path_props = [x for x in path[:-1][1::2]]
                prop_lists.append(path_props)
                path_sentence = " nodes: "
                for n in path_props:
                    path_sentence = path_sentence + "\"" + n + "\" "
                    nodes = self.__add_dict(nodes, n)
                try:
                    rec_name = self.prop_set.loc[rec][self.prop_cols[1]].unique()[0]
                except AttributeError:
                    rec_name = self.prop_set.loc[rec][self.prop_cols[1]]
                destination = "destination: \"" + rec_name + "\""
            except KeyError:
                origin = ""
                path_sentence = ""
                try:
                    rec_name = self.prop_set.loc[rec][self.prop_cols[1]].unique()[0]
                except AttributeError:
                    rec_name = self.prop_set.loc[rec][self.prop_cols[1]]
                destination = "destination: \"" + rec_name + "\""
                hist_lists.append([])
                prop_lists.append([])

            print(origin + path_sentence + destination)
            file.write(origin + path_sentence + destination)

        lir = eval.lir_metric(0.3, user, hist_lists, self.train_set)
        sep = eval.sep_metric(0.3, prop_lists, self.prop_set, memo_sep)
        etd = eval.etd_metric(list(nodes.keys()), len(recommeded), len(self.prop_set['obj'].unique()))
        return hist_items, nodes, (lir, sep, etd)

    def __webmedia_embedding(self, model_name: str, recommeded: list, historic: list, user: int,
                             file: _io.TextIOWrapper, memo_sep: dict):
        hist_props = self.prop_set.loc[historic]
        model_path = self.train_file.split("/")
        model_path = '/'.join(model_path[:-3])
        model_path = model_path + "/models/" + model_name + "_webmedia_model"
        triples = self.prop_set.to_numpy()
        tf = TriplesFactory.from_labeled_triples(triples)

        model = torch.load(model_path + "/trained_model.pkl")
        entity_embeds = pd.DataFrame(model.entity_representations[0](indices=None).detach().numpy()).rename(
            tf.entity_id_to_label)
        relation_embeds = pd.DataFrame(model.relation_representations[0](indices=None).detach().numpy()).rename(
            tf.relation_id_to_label)

        # create user embeding as pooling of entity
        user_embed = pd.Series(np.zeros(shape=(model.entity_representations[0].embedding_dim,)))
        for i in historic:
            try:
                title = self.prop_set.loc[i][self.prop_set.columns[0]].unique()[0]
            except AttributeError:
                title = str(self.prop_set.loc[i][self.prop_set.columns[0]])

            item_embed = entity_embeds.loc[title]
            user_embed = user_embed + item_embed

        sem_path_dist = pd.DataFrame(columns=['historic', 'recommended', 'path_s', 'path_v'])
        historic_codes = ['I' + str(i) for i in historic]
        recommeded_codes = ['I' + str(i) for i in recommeded]
        historic_props = list(set(self.prop_set.loc[self.prop_set.index.isin(historic)]['obj']))

        subgraph = self.graph.subgraph(historic_codes + recommeded_codes + historic_props)
        titles = self.prop_set[self.prop_cols[1]].to_dict()
        relations = self.prop_set.set_index([self.prop_cols[1], self.prop_cols[-1]]).to_dict()['prop']
        # obtain paths from historic item to recommended
        for hm in historic:
            hm_node = 'I' + str(hm)
            for rm in recommeded:
                rm_name = 'I' + str(rm)
                try:
                    paths = nx.all_shortest_paths(subgraph, source=hm_node, target=rm_name)
                    max = -1
                    max_path = []
                    for p in paths:
                        if len(p) > 5:
                            continue
                        path_embed = pd.Series(np.zeros(shape=(model.entity_representations[0].embedding_dim,)))
                        uflag = False
                        count = 0
                        buff = [None, None]
                        for prop in p:
                            try:
                                title = titles[int(prop[1:])]
                                path_embed = path_embed + entity_embeds.loc[title]
                                buff[0] = title
                            except (ValueError, KeyError):
                                path_embed = path_embed + entity_embeds.loc[prop]
                                buff[1] = prop
                            count = count + 1

                            if count == 2:
                                count = 1
                                try:
                                    edge = relations[(buff[0], buff[1])]
                                except KeyError:
                                    try:
                                        edge = self.prop_set[self.prop_set[self.prop_set.columns[-1]] == buff[1]][
                                            "prop"].unique()[0]
                                    except AttributeError:
                                        edge = str(self.prop_set[self.prop_set[self.prop_set.columns[-1]] == buff[1]][
                                                       "prop"])
                                    except IndexError:
                                        uflag = True
                                        break

                                path_embed = path_embed + relation_embeds.loc[edge]

                        # calculate similarity and check if is it higher
                        score = cosine_similarity([user_embed.to_numpy()], [path_embed.to_numpy()])[0][0]
                        if score > max and not uflag:
                            max = score
                            max_path = p

                    sem_path_dist = sem_path_dist.append(
                        {'historic': hm, 'recommended': rm, 'path_s': max_path, 'path_v': max}, ignore_index=True)

                except (nx.exception.NetworkXNoPath, ValueError):
                    sem_path_dist = sem_path_dist.append(
                        {'historic': hm, 'recommended': rm, 'path_s': [], 'path_v': -1},
                        ignore_index=True)

        # choose path with highest mean similarity of sem_path_dist
        sem_path_dist_m = sem_path_dist.pivot(index='historic', columns='recommended', values='path_v')
        max_paths = sem_path_dist_m.max().to_frame().T

        hist_items = {}
        nodes = {}
        hist_lists = []
        prop_lists = []
        for rec in recommeded:
            print("\nPaths for the Recommended Item: " + str(rec))
            file.write("\nPaths for the Recommended Item: " + str(rec) + "\n")
            try:
                max_value = max_paths[rec][0]
                if max_value == -1:
                    raise KeyError
                origin = sem_path_dist_m[sem_path_dist_m[rec] == max_value].index[0]
                row = sem_path_dist[(sem_path_dist['historic'] == origin) & (sem_path_dist['recommended'] == rec)]
                path = list(row["path_s"])[0]

                origin = ""
                hist_names = hist_props.loc[hist_props.index.isin([int(x[1:]) for x in path[:-1][0::2]])][
                    self.prop_cols[1]].unique()
                hist_lists.append([int(x[1:]) for x in path[:-1][0::2]])
                for i in hist_names:
                    origin = origin + "\"" + i + "\"; "
                    hist_items = self.__add_dict(hist_items, i)
                origin = origin[:-2]

                path_props = [x for x in path[:-1][1::2]]
                prop_lists.append(path_props)
                path_sentence = " nodes: "
                for n in path_props:
                    path_sentence = path_sentence + "\"" + n + "\" "
                    nodes = self.__add_dict(nodes, n)
                try:
                    rec_name = self.prop_set.loc[rec][self.prop_cols[1]].unique()[0]
                except AttributeError:
                    rec_name = self.prop_set.loc[rec][self.prop_cols[1]]
                destination = "destination: \"" + rec_name + "\""
            except KeyError:
                origin = ""
                path_sentence = ""
                try:
                    rec_name = self.prop_set.loc[rec][self.prop_cols[1]].unique()[0]
                except AttributeError:
                    rec_name = self.prop_set.loc[rec][self.prop_cols[1]]
                destination = "destination: \"" + rec_name + "\""
                hist_lists.append([])
                prop_lists.append([])

            print(origin + path_sentence + destination)
            file.write(origin + path_sentence + destination)

        lir = eval.lir_metric(0.3, user, hist_lists, self.train_set)
        sep = eval.sep_metric(0.3, prop_lists, self.prop_set, memo_sep)
        etd = eval.etd_metric(list(nodes.keys()), len(recommeded), len(self.prop_set['obj'].unique()))
        return hist_items, nodes, (lir, sep, etd)

    def __llm_paths(self, model_name: str, recommeded: list, historic: list, user: int,
                    file: _io.TextIOWrapper, memo_sep: dict):
        sem_path_dist = pd.DataFrame(columns=['historic', 'recommended', 'path_s'])

        key = None
        if model_name.startswith("gpt"):
            key = os.getenv("OPEN_AI_KEY")

        historic_codes = ['I' + str(i) for i in historic]
        recommended_codes = ['I' + str(i) for i in recommeded]
        historic_props = list(set(self.prop_set.loc[self.prop_set.index.isin(historic)]['obj']))
        subgraph = self.graph.subgraph(historic_codes + recommended_codes + historic_props)

        titles = self.prop_set[self.prop_cols[1]].to_dict()
        relations = self.prop_set.set_index([self.prop_cols[1], self.prop_cols[-1]]).to_dict()['prop']

        # obtain paths from historic item to recommended
        for hm in historic:
            hm_node = 'I' + str(hm)
            for rm in recommeded:
                rm_name = 'I' + str(rm)
                try:
                    paths = list(nx.all_shortest_paths(subgraph, source=hm_node, target=rm_name))
                    for p in paths:
                        if len(p) > 3 and not("last-fm" in self.prop_path):
                            continue
                        str_path = []
                        p_items = [x for x in p[:][0::2]]
                        for i in range(0, len(p) - 1):
                            elem = p[i]
                            if elem in p_items:
                                item_name = titles[int(elem[1:])]
                                str_path.append(item_name)
                                # str_path.append(relations[item_name, p[i+1]])
                            else:
                                str_path.append(elem)
                                # str_path.append(relations[titles[int(p[i+1][1:])], elem])

                        rec_title_name = titles[int(p[-1][1:])]
                        str_path.append(rec_title_name)
                        sem_path_dist = sem_path_dist.append(
                            {'historic': hm, 'recommended': rm, 'path_s': str_path}, ignore_index=True)

                except (nx.exception.NetworkXNoPath, ValueError):
                    sem_path_dist = sem_path_dist.append(
                        {'historic': hm, 'recommended': rm, 'path_s': []},
                        ignore_index=True)

        sem_path_dist = sem_path_dist.set_index("recommended")
        prompt = self.llm_prompt(model_name, historic, recommeded, titles, sem_path_dist)
        new_prompt = ""

        print("\nLLM " + model_name + " Prompt:")
        print(prompt)
        file.write("LLM " + model_name + " Prompt:")
        file.write(prompt)

        if model_name.startswith("gpt") or model_name == "llama3-70b-8192":
            try:
                response = self.invoke_model(model_name, prompt, key)
            except RateLimitError:
                signature = inspect.signature(self.llm_prompt)
                new_expl_count = signature.parameters["expl_count"].default
                new_prompt = self.llm_prompt(model_name,
                                             historic,
                                             recommeded,
                                             titles,
                                             sem_path_dist,
                                             expl_count=new_expl_count - 10)
                print("\nLLM New " + model_name + " Prompt:")
                file.write("\n\nLLM New " + model_name + " Response: ")
                print(new_prompt)
                file.write("\n" + new_prompt + "\n")
                response = self.invoke_model(model_name, new_prompt, key)
        else:
            lines = []
            while True:
                try:
                    line = input()
                    if line == "END":
                        break
                    lines.append(line)
                except EOFError:
                    break

            response = "\n".join(lines)

        print("\nLLM " + model_name + " Response: ")
        file.write("\n\nLLM " + model_name + " Response: ")
        print(response)
        file.write("\n" + response + "\n")

        if model_name.startswith("gpt"):
            response_arr = response.split("\n")
        else:
            tries = 5
            response_arr = re.findall(r'^.*\|.*->.*$', response, re.MULTILINE)
            while tries > 0 and len(response_arr) == 0:
                tries = tries - 1
                if new_prompt == "":
                    response = self.invoke_model(model_name, prompt, key)
                else:
                    response = self.invoke_model(model_name, new_prompt, key)
                response_arr = re.findall(r'^.*\|.*->.*$', response, re.MULTILINE)
            if tries == 0 and len(response_arr) == 0:
                raise Exception("Could not parse LLLM output")

        resp_paths_dict = {}
        for i in range(0, len(response_arr)):
            split_reponse = response_arr[i].split("|")
            title = split_reponse[0].strip()
            try:
                title_id = list(titles.keys())[list(titles.values()).index(title)]
            except ValueError:
                try:
                    title_id = list(titles.keys())[list(map(lambda x: x.lower(), list(titles.values()))).index(title.lower())]
                except ValueError:
                    try:
                        title_id = list(titles.keys())[list(map(lambda x: x.lower(), list(titles.values()))).index(split_reponse[-1].split("->")[-1].strip().lower())]
                    except ValueError:
                        for ind, (key, value) in enumerate(titles.items()):
                            if str(value).lower() == title.lower():
                                break
                        title_id = key
            resp_paths_dict[title_id] = split_reponse[1]

        hist_items = {}
        nodes = {}
        hist_lists = []
        prop_lists = []
        for rec in recommeded:
            print("\nPaths for the Recommended Item: " + str(rec))
            file.write("\nPaths for the Recommended Item: " + str(rec) + "\n")

            try:
                key1 = rec
                keys = [k for k, v in titles.items() if v == titles[rec]]
                keys_p = [key for key in keys if resp_paths_dict.get(key) is not None]
                if len(keys_p) > 0:
                    if keys_p[0] != rec:
                        key1 = keys_p[0]
                path = resp_paths_dict[key1]
                path_split = path.split(" -> ")
                origin = ""

                hist_names = [str(x).strip() for x in path_split[:-1][0::2]]
                hist_ids = []
                for h_name in hist_names:
                    try:
                        hist_ids.append(list(titles.keys())[list(titles.values()).index(h_name)])
                    except ValueError:
                        rem_title = ''.join(letter for letter in h_name if letter.isalnum())
                        for ind, (key, value) in enumerate(titles.items()):
                            rem_value = ''.join(letter for letter in value if letter.isalnum())
                            if str(rem_value).lower() == rem_title.lower():
                                hist_ids.append(key)
                                break

                hist_lists.append(hist_ids)
                for i in hist_names:
                    origin = origin + "\"" + i + "\"; "
                    hist_items = self.__add_dict(hist_items, i)
                origin = origin[:-2]

                path_props = [str(x).strip() for x in path_split[:-1][1::2]]
                prop_lists.append(path_props)
                path_sentence = " nodes: "
                for n in path_props:
                    path_sentence = path_sentence + "\"" + n + "\" "
                    nodes = self.__add_dict(nodes, n)

                rec_name = str(path_split[-1]).strip()
                destination = "destination: \"" + rec_name + "\""

            except KeyError:
                origin = ""
                path_sentence = ""
                try:
                    rec_name = self.prop_set.loc[rec][self.prop_cols[1]].unique()[0]
                except AttributeError:
                    rec_name = self.prop_set.loc[rec][self.prop_cols[1]]
                destination = "destination: \"" + rec_name + "\""
                hist_lists.append([])
                prop_lists.append([])

            file.write(origin + path_sentence + destination + "\n")
            print(origin + path_sentence + destination)

        lir = eval.lir_metric(0.3, user, hist_lists, self.train_set)
        sep = eval.sep_metric(0.3, prop_lists, self.prop_set, memo_sep)
        etd = eval.etd_metric(list(nodes.keys()), len(recommeded), len(self.prop_set['obj'].unique()))
        return hist_items, nodes, (lir, sep, etd)

    def llm_prompt(self, model: str, historic: list, recommended: list, titles: dict, sem_path_dist: pd.DataFrame,
                   random_seed=64, hist_count=50, expl_count=40):

        prompt = ""
        if model.startswith("gpt") or model.startswith("llama"):
            prompt = prompt + "\nIn a recommender system, a user has interacted some items, chronologically, " \
                              "in descending order, the last items interacted were:\n"

            last_items = historic[:hist_count]
            for i in range(0, len(last_items)):
                prompt = prompt + "'" + titles[historic[i]] + "'\n"

            prompt = prompt + "The user has top-5 recommendations, and possible explanations paths. " \
                              "Explanation paths connect an interacted item is connected to a recommended item " \
                              "by attributes. Bellow are the user recommendations with the followed by " \
                              "enumerated explanation paths, the symbol '->' means the connection between " \
                              "an item and an attribute:\n"

            for i in range(0, len(recommended)):
                rec_item = int(recommended[i])
                prompt = prompt + "\nFor the recommended item '" + titles[rec_item] + "':\n"
                all_expl = sem_path_dist.loc[rec_item]
                # extract paths from last items
                try:
                    top_expl = all_expl[(all_expl["historic"].isin(last_items))][:expl_count].sample(frac=1, random_state=random_seed)
                except AttributeError:
                    top_expl = pd.DataFrame(all_expl.copy()).T

                # fill with other items until reach the expl_count
                if top_expl.shape[0] == 0 or all_expl.shape[0] < expl_count:
                    top_expl = all_expl.sample(frac=1, random_state=random_seed)[:expl_count]
                elif top_expl.shape[0] < expl_count:
                    if isinstance(top_expl, pd.Series):
                        rest = expl_count - 1
                        temp = all_expl.sample(frac=1, random_state=random_seed)[:rest]
                        top_expl = temp.append(top_expl, ignore_index=True)
                    else:
                        lgth = top_expl.shape[0]
                        rest = expl_count - lgth
                        top_expl = pd.concat(
                            [top_expl, all_expl[(~all_expl["historic"].isin(last_items))][:rest]]).sample(
                            frac=1, random_state=random_seed)
                
                if isinstance(top_expl, pd.Series):
                    top_expl = pd.DataFrame(top_expl.copy()).T

                c = 1
                for _, row in top_expl.iterrows():
                    prompt = prompt + str(c) + "."
                    for elem in row["path_s"]:
                        prompt = prompt + elem + " -> "
                    prompt = prompt[:-3] + "\n"
                    c = c + 1

            prompt = prompt + "\nPlease choose one explanation path for each recommendation " \
                              "considering the following criteria:\n"

            prompt = prompt + "1. Diversity of attributes: Each path is composed of attributes that connect" \
                              "an interacted item node with a recommended. Diversify these attributes for the chosen" \
                              " explanation paths set of all recommendations;\n"

            prompt = prompt + "2. Popularity of attributes: Each path is composed of attributes that connect" \
                              "an interacted item node with a recommended. Use popular attributes for the chosen" \
                              " explanation paths set of all recommendations;\n"

            prompt = prompt + "3. Recency of interacted items: Use explanation paths that connects recently interacted " \
                              "items with the recommended.\n\n"

            if model.startswith("gpt"):
                prompt = prompt + "Output exactly in the same format as bellow with only the chosen explanation for each " \
                                  "recommendation starting with the name " \
                                  "and then the explanation, separated with a the symbol '|'." \
                                  "The path's attributes are separated with '->'. An example of the exact format I want" \
                                  " you to output is:\n" \
                                  "Gangs of New York | Titanic -> Leonardo DiCaprio -> Gangs of New York\n" \
                                  "Gladiator | Erin Brockovich -> Academy Award for Best Director -> Gladiator\n" \
                                  "Tarzan | Ratatouille -> Walt Disney Pictures -> Tarzan\n" \
                                  "A Bug's Life | Ghostbusters -> adventure film -> A Bug's Life\n" \
                                  "War Horse | Band of Brothers -> Steven Spielberg -> War Horse"
            else:
                prompt = prompt + "The output format for every recommendation is: in one line, start with the name of " \
                                  "the recommendation, then add the symbol '|'  followed by the explanation path chosen." \
                                  "Do not create explanation paths, choose from the ones listed."

        return prompt

    def invoke_model(self, model_name: str, prompt: str, key=None):
        response_text = ""
        tries = 6
        if model_name.startswith("gpt"):
            openai.api_key = key
            while tries > 0:
                try:
                    response = openai.ChatCompletion.create(
                        model=model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,
                        seed=42
                    )

                    # extract the response
                    response_text = response.choices[0].message["content"]
                    break

                except (openai.error.InvalidRequestError, openai.error.AuthenticationError,
                        openai.error.PermissionError, openai.error.Timeout, openai.error.APIConnectionError,
                        openai.error.APIError) as e:
                    print(f"Invalid request: {e}")
                    tries = tries - 1
                    time.sleep(60 * 10)  # Wait before retrying
                except (openai.error.ServiceUnavailableError,
                        openai.error.RateLimitError) as e:
                    print(f"Service unavailable: {e}")
                    tries = tries - 1
                    time.sleep(60 * 10)  # Wait before retrying
                except Exception as e:
                    print(f"Error: {e}")
                    tries = tries - 1
                    time.sleep(60 * 10)  # Wait before retrying

        if model_name == "llama3-70b-8192":
            while tries > 0:
                try:
                    response = Groq().chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        temperature=0.6,
                        model="llama3-70b-8192",
                    )
                    response_text = response.choices[0].message.content
                    break
                except RateLimitError as e:
                    print(f"Error: {e}")
                    print("Waiting one minute")
                    time.sleep(60)
                    raise
                except Exception as e:
                    print(f"Error: {e}")
                    tries = tries - 1
                    time.sleep(60 * 10)  # Wait before retrying

        if tries == 0 and response_text == "":
            raise Exception("Tries exceeded")
        if response_text == "":
            raise Exception("LLM did not respond")

        return response_text

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

    def __sub_dict(self, d: dict, key) -> dict:
        """
        Function to increment one in the key
        :param d: dictionary
        :param key: key to increment value
        :return: new dictionary
        """
        try:
            d[key] = d[key] - 1
            if d[key] == 0:
                del d[key]
        except KeyError:
            return d

        return d
