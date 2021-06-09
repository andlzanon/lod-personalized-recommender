import pandas as pd
from recommenders.lod_reordering import LODPersonalizedReordering


class PropReordering(LODPersonalizedReordering):
    def __init__(self, train_file: str, output_rec_file: str, prop_path: str, prop_cols: list, cols_used: list, n_reorder: int,
                 hybrid=False, n_sentences=3):
        """
        Path Reordering class: this algorithm will reorder the output of other recommendation algorithm based on the
        best path from an historic item and a recommended one. The best paths are extracted based on the value for each
        object of the LOD with the semantic profile
        :param train_file: train file in which the recommendations of where computed
        :param output_rec_file: output file of the recommendation algorithm
        :param prop_path: path to the properties on dbpedia or wikidata
        :param prop_cols: columns of the property set
        :param cols_used: columns used from the test and train set
        :param hybrid: if the reorder of the recommendations should [True] or not consider the score from the recommender
        :param n_sentences: number of paths to generate the sentence of explanation
        """

        self.output_name = 'prop[reorder=' + str(n_reorder) + "]"

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
        reorder = pd.DataFrame()

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

            # new items interacted based on policy and percentage
            sem_dist = self.__semantic_mean(u, user_semantic_profile, items_recommended)

            reorder = pd.concat([reorder, sem_dist], ignore_index=True)

        reorder.to_csv(self.output_path, mode='w', header=False, index=False)

    def __semantic_mean(self, user: int, semantic_profile: dict, recommended: list) -> pd.DataFrame:
        """
        Returns the mean of the properties of the item considering the semantic profile
        :param user: user to recommend item
        :param semantic_profile: semantic profile of the user
        :param recommended: recommended items
        :return: DataFrame with user_id, item_id and score
        """
        recommended_props = self.prop_set.loc[self.prop_set.index.isin(recommended)].copy()
        recommended_props['obj_score'] = list(map(semantic_profile.get, recommended_props['obj']))
        recommended_props['obj_score'] = recommended_props['obj_score'].fillna(0)

        rec_score = {}

        for i in recommended_props.index.unique():
            score = recommended_props.loc[i]['obj_score'].mean()
            rec_score[i] = score

        if self.hybrid:
            output_rec = self.output_rec_set.loc[user].set_index('item_id')
            for key in rec_score.keys():
                rec_score[key] = rec_score[key] * float(output_rec.loc[key])

        rec_score = dict(sorted(rec_score.items(), key=lambda item: item[1], reverse=True))

        reorder_list = []
        for key in rec_score.keys():
            reorder_list.append({'user_id': int(user), 'item_id': int(key), 'score': rec_score[key]})

        return pd.DataFrame(reorder_list)