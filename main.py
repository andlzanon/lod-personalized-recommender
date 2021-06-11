from preprocessing import movielens_small_utils as ml_small
from caserec.recommenders.item_recommendation.most_popular import MostPopular
from caserec.recommenders.item_recommendation.userknn import UserKNN
from caserec.recommenders.item_recommendation.bprmf import BprMF
from recommenders import page_rank_recommender as pagerank
from recommenders.prop_reordering import PropReordering
from recommenders.path_reordering import PathReordering
import utils


def run_experiments(fold: str, n_folds: int, baseline: list, proposed: list):
    output_names = set([])

    for i in range(n_folds + 1):
        train_file = fold + str(i) + "/train.dat"
        test_file = fold + str(i) + "/test.dat"

        # 1 - Most Popular Algorithm
        most_pop_output_file = fold + str(i) + "/outputs/mostpop.csv"
        if baseline[0]:
            MostPopular(train_file, test_file, sep=',', output_file=most_pop_output_file, rank_length=20).compute()
            utils.evaluate("Most Pop Algorithm", most_pop_output_file, test_file)
        output_names.add(most_pop_output_file.split("/")[-1])

        # 2 - BPR MF
        bprmf_output_file = fold + str(i) + "/outputs/bprmf.csv"
        if baseline[1]:
            BprMF(train_file, test_file, sep=',', output_file=bprmf_output_file, rank_length=20, random_seed=42).compute()
            utils.evaluate("BPR-MF Algorithm", bprmf_output_file, test_file)
        output_names.add(bprmf_output_file.split("/")[-1])

        # 3 - User KNN
        knn_output_file = fold + str(i) + "/outputs/userknn.csv"
        if baseline[2]:
            UserKNN(train_file, test_file, sep=",", output_file=knn_output_file, rank_length=20).compute()
            utils.evaluate("User KNN Algorithm", knn_output_file, test_file)
        output_names.add(knn_output_file.split("/")[-1])

        # 4 - PageRank params: weights=[80, 0, 20]
        pr = pagerank.PageRankRecommnder(fold + str(i), "wikidata_page_rank8020.csv", 20,
                                         "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                         cols_used=[0, 1, 2], col_names=['user_id', 'movie_id', 'feedback'])
        if baseline[3]:
            pr.run()
            utils.evaluate("Page Rank 80/20 Algorithm", pr.output_path, test_file)
        output_names.add(pr.output_path.split("/")[-1])

    for i in range(n_folds + 1):

        train_file = fold + str(i) + "/train.dat"
        test_file = fold + str(i) + "/test.dat"
        output_files = [fold + str(i) + "/outputs/" + name for name in output_names]

        if proposed[0]:
            # 6 - Prop reorder params: reorder=10
            for output_file in output_files:
                prop_reord = PropReordering(train_file, output_file,
                                              "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                              cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                              n_reorder=10,
                                              prop_cols=['movieId', 'title', 'prop', 'obj'], hybrid=True)
                prop_reord.reorder()
                utils.evaluate("Prop Reorder Algorithm reorder=10", prop_reord.output_path, test_file)

        if proposed[1]:
            # 7 - Prop reorder params: reorder=20
            for output_file in output_files:
                prop_reord = PropReordering(train_file, output_file,
                                              "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                              cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                              n_reorder=20,
                                              prop_cols=['movieId', 'title', 'prop', 'obj'], hybrid=True)
                prop_reord.reorder()
                utils.evaluate("Prop Reorder Algorithm reorder=20", prop_reord.output_path, test_file)

        if proposed[2]:
            # 8 - Path reorder params: reorder=10, policy=last, p_items=0.1
            for output_file in output_files:
                path_reord = PathReordering(train_file, output_file,
                                            "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                            cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                            prop_cols=['movieId', 'title', 'prop', 'obj'], n_reorder=10, p_items=0.1,
                                            policy='last', hybrid=True)
                path_reord.reorder()
                utils.evaluate("Reorder Path Algorithm p_items=0.1, policy=last, n_reorder=10", path_reord.output_path,
                               test_file)

        if proposed[3]:
            # 9 - Path reorder params: reorder=10, policy=random, p_items=0.1
            for output_file in output_files:
                path_reord = PathReordering(train_file, output_file,
                                            "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                            cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                            prop_cols=['movieId', 'title', 'prop', 'obj'], n_reorder=10, p_items=0.1,
                                            policy='random', hybrid=True)
                path_reord.reorder()
                utils.evaluate("Reorder Path USER KNN Algorithm p_items=0.1, policy=random, n_reorder=10",
                               path_reord.output_path, test_file)

        if proposed[4]:
            # 10 - Path reorder params: reorder=10, policy=last, p_items=0.1, not hybrid
            for output_file in output_files:
                path_reord = PathReordering(train_file, output_file,
                                            "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                            cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                            prop_cols=['movieId', 'title', 'prop', 'obj'], n_reorder=10, p_items=0.1,
                                            policy='last', hybrid=False)
                path_reord.reorder()
                utils.evaluate("Reorder Path Algorithm p_items=0.1, policy=last, n_reorder=10 not hybrid", path_reord.output_path,
                               test_file)

        if proposed[5]:
            # 11 - Prop reorder params: reorder=10, not hybrid
            for output_file in output_files:
                prop_reord = PropReordering(train_file, output_file,
                                              "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                              cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                              n_reorder=10,
                                              prop_cols=['movieId', 'title', 'prop', 'obj'], hybrid=False)
                prop_reord.reorder()
                utils.evaluate("Prop Reorder Algorithm reorder=10 not hybrid", prop_reord.output_path, test_file)

        if proposed[6]:
            # 10 - Path reorder params: reorder=10, policy=last, p_items=0.1, not hybrid
            for output_file in output_files:
                path_reord = PathReordering(train_file, output_file,
                                            "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                            cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                            prop_cols=['movieId', 'title', 'prop', 'obj'], n_reorder=20, p_items=0.1,
                                            policy='last', hybrid=False)
                path_reord.reorder()
                utils.evaluate("Reorder Path Algorithm p_items=0.1, policy=last, n_reorder=10 not hybrid", path_reord.output_path,
                               test_file)


# utils.cross_validation_ml_small(rs=42)
# ml_small.extract_wikidata_prop()
# utils.split_dataset_by_timestamp("./datasets/ml-latest-small/ratings.csv", 0.1,
#   "./datasets/ml-latest-small/folds/timed")
folds_path = "./datasets/ml-latest-small/folds/"
run_experiments(folds_path, 9, [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1])
