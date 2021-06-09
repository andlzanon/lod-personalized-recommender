from preprocessing import movielens_small_utils as ml_small
from caserec.recommenders.item_recommendation.most_popular import MostPopular
from caserec.recommenders.item_recommendation.userknn import UserKNN
from caserec.recommenders.item_recommendation.bprmf import BprMF
from recommenders import page_rank_recommender as pagerank
from recommenders.prop_reordering import PropReordering
from recommenders.path_reordering import PathReordering
import utils


def run_experiments(fold: str):
    train_file = fold + "/train.dat"
    test_file = fold + "/test.dat"
    output_files = []

    # 1 - Most Popular Algorithm
    most_pop_output_file = fold + "/outputs/mostpop.csv"
    MostPopular(train_file, test_file,  sep=',', output_file=most_pop_output_file, rank_length=20).compute()
    utils.evaluate("Most Pop Algorithm", most_pop_output_file, test_file)
    output_files.append(most_pop_output_file)

    # 2 - BPR MF
    bprmf_output_file = fold + "/outputs/bprmf.csv"
    BprMF(train_file, test_file, sep=',', output_file=bprmf_output_file, rank_length=20, random_seed=42).compute()
    utils.evaluate("BPR-MF Algorithm", bprmf_output_file, test_file)
    output_files.append(bprmf_output_file)

    # 3 - User KNN
    knn_output_file = fold + "/outputs/userknn.csv"
    UserKNN(train_file, test_file, sep=",", output_file=knn_output_file, rank_length=20).compute()
    utils.evaluate("User KNN Algorithm", knn_output_file, test_file)
    output_files.append(knn_output_file)

    # 4 - PageRank params: weights=[80, 0, 20]
    pr = pagerank.PageRankRecommnder(fold, "wikidata_page_rank8020.csv", 20,
                                     "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                     cols_used=[0, 1, 2], col_names=['user_id', 'movie_id', 'feedback'])
    pr.run()
    utils.evaluate("Page Rank 80/20 Algorithm", pr.output_path, test_file)
    output_files.append(pr.output_path)

    # 5 - PageRank params: weights=[40,20,20]
    pr2 = pagerank.PageRankRecommnder(fold, "wikidata_page_rank404020.csv", 20,
                                      "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                      cols_used=[0, 1, 2], col_names=['user_id', 'movie_id', 'feedback'],
                                      node_weighs=[0.4, 0.4, 0.2])
    pr2.run()
    utils.evaluate("Page Rank 40/20/20 Algorithm", pr2.output_path, test_file)
    output_files.append(pr2.output_path)

    # 6 - Prop reorder params: reorder=10
    for output_file in output_files:
        prop_reord10 = PropReordering(train_file, output_file,
                                      "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                      cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'], n_reorder=10,
                                      prop_cols=['movieId', 'title', 'prop', 'obj'], hybrid=True)
        prop_reord10.reorder()
        utils.evaluate("Prop Reorder Algorithm reorder=10", prop_reord10.output_path, test_file)

    # 7 - Prop reorder params: reorder=20
    for output_file in output_files:
        prop_reord20 = PropReordering(train_file, output_file,
                                      "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                      cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'], n_reorder=20,
                                      prop_cols=['movieId', 'title', 'prop', 'obj'], hybrid=True)
        prop_reord20.reorder()
        utils.evaluate("Prop Reorder Algorithm reorder=20", prop_reord20.output_path, test_file)

    # 8 - Path reorder params: reorder=10, policy=last, p_items=0.1
    for output_file in output_files:
        path_reord = PathReordering(train_file, output_file,
                                    "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                    cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                    prop_cols=['movieId', 'title', 'prop', 'obj'], n_reorder=10, p_items=0.1,
                                    policy='last', hybrid=True)
        path_reord.reorder()
        utils.evaluate("Reorder Path Algorithm p_items=0.1, policy=last, n_reorder=10", path_reord.output_path, test_file)

    # 9 - Path reorder params: reorder=10, policy=last, p_items=0.2
    for output_file in output_files:
        path_reord = PathReordering(train_file, output_file,
                                    "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                    cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                    prop_cols=['movieId', 'title', 'prop', 'obj'], n_reorder=10, p_items=0.2,
                                    policy='last', hybrid=True)
        path_reord.reorder()
        utils.evaluate("Reorder Path Algorithm p_items=0.2, policy=last, n_reorder=10", path_reord.output_path, test_file)

    # 10 - Path reorder params: reorder=10, policy=random, p_items=0.1
    for output_file in output_files:
        path_reord = PathReordering(train_file, output_file,
                                    "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                    cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                    prop_cols=['movieId', 'title', 'prop', 'obj'], n_reorder=10, p_items=0.1,
                                    policy='random', hybrid=True)
        path_reord.reorder()
        utils.evaluate("Reorder Path USER KNN Algorithm p_items=0.1, policy=random, n_reorder=10", path_reord.output_path, test_file)


# utils.cross_validation_ml_small(rs=42)
# ml_small.extract_wikidata_prop()
# utils.split_dataset_by_timestamp("./datasets/ml-latest-small/ratings.csv", 0.1,
#   "./datasets/ml-latest-small/folds/timed")

run_experiments("./datasets/ml-latest-small/folds/0")
run_experiments("./datasets/ml-latest-small/folds/timed")