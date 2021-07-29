from preprocessing import movielens_small_utils as ml_small
from preprocessing import lastfm_utils as fm
from evaluation_utils import statistical_relevance
from caserec.recommenders.item_recommendation.most_popular import MostPopular
from caserec.recommenders.item_recommendation.userknn import UserKNN
from caserec.recommenders.item_recommendation.bprmf import BprMF
from recommenders import NCF as ncf
from recommenders import EASE as ease
from recommenders import page_rank_recommender as pagerank
from recommenders.prop_reordering import PropReordering
from recommenders.path_reordering import PathReordering
from evaluation_utils import evaluate


def run_experiments_ml(fold: str, start_fold: int, end_fold: int, baseline: list, proposed: list, reorder=None):
    """
    Run experiments for the movie-lens 100k dataset for the quantity of folds passed by in the parameter n_folds.
    E.g. if 10 then will run for all folds if 6 it will run from fold 0 to 5, etc
    :param fold: path of the folds
    :param start_fold: folds to start evaluation to evaluate
    :param end_fold: fold to end evaluation
    :param baseline: binary list for evaluating the current 4 recsys (MostPop, BPR-MF, User-KNN, PageRank).
    :param proposed: binary list to run the proposed reorder by property and by path. Is currently running 5 reorders
    :param reorder: binary list to reorder recommender algorithms that had run already
    :return: folds evaluated by accuracy and diversity metrics
    """

    output_names = set([])

    # BASELINES
    for i in range(start_fold, end_fold + 1):
        train_file = fold + str(i) + "/train.dat"
        test_file = fold + str(i) + "/test.dat"

        # 1 - Most Popular Algorithm
        most_pop_output_file = fold + str(i) + "/outputs/mostpop.csv"
        if baseline[0]:
            MostPopular(train_file, test_file, sep=',', output_file=most_pop_output_file, rank_length=20).compute()
            evaluate("Most Pop Algorithm", most_pop_output_file, train_file, test_file)
            output_names.add(most_pop_output_file.split("/")[-1])
        if reorder is not None and reorder[0] == 1:
            output_names.add(most_pop_output_file.split("/")[-1])

        # 2 - BPR MF
        bprmf_output_file = fold + str(i) + "/outputs/bprmf.csv"
        if baseline[1]:
            BprMF(train_file, test_file, sep=',', output_file=bprmf_output_file, rank_length=20, factors=32,
                  random_seed=42).compute()
            evaluate("BPR-MF Algorithm", bprmf_output_file, train_file, test_file)
            output_names.add(bprmf_output_file.split("/")[-1])
        if reorder is not None and reorder[1] == 1:
            output_names.add(bprmf_output_file.split("/")[-1])

        # 3 - User KNN
        knn_output_file = fold + str(i) + "/outputs/userknn.csv"
        if baseline[2]:
            UserKNN(train_file, test_file, sep=",", output_file=knn_output_file, rank_length=20).compute()
            evaluate("User KNN Algorithm", knn_output_file, train_file, test_file)
            output_names.add(knn_output_file.split("/")[-1])
        if reorder is not None and reorder[2] == 1:
            output_names.add(knn_output_file.split("/")[-1])

        # 4 - Wikidata PageRank params: weights=[80, 0, 20]
        pr = pagerank.PageRankRecommnder(fold + str(i), "wikidata_page_rank8020.csv", 20,
                                         "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                         node_weighs=[0.8, 0, 0.2], prop_cols=['movieId', 'title', 'prop', 'obj'],
                                         cols_used=[0, 1, 2], col_names=['user_id', 'movie_id', 'feedback'])
        if baseline[3]:
            pr.run()
            evaluate("Wikidata Page Rank 80/20 Algorithm", pr.output_path, train_file, test_file)
            output_names.add(pr.output_path.split("/")[-1])
        if reorder is not None and reorder[3] == 1:
            output_names.add(pr.output_path.split("/")[-1])

    # REORDERS
    for i in range(start_fold, end_fold + 1):

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
                evaluate("Prop Reorder Algorithm reorder=10", prop_reord.output_path, train_file, test_file)

        if proposed[1]:
            # 7 - Prop reorder params: reorder=20
            for output_file in output_files:
                prop_reord = PropReordering(train_file, output_file,
                                            "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                            cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                            n_reorder=20,
                                            prop_cols=['movieId', 'title', 'prop', 'obj'], hybrid=True)
                prop_reord.reorder()
                evaluate("Prop Reorder Algorithm reorder=20", prop_reord.output_path, train_file, test_file)

        if proposed[2]:
            # 8 - Path reorder params: reorder=10, policy=last, p_items=0.1
            for output_file in output_files:
                path_reord = PathReordering(train_file, output_file,
                                            "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                            cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                            prop_cols=['movieId', 'title', 'prop', 'obj'], n_reorder=10, p_items=0.1,
                                            policy='last', hybrid=True)
                path_reord.reorder()
                evaluate("Reorder Path Algorithm p_items=0.1, policy=last, n_reorder=10", path_reord.output_path,
                         train_file, test_file)

        if proposed[3]:
            # 9 - Path reorder params: reorder=10, policy=random, p_items=0.1
            for output_file in output_files:
                path_reord = PathReordering(train_file, output_file,
                                            "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                            cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                            prop_cols=['movieId', 'title', 'prop', 'obj'], n_reorder=10, p_items=0.1,
                                            policy='random', hybrid=True)
                path_reord.reorder()
                evaluate("Reorder Path USER KNN Algorithm p_items=0.1, policy=random, n_reorder=10",
                         path_reord.output_path, train_file, test_file)

        if proposed[4]:
            # 10 - Path reorder params: reorder=10, policy=last, p_items=0.1, not hybrid
            for output_file in output_files:
                path_reord = PathReordering(train_file, output_file,
                                            "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                            cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                            prop_cols=['movieId', 'title', 'prop', 'obj'], n_reorder=20, p_items=0.1,
                                            policy='last', hybrid=True)
                path_reord.reorder()
                evaluate("Reorder Path Algorithm p_items=0.1, policy=last, n_reorder=10 not hybrid",
                         path_reord.output_path,
                         train_file, test_file)


def run_experiments_lastfm(fold: str, start_fold: int, end_fold: int, baseline: list, proposed: list, reorder=None):
    """
    Run experiments for the lastfm dataset for the quantity of folds passed by in the parameter n_folds.
    E.g. if 10 then will run for all folds if 6 it will run from fold 0 to 5, etc
    :param fold: path of the folds
    :param start_fold: folds to start evaluation to evaluate
    :param end_fold: fold to end evaluation
    :param baseline: binary list for evaluating the current 4 recsys (MostPop, BPR-MF, User-KNN, PageRank).
    :param proposed: binary list to run the proposed reorder by property and by path. Is currently running 5 reorders
    :param reorder: binary list to reorder recommender algorithms that had run already
    :return: folds evaluated by accuracy and diversity metrics
    """

    output_names = set([])

    # BASELINES
    for i in range(start_fold, end_fold + 1):
        train_file = fold + str(i) + "/train.dat"
        test_file = fold + str(i) + "/test.dat"

        # 1 - Most Popular Algorithm
        most_pop_output_file = fold + str(i) + "/outputs/mostpop.csv"
        if baseline[0]:
            MostPopular(train_file, test_file, sep=',', output_file=most_pop_output_file, rank_length=20).compute()
            evaluate("Most Pop Algorithm", most_pop_output_file, train_file, test_file)
            output_names.add(most_pop_output_file.split("/")[-1])
        if reorder is not None and reorder[0] == 1:
            output_names.add(most_pop_output_file.split("/")[-1])

        # 2 - BPR MF
        bprmf_output_file = fold + str(i) + "/outputs/bprmf.csv"
        if baseline[1]:
            BprMF(train_file, test_file, sep=',', output_file=bprmf_output_file, rank_length=20, factors=32,
                  random_seed=42).compute()
            evaluate("BPR-MF Algorithm", bprmf_output_file, train_file, test_file)
            output_names.add(bprmf_output_file.split("/")[-1])
        if reorder is not None and reorder[1] == 1:
            output_names.add(bprmf_output_file.split("/")[-1])

        # 3 - User KNN
        knn_output_file = fold + str(i) + "/outputs/userknn.csv"
        if baseline[2]:
            UserKNN(train_file, test_file, sep=",", output_file=knn_output_file, rank_length=20).compute()
            evaluate("User KNN Algorithm", knn_output_file, train_file, test_file)
            output_names.add(knn_output_file.split("/")[-1])
        if reorder is not None and reorder[2] == 1:
            output_names.add(knn_output_file.split("/")[-1])

        # 4 - Wikidata PageRank params: weights=[80, 0, 20]
        pr = pagerank.PageRankRecommnder(fold + str(i), "wikidata_page_rank8020.csv", 20,
                                         "./generated_files/wikidata/last-fm/props_artists_id.csv",
                                         node_weighs=[0.8, 0, 0.2], prop_cols=['id', 'artist', 'prop', 'obj'],
                                         cols_used=[0, 1, 2], col_names=['user_id', 'artist_id', 'feedback'])
        if baseline[3]:
            pr.run()
            evaluate("Wikidata Page Rank 80/20 Algorithm", pr.output_path, train_file, test_file)
            output_names.add(pr.output_path.split("/")[-1])
        if reorder is not None and reorder[3] == 1:
            output_names.add(pr.output_path.split("/")[-1])

    # REORDERS
    for i in range(start_fold, end_fold + 1):

        train_file = fold + str(i) + "/train.dat"
        test_file = fold + str(i) + "/test.dat"
        output_files = [fold + str(i) + "/outputs/" + name for name in output_names]

        if proposed[0]:
            # 6 - Prop reorder params: reorder=10
            for output_file in output_files:
                prop_reord = PropReordering(train_file, output_file,
                                            "./generated_files/wikidata/last-fm/props_artists_id.csv",
                                            cols_used=['user_id', 'artist_id', 'interaction'],
                                            n_reorder=10,
                                            prop_cols=['id', 'artist', 'prop', 'obj'], hybrid=True)
                prop_reord.reorder()
                evaluate("Prop Reorder Algorithm reorder=10", prop_reord.output_path, train_file, test_file)

        if proposed[1]:
            # 7 - Prop reorder params: reorder=20
            for output_file in output_files:
                prop_reord = PropReordering(train_file, output_file,
                                            "./generated_files/wikidata/last-fm/props_artists_id.csv",
                                            cols_used=['user_id', 'artist_id', 'interaction'],
                                            n_reorder=20,
                                            prop_cols=['id', 'artist', 'prop', 'obj'], hybrid=True)
                prop_reord.reorder()
                evaluate("Prop Reorder Algorithm reorder=20", prop_reord.output_path, train_file, test_file)

        if proposed[2]:
            # 8 - Path reorder params: reorder=10, policy=last, p_items=0.1
            for output_file in output_files:
                path_reord = PathReordering(train_file, output_file,
                                            "./generated_files/wikidata/last-fm/props_artists_id.csv",
                                            cols_used=['user_id', 'artist_id', 'interaction'],
                                            prop_cols=['id', 'artist', 'prop', 'obj'], n_reorder=10, p_items=0.1,
                                            policy='last', hybrid=True)
                path_reord.reorder()
                evaluate("Reorder Path Algorithm p_items=0.1, policy=last, n_reorder=10", path_reord.output_path,
                         train_file, test_file)

        if proposed[3]:
            # 9 - Path reorder params: reorder=10, policy=random, p_items=0.1
            for output_file in output_files:
                path_reord = PathReordering(train_file, output_file,
                                            "./generated_files/wikidata/last-fm/props_artists_id.csv",
                                            cols_used=['user_id', 'artist_id', 'interaction'],
                                            prop_cols=['id', 'artist', 'prop', 'obj'], n_reorder=10, p_items=0.1,
                                            policy='random', hybrid=True)
                path_reord.reorder()
                evaluate("Reorder Path USER KNN Algorithm p_items=0.1, policy=random, n_reorder=10",
                         path_reord.output_path, train_file, test_file)

        if proposed[4]:
            # 10 - Path reorder params: reorder=10, policy=last, p_items=0.1, not hybrid
            for output_file in output_files:
                path_reord = PathReordering(train_file, output_file,
                                            "./generated_files/wikidata/last-fm/props_artists_id.csv",
                                            cols_used=['user_id', 'artist_id', 'interaction'],
                                            prop_cols=['id', 'artist', 'prop', 'obj'], n_reorder=20, p_items=0.1,
                                            policy='last', hybrid=True)
                path_reord.reorder()
                evaluate("Reorder Path Algorithm p_items=0.1, policy=last, n_reorder=10 not hybrid",
                         path_reord.output_path,
                         train_file, test_file)


# fm.user_artist_filter_interaction(5, n_iter_flag=True)
# fm.cross_validation_lasfm(rs=42)

folds_path_ml = "./datasets/ml-latest-small/folds/"
folds_path_lastfm = "./datasets/hetrec2011-lastfm-2k/folds/"

ncf_rec = ncf.NCF(folds_path_ml + str(0), "ncf.csv", rank_size=20, factors=32, layers=[64, 32, 16, 8],
                  epochs=5, neg_smp_train=4, neg_smp_test=99, cols_used=[0, 1, 2],
                  col_names=['user_id', 'movie_id', 'feedback'],
                  model_path=folds_path_ml + str(0) + "/model.pt", batch_size=256, seed=42, model_disk='w')
ncf_rec.train()
ncf_rec.run()

evaluate("NCF",
         ncf_rec.output_path,
         "./datasets/ml-latest-small/folds/0/train.dat", "./datasets/ml-latest-small/folds/0/test.dat")

#ease = ease.EASE(folds_path_ml + str(0), "ease.csv", rank_size=20, cols_used=[0, 1, 2],
#                  col_names=['user_id', 'movie_id', 'feedback'], lambda_=500)
#ease.train()
#ease.run()
#evaluate("EASE", ease.output_path, "./datasets/ml-latest-small/folds/0/train.dat",
#         "./datasets/ml-latest-small/folds/0/test.dat")

# run_experiments_lastfm(folds_path_lastfm, 0, 9, [0, 0, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1])
# statistical_relevance("path[policy=last_items=01_reorder=10_hybrid]", "bprmf", folds_path_lastfm,
#                      ["MAP", "NDCG", "GINI", "ENTROPY", "COVERAGE"], save=False)
