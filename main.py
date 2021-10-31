from preprocessing import movielens_small_utils as ml_small
from preprocessing import lastfm_utils as fm
from evaluation_utils import statistical_relevance
from caserec.recommenders.item_recommendation.most_popular import MostPopular
from caserec.recommenders.item_recommendation.userknn import UserKNN
from caserec.recommenders.item_recommendation.bprmf import BprMF
from recommenders import NCF as ncf
from recommenders import EASE as ease
from recommenders import page_rank_recommender as pagerank
from recommenders.path_reordering import PathReordering
from evaluation_utils import evaluate
import argparse


def run_experiments_ml(fold: str, start_fold: int, end_fold: int, baselines: list, reorders=None, n_reorder=10,
                       p_items=0.1, policy='last'):
    """
    Run experiments for the movie-lens 100k dataset for the quantity of folds passed by in the parameter n_folds.
    E.g. if 9 then will run for all folds if 6 it will run from fold 0 to 5, etc
    :param fold: path of the folds
    :param start_fold: folds to start evaluation to evaluate
    :param end_fold: fold to end evaluation
    :param baselines: list for evaluating the current 6 recsys (MostPop, BPRMF, UserKNN, PageRank, NCF and EASE).
    :param reorders: list to reorder recommender algorithms that had run already
    :param n_reorder: quantity of items to reorder
    :param p_items: percentage of items from historic to build semantic profile
    :param policy: policy to choose items
    :return: folds evaluated by accuracy and diversity metrics
    """

    output_names = set([])

    # BASELINES
    for i in range(start_fold, end_fold + 1):
        train_file = fold + str(i) + "/train.dat"
        test_file = fold + str(i) + "/test.dat"

        # 1 - Most Popular Algorithm
        most_pop_output_file = fold + str(i) + "/outputs/mostpop.csv"
        if 'MostPop' in baselines:
            MostPopular(train_file, test_file, sep=',', output_file=most_pop_output_file, rank_length=20).compute()
            evaluate("Most Pop Algorithm", most_pop_output_file, train_file, test_file)
        if reorders is not None and 'MostPop' in reorders:
            output_names.add(most_pop_output_file.split("/")[-1])

        # 2 - BPR MF
        bprmf_output_file = fold + str(i) + "/outputs/bprmf.csv"
        if 'BPRMF' in baselines:
            BprMF(train_file, test_file, sep=',', output_file=bprmf_output_file, rank_length=20, factors=32,
                  random_seed=42).compute()
            evaluate("BPR-MF Algorithm", bprmf_output_file, train_file, test_file)
        if reorders is not None and 'BPRMF' in reorders:
            output_names.add(bprmf_output_file.split("/")[-1])

        # 3 - User KNN
        knn_output_file = fold + str(i) + "/outputs/userknn.csv"
        if 'UserKNN' in baselines:
            UserKNN(train_file, test_file, sep=",", output_file=knn_output_file, rank_length=20).compute()
            evaluate("User KNN Algorithm", knn_output_file, train_file, test_file)
        if reorders is not None and 'UserKNN' in reorders:
            output_names.add(knn_output_file.split("/")[-1])

        # 4 - Wikidata PageRank params: weights=[80, 0, 20]
        pr = pagerank.PageRankRecommnder(fold + str(i), "wikidata_page_rank8020.csv", 20,
                                         "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                         node_weighs=[0.8, 0, 0.2], prop_cols=['movieId', 'title', 'prop', 'obj'],
                                         cols_used=[0, 1, 2], col_names=['user_id', 'movie_id', 'feedback'])
        if 'PageRank' in baselines:
            pr.run()
            evaluate("Wikidata Page Rank 80/20 Algorithm", pr.output_path, train_file, test_file)
        if reorders is not None and 'PageRank' in reorders:
            output_names.add(pr.output_path.split("/")[-1])

        # 5 - Neural Collaborative Filtering
        ncf_rec = ncf.NCF(fold + str(i), "ncf.csv", rank_size=20, factors=32, layers=[64, 32, 16, 8],
                          epochs=10, neg_smp_train=4, neg_smp_test=100, cols_used=[0, 1, 2],
                          col_names=['user_id', 'movie_id', 'feedback'],
                          model_path=folds_path_ml + str(0) + "/model.pt", batch_size=256, seed=42, model_disk='w')
        if 'NCF' in baselines:
            ncf_rec.train()
            ncf_rec.run()
            evaluate("Neural Collaborative Filtering", ncf_rec.output_path, train_file, test_file)
        if reorders is not None and 'NCF' in reorders:
            output_names.add(ncf_rec.output_path.split("/")[-1])

        # 6 - EASE Algorithm
        ease_rec = ease.EASE(fold + str(i), "ease.csv", rank_size=20, cols_used=[0, 1, 2],
                             col_names=['user_id', 'movie_id', 'feedback'], lambda_=500)
        if 'EASE' in baselines:
            ease_rec.train()
            ease_rec.run()
            evaluate("EASE", ease_rec.output_path, train_file, test_file)
        if reorders is not None and 'EASE' in reorders:
            output_names.add(ease_rec.output_path.split("/")[-1])

    # REORDERS
    for i in range(start_fold, end_fold + 1):

        train_file = fold + str(i) + "/train.dat"
        test_file = fold + str(i) + "/test.dat"
        output_files = [fold + str(i) + "/outputs/" + name for name in output_names]

        # Path reorder
        for output_file in output_files:
            path_reord = PathReordering(train_file, output_file,
                                        "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                        cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                        prop_cols=['movieId', 'title', 'prop', 'obj'], n_reorder=n_reorder,
                                        p_items=p_items, policy=policy, hybrid=True)
            path_reord.reorder()
            evaluate("Reorder Path Algorithm p_items=" + str(p_items) + "policy=" + str(policy) + "n_reorder=" + str(
                n_reorder),
                     path_reord.output_path, train_file, test_file)


def run_experiments_lastfm(fold: str, start_fold: int, end_fold: int, baselines: list, reorders=None, n_reorder=10,
                           p_items=0.1, policy='last'):
    """
    Run experiments for the lastfm dataset for the quantity of folds passed by in the parameter n_folds.
    E.g. if 10 then will run for all folds if 6 it will run from fold 0 to 5, etc
    :param fold: path of the folds
    :param start_fold: folds to start evaluation to evaluate
    :param end_fold: fold to end evaluation
    :param baselines: list for evaluating the current 6 recsys (MostPop, BPRMF, UserKNN, PageRank, NCF and EASE).
    :param reorders: list to reorder recommender algorithms that had run already
    :param n_reorder: quantity of items to reorder
    :param p_items: percentage of items from historic to build semantic profile
    :param policy: policy to choose items
    :return: folds evaluated by accuracy and diversity metrics
    """

    output_names = set([])

    # BASELINES
    for i in range(start_fold, end_fold + 1):
        train_file = fold + str(i) + "/train.dat"
        test_file = fold + str(i) + "/test.dat"

        # 1 - Most Popular Algorithm
        most_pop_output_file = fold + str(i) + "/outputs/mostpop.csv"
        if 'MostPop' in baselines:
            MostPopular(train_file, test_file, sep=',', output_file=most_pop_output_file, rank_length=20).compute()
            evaluate("Most Pop Algorithm", most_pop_output_file, train_file, test_file)
        if reorders is not None and 'MostPop' in reorders:
            output_names.add(most_pop_output_file.split("/")[-1])

        # 2 - BPR MF
        bprmf_output_file = fold + str(i) + "/outputs/bprmf.csv"
        if 'BPRMF' in baselines:
            BprMF(train_file, test_file, sep=',', output_file=bprmf_output_file, rank_length=20, factors=32,
                  random_seed=42).compute()
            evaluate("BPR-MF Algorithm", bprmf_output_file, train_file, test_file)
        if reorders is not None and 'BPRMF' in reorders:
            output_names.add(bprmf_output_file.split("/")[-1])

        # 3 - User KNN
        knn_output_file = fold + str(i) + "/outputs/userknn.csv"
        if 'UserKNN' in baselines:
            UserKNN(train_file, test_file, sep=",", output_file=knn_output_file, rank_length=20).compute()
            evaluate("User KNN Algorithm", knn_output_file, train_file, test_file)
        if reorders is not None and 'UserKNN' in reorders:
            output_names.add(knn_output_file.split("/")[-1])

        # 4 - Wikidata PageRank params: weights=[80, 0, 20]
        pr = pagerank.PageRankRecommnder(fold + str(i), "wikidata_page_rank8020.csv", 20,
                                         "./generated_files/wikidata/last-fm/props_artists_id.csv",
                                         node_weighs=[0.8, 0, 0.2], prop_cols=['id', 'artist', 'prop', 'obj'],
                                         cols_used=[0, 1, 2], col_names=['user_id', 'artist_id', 'feedback'])
        if 'PageRank' in baselines:
            pr.run()
            evaluate("Wikidata Page Rank 80/20 Algorithm", pr.output_path, train_file, test_file)
        if reorders is not None and 'PageRank' in reorders:
            output_names.add(pr.output_path.split("/")[-1])

        # 5 - Neural Collaborative Filtering
        ncf_rec = ncf.NCF(fold + str(i), "ncf.csv", rank_size=20, factors=32, layers=[64, 32, 16, 8],
                          epochs=10, neg_smp_train=4, neg_smp_test=100, cols_used=[0, 1, 2],
                          col_names=['user_id', 'artist_id', 'feedback'],
                          model_path=folds_path_ml + str(0) + "/model.pt", batch_size=256, seed=42, model_disk='w')
        if 'NCF' in baselines:
            ncf_rec.train()
            ncf_rec.run()
            evaluate("Neural Collaborative Filtering", ncf_rec.output_path, train_file, test_file)
        if reorders is not None and 'NCF' in reorders:
            output_names.add(ncf_rec.output_path.split("/")[-1])

        # 6 - EASE Algorithm
        ease_rec = ease.EASE(fold + str(i), "ease.csv", rank_size=20, cols_used=[0, 1, 2],
                             col_names=['user_id', 'artist_id', 'feedback'], lambda_=500)
        if 'EASE' in baselines:
            ease_rec.train()
            ease_rec.run()
            evaluate("EASE", ease_rec.output_path, train_file, test_file)
        if reorders is not None and 'EASE' in reorders:
            output_names.add(ease_rec.output_path.split("/")[-1])

    # REORDERS
    for i in range(start_fold, end_fold + 1):

        train_file = fold + str(i) + "/train.dat"
        test_file = fold + str(i) + "/test.dat"
        output_files = [fold + str(i) + "/outputs/" + name for name in output_names]

        # Path reorder
        for output_file in output_files:
            path_reord = PathReordering(train_file, output_file,
                                        "./generated_files/wikidata/last-fm/props_artists_id.csv",
                                        cols_used=['user_id', 'artist_id', 'interaction'],
                                        prop_cols=['id', 'artist', 'prop', 'obj'], n_reorder=n_reorder, p_items=p_items,
                                        policy=policy, hybrid=True)
            path_reord.reorder()
            evaluate("Reorder Path Algorithm p_items=" + str(p_items) + "policy=" + str(policy) + "n_reorder=" + str(
                n_reorder),
                     path_reord.output_path, train_file, test_file)


def run_explanations_ml(fold: str, n_fold: int, reorders=None, n_reorder=10, p_items=0.1, policy='last', h_min=0,
                        h_max=20, max_users=1):
    """
    Run experiments for the movie-lens 100k dataset for the quantity of folds passed by in the parameter n_folds.
    E.g. if 9 then will run for all folds if 6 it will run from fold 0 to 5, etc
    :param fold: path of the folds
    :param n_fold: number of the fold to generate explanations to users
    :param reorders: list to reorder recommender algorithms that had run already
    :param n_reorder: quantity of items to reorder
    :param p_items: percentage of items from historic to build semantic profile
    :param policy: policy to choose items
    :param: h_min: minimum number of users' historic items to generate the recommendations and explanations to, if a
        user has a smaller number of interacted items than this parameter the algorithm will not generate explanations
    :param: h_max: maximum number of users' historic items to generate the recommendations and explanations to, if a
        user has a bigger number of interacted items than this parameter the algorithm will not generate explanations
    :param: max_users: maximum number of user to generate explanations to, when the program reaches max_number it stops
    :return: users are displayed on console with interacted items, recommended items, semantic profile, reordered items
        and explanation paths for each recommended item
    """

    output_names = set([])

    # BASELINES

    # 1 - Most Popular Algorithm
    most_pop_output_file = fold + str(n_fold) + "/outputs/mostpop.csv"
    if reorders is not None and 'MostPop' in reorders:
        output_names.add(most_pop_output_file.split("/")[-1])

    # 2 - BPR MF
    bprmf_output_file = fold + str(n_fold) + "/outputs/bprmf.csv"
    if reorders is not None and 'BPRMF' in reorders:
        output_names.add(bprmf_output_file.split("/")[-1])

    # 3 - User KNN
    knn_output_file = fold + str(n_fold) + "/outputs/userknn.csv"
    if reorders is not None and 'UserKNN' in reorders:
        output_names.add(knn_output_file.split("/")[-1])

    # 4 - Wikidata PageRank params: weights=[80, 0, 20]
    pr = pagerank.PageRankRecommnder(fold + str(n_fold), "wikidata_page_rank8020.csv", 20,
                                     "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                     node_weighs=[0.8, 0, 0.2], prop_cols=['movieId', 'title', 'prop', 'obj'],
                                     cols_used=[0, 1, 2], col_names=['user_id', 'movie_id', 'feedback'])
    if reorders is not None and 'PageRank' in reorders:
        output_names.add(pr.output_path.split("/")[-1])

    # 5 - Neural Collaborative Filtering
    ncf_rec = ncf.NCF(fold + str(n_fold), "ncf.csv", rank_size=20, factors=32, layers=[64, 32, 16, 8],
                      epochs=10, neg_smp_train=4, neg_smp_test=100, cols_used=[0, 1, 2],
                      col_names=['user_id', 'movie_id', 'feedback'],
                      model_path=folds_path_ml + str(0) + "/model.pt", batch_size=256, seed=42, model_disk='w')
    if reorders is not None and 'NCF' in reorders:
        output_names.add(ncf_rec.output_path.split("/")[-1])

    # 6 - EASE Algorithm
    ease_rec = ease.EASE(fold + str(n_fold), "ease.csv", rank_size=20, cols_used=[0, 1, 2],
                         col_names=['user_id', 'movie_id', 'feedback'], lambda_=500)
    if reorders is not None and 'EASE' in reorders:
        output_names.add(ease_rec.output_path.split("/")[-1])

    # REORDERS
    train_file = fold + str(n_fold) + "/train.dat"
    output_files = [fold + str(n_fold) + "/outputs/" + name for name in output_names]

    # Path reorder
    for output_file in output_files:
        path_reord = PathReordering(train_file, output_file,
                                    "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                    cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                                    prop_cols=['movieId', 'title', 'prop', 'obj'], n_reorder=n_reorder,
                                    p_items=p_items, policy=policy, hybrid=True)
        path_reord.reorder_with_path(h_min, h_max, max_users)


def run_explanations_lastfm(fold: str, n_fold: int, reorders=None, n_reorder=10, p_items=0.1, policy='last', h_min=0,
                        h_max=20, max_users=1):
    """
    Run explanation experiments for the lastfm dataset for the quantity of folds passed by in the parameter n_folds.
    E.g. if 10 then will run for all folds if 6 it will run from fold 0 to 5, etc
    :param fold: path of the folds
    :param n_fold: number of the fold to generate explanations to users
    :param reorders: list to reorder recommender algorithms that had run already
    :param n_reorder: quantity of items to reorder
    :param p_items: percentage of items from historic to build semantic profile
    :param policy: policy to choose items
    :param: h_min: minimum number of users' historic items to generate the recommendations and explanations to, if a
        user has a smaller number of interacted items than this parameter the algorithm will not generate explanations
    :param: h_max: maximum number of users' historic items to generate the recommendations and explanations to, if a
        user has a bigger number of interacted items than this parameter the algorithm will not generate explanations
    :param: max_users: maximum number of user to generate explanations to, when the program reaches max_number it stops
    :return: users are displayed on console with interacted items, recommended items, semantic profile, reordered items
        and explanation paths for each recommended item
    """

    output_names = set([])

    # BASELINES
    # 1 - Most Popular Algorithm
    most_pop_output_file = fold + str(n_fold) + "/outputs/mostpop.csv"
    if reorders is not None and 'MostPop' in reorders:
        output_names.add(most_pop_output_file.split("/")[-1])

    # 2 - BPR MF
    bprmf_output_file = fold + str(n_fold) + "/outputs/bprmf.csv"
    if reorders is not None and 'BPRMF' in reorders:
        output_names.add(bprmf_output_file.split("/")[-1])

    # 3 - User KNN
    knn_output_file = fold + str(n_fold) + "/outputs/userknn.csv"
    if reorders is not None and 'UserKNN' in reorders:
        output_names.add(knn_output_file.split("/")[-1])

    # 4 - Wikidata PageRank params: weights=[80, 0, 20]
    pr = pagerank.PageRankRecommnder(fold + str(n_fold), "wikidata_page_rank8020.csv", 20,
                                     "./generated_files/wikidata/last-fm/props_artists_id.csv",
                                     node_weighs=[0.8, 0, 0.2], prop_cols=['id', 'artist', 'prop', 'obj'],
                                     cols_used=[0, 1, 2], col_names=['user_id', 'artist_id', 'feedback'])
    if reorders is not None and 'PageRank' in reorders:
        output_names.add(pr.output_path.split("/")[-1])

    # 5 - Neural Collaborative Filtering
    ncf_rec = ncf.NCF(fold + str(n_fold), "ncf.csv", rank_size=20, factors=32, layers=[64, 32, 16, 8],
                      epochs=10, neg_smp_train=4, neg_smp_test=100, cols_used=[0, 1, 2],
                      col_names=['user_id', 'artist_id', 'feedback'],
                      model_path=folds_path_ml + str(0) + "/model.pt", batch_size=256, seed=42, model_disk='w')
    if reorders is not None and 'NCF' in reorders:
        output_names.add(ncf_rec.output_path.split("/")[-1])

    # 6 - EASE Algorithm
    ease_rec = ease.EASE(fold + str(n_fold), "ease.csv", rank_size=20, cols_used=[0, 1, 2],
                         col_names=['user_id', 'artist_id', 'feedback'], lambda_=500)
    if reorders is not None and 'EASE' in reorders:
        output_names.add(ease_rec.output_path.split("/")[-1])

    # REORDERS
    train_file = fold + str(n_fold) + "/train.dat"
    output_files = [fold + str(n_fold) + "/outputs/" + name for name in output_names]

    # Path reorder
    for output_file in output_files:
        path_reord = PathReordering(train_file, output_file,
                                    "./generated_files/wikidata/last-fm/props_artists_id.csv",
                                    cols_used=['user_id', 'artist_id', 'interaction'],
                                    prop_cols=['id', 'artist', 'prop', 'obj'], n_reorder=n_reorder, p_items=p_items,
                                    policy=policy, hybrid=True)
        path_reord.reorder_with_path(h_min, h_max, max_users)


parser = argparse.ArgumentParser()

# required arguments
parser.add_argument("--mode",
                    type=str,
                    default="run",
                    required=True,
                    help="Set 'run' to run experiments, 'validate' to run statistical relevance tests and "
                         "'explanation' to generate explanation paths to users")
parser.add_argument("--dataset",
                    type=str,
                    default="ml",
                    required=True,
                    help="Data set. Either 'ml' for the movielens dataset or 'lastfm' for the lastfm dataset")

# run commands
parser.add_argument("--begin",
                    type=int,
                    default=0,
                    help="Fold to start the experiment")
parser.add_argument("--end",
                    type=int,
                    default=9,
                    help="Fold to end the experiment")
parser.add_argument("--alg",
                    type=str,
                    default="None",
                    help="Algoritms to run separated by space. E.g.: MostPop BPRMF UserKNN PageRank NCF EASE."
                         "Only works on the 'run' mode")
parser.add_argument("--reord",
                    type=str,
                    default="None",
                    help="Algoritms to reorder separated by space. E.g.: MostPop BPRMF UserKNN PageRank NCF EASE."
                         "Only works on the 'run' mode")
parser.add_argument("--nreorder",
                    type=int,
                    default=10,
                    help="Number of recommendations to reorder. Only works on the 'run' and 'explanation' modes.")
parser.add_argument("--pitems",
                    type=float,
                    default=0.1,
                    help="Set of items to build user semantic profile. Only works on the 'run' and 'explanation' modes.")
parser.add_argument("--policy",
                    type=str,
                    default='last',
                    help="Policy to extract set of items to build semantic profile. 'all' to get all items, 'last' for"
                         "the last interacted, 'first' for the first interacted, 'random' for random items."
                         "Only works on the 'run' and 'explanation' modes.")

# validate commands
parser.add_argument("--baseline",
                    type=str,
                    default="userknn",
                    help="Name of the file without extension of the baseline to validate results. "
                         "E.g.: 'bprmf'. Only works on the 'validation' mode.")
parser.add_argument("--sufix",
                    type=str,
                    default="path[policy=last_items=01_reorder=10_hybrid]",
                    help="Reorder sufix on result file after the string of the baseline. "
                         "E.g.: path[policy=last_items=01_reorder=10_hybrid]. Only works on the 'validation' mode.")
parser.add_argument("--metrics",
                    type=str,
                    default="PREC RECALL MAP NDCG GINI ENTROPY AGG_DIV COVERAGE",
                    help="Metrics to evaluate the statistical relevance. "
                         "E.g.: PREC RECALL MAP NDCG GINI ENTROPY AGG_DIV COVERAGE."
                         "Only works on the 'validation' mode.")
parser.add_argument("--method",
                    type=str,
                    default='wilcoxon',
                    help="Statistical relevance test. Either 'ttest', 'wilcoxon' or 'both'. "
                         "Only works on the 'validation' mode.")
parser.add_argument("--save",
                    type=bool,
                    default=False,
                    help="Boolean argument to save or not result in file. Only works on the 'validation' mode.")

# reorder commdands
parser.add_argument("--fold",
                    type=int,
                    default=0,
                    help="Fold to consider when generating explanations. Only works on 'explanation' mode")

parser.add_argument("--min",
                    type=int,
                    default=0,
                    help="Minimum number of user interacted items to explain. Works on the 'explanation' mode")

parser.add_argument("--max",
                    type=int,
                    default=20,
                    help="Maximum number of user interacted items to explain. Works on the 'explanation' mode")

parser.add_argument("--max_users",
                    type=int,
                    default=1,
                    help="Maximum number of users to generate explanations to. Works on the 'explanation' mode.")

# parse arguments
args = parser.parse_args()

folds_path_ml = "./datasets/ml-latest-small/folds/"
folds_path_lastfm = "./datasets/hetrec2011-lastfm-2k/folds/"

if args.mode == "run" and args.dataset == "ml":
    run_experiments_ml(folds_path_ml, args.begin, args.end, args.alg.split(), args.reord.split(),
                       args.nreorder, args.pitems, args.policy)

if args.mode == "run" and args.dataset == "lastfm":
    run_experiments_lastfm(folds_path_lastfm, args.begin, args.end, args.alg.split(), args.reord.split(),
                           args.nreorder, args.pitems, args.policy)

if args.mode == "validate" and args.dataset == "ml":
    statistical_relevance(args.sufix, args.baseline, folds_path_ml,
                          args.metrics.split(), method=args.method, save=args.save)

if args.mode == "validate" and args.dataset == "lastfm":
    statistical_relevance(args.sufix, args.baseline, folds_path_lastfm,
                          args.metrics.split(), method=args.method, save=args.save)

if args.mode == "explanation" and args.dataset == "ml":
    run_explanations_ml(folds_path_ml, args.fold, args.reord.split(), args.nreorder, args.pitems, args.policy, args.min,
                        args.max, args.max_users)

if args.mode == "explanation" and args.dataset == "lastfm":
    run_explanations_lastfm(folds_path_lastfm, args.fold, args.reord.split(), args.nreorder, args.pitems, args.policy, args.min,
                        args.max, args.max_users)
