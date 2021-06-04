from preprocessing import movielens_small_utils as ml_small
from caserec.recommenders.item_recommendation.userknn import UserKNN
from caserec.recommenders.item_recommendation.most_popular import MostPopular
from caserec.recommenders.item_recommendation.bprmf import BprMF
from recommenders import page_rank_recommender as pagerank
from recommenders.prop_reordering import PropReordering
from recommenders.path_reordering import PathReordering
import utils

fold0_train = "./datasets/ml-latest-small/folds/0/train.dat"
fold0_test = "./datasets/ml-latest-small/folds/0/test.dat"

# utils.cross_validation_ml_small(rs=42)
# ml_small.extract_wikidata_prop()
utils.split_dataset_by_timestamp("./datasets/ml-latest-small/ratings.csv", 0.1, "./datasets/ml-latest-small/folds/timed")


# Page rank weights 80/20
pr_rec8020 = pagerank.PageRankRecommnder("./datasets/ml-latest-small/folds/0", "wikidata_page_rank8020.csv", 20,
                                         "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                         cols_used=[0, 1, 2], col_names=['user_id', 'movie_id', 'feedback'])

# pr_rec8020.run()
# utils.evaluate("Page Rank 80/20 Algorithm", pr_rec8020.output_path, fold0_test)

# Page Rank 40/40/20
pr2_rec = pagerank.PageRankRecommnder("./datasets/ml-latest-small/folds/0", "wikidata_page_rank404020.csv", 20,
                                      "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                                      cols_used=[0, 1, 2], col_names=['user_id', 'movie_id', 'feedback'],
                                      node_weighs=[0.4, 0.4, 0.2])

# pr2_rec.run()
# utils.evaluate("Page Rank 80/20 Algorithm", pr2_rec.output_path, fold0_test)

# user knn
knn_output_file = "datasets/ml-latest-small/folds/0/outputs/userknn.csv"
# UserKNN(fold0_train, fold0_test, sep=",", output_file=knn_output_file, rank_length=20).compute()
# utils.evaluate("KNN Algorithm", knn_output_file, fold0_test)

# most popular
most_pop_output_file = "datasets/ml-latest-small/folds/0/outputs/mostpop.csv"
# MostPopular(fold0_train, fold0_test, sep=',', output_file=most_pop_output_file, rank_length=20).compute()
# utils.evaluate("Most Pop Algorithm", most_pop_output_file, fold0_test)

# BPRMF
bprmf_output_file = "datasets/ml-latest-small/folds/0/outputs/bprmf.csv"
# BprMF(fold0_train, fold0_test, sep=',', output_file=bprmf_output_file, rank_length=20).compute()
# utils.evaluate("BPR-MF Algorithm", bprmf_output_file, fold0_test)

#prop_reord = PropReordering(fold0_train,
#                            "./datasets/ml-latest-small/folds/0/outputs/bprmf.csv",
#                            "./generated_files/wikidata/props_wikidata_movielens_small.csv",
#                            cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
#                            prop_cols=['movieId', 'title', 'prop', 'obj'], hybrid=True)

# prop_reord.reorder()
# utils.evaluate("BPRMF Prop Reorder Algorithm", prop_reord.output_path, fold0_test)


#path_reord = PathReordering(fold0_train,
#                            "./datasets/ml-latest-small/folds/0/outputs/bprmf.csv",
#                            "./generated_files/wikidata/props_wikidata_movielens_small.csv",
#                            cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
#                            prop_cols=['movieId', 'title', 'prop', 'obj'], p_items=0.1, policy='last', hybrid=True)

#path_reord.reorder()
#utils.evaluate("Reorder Path MEAN BPRMF Algorithm", path_reord.output_path, fold0_test)


#path_reord = PathReordering(fold0_train,
#                            "./datasets/ml-latest-small/folds/0/outputs/wikidata_page_rank8020.csv",
#                            "./generated_files/wikidata/props_wikidata_movielens_small.csv",
#                            cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
#                            prop_cols=['movieId', 'title', 'prop', 'obj'], p_items=0.1, policy='last', hybrid=True)

#path_reord.reorder()
#utils.evaluate("Reorder Path Page Rank 80/20 Algorithm", path_reord.output_path, fold0_test)

path_reord = PathReordering(fold0_train,
                            "./datasets/ml-latest-small/folds/0/outputs/userknn.csv",
                            "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                            cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                            prop_cols=['movieId', 'title', 'prop', 'obj'], p_items=0.1, policy='last', hybrid=True)

path_reord.reorder()
utils.evaluate("Reorder Path USER KNN Algorithm", path_reord.output_path, fold0_test)
