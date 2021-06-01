from preprocessing import movielens_small_utils as ml_small
from caserec.recommenders.item_recommendation.userknn import UserKNN
from caserec.recommenders.item_recommendation.most_popular import MostPopular
from caserec.recommenders.item_recommendation.bprmf import BprMF
from recommenders import page_rank_recommender as pagerank
from recommenders import lod_reordering
import utils

fold0_train = "./datasets/ml-latest-small/folds/0/train.dat"
fold0_test = "./datasets/ml-latest-small/folds/0/test.dat"

# utils.cross_validation_ml_small(rs=42)
# ml_small.extract_wikidata_prop()

# Page rank weights 80/20
#pr_rec8020 = pagerank.PageRankRecommnder("./datasets/ml-latest-small/folds/0", "wikidata_page_rank8020.csv", 10,
#                                        "./generated_files/wikidata/props_wikidata_movielens_small.csv",
#                                        cols_used=[0, 1, 2], col_names=['user_id', 'movie_id', 'feedback'])

#pr_rec8020.run()

#utils.evaluate("Page Rank 80/20 Algorithm", pr_rec8020.output_path, fold0_test,
#               "./datasets/ml-latest-small/folds/0/results/" + pr_rec8020.output_filename)


# Page Rank 40/40/20
#pr2_rec = pagerank.PageRankRecommnder("./datasets/ml-latest-small/folds/0", "wikidata_page_rank404020.csv", 10,
#                                      "./generated_files/wikidata/props_wikidata_movielens_small.csv",
#                                      cols_used=[0, 1, 2], col_names=['user_id', 'movie_id', 'feedback'],
#                                      node_weighs=[0.4, 0.4, 0.2])

#pr2_rec.run()
#utils.evaluate("Page Rank 80/20 Algorithm", pr2_rec.output_path, fold0_test,
#               "./datasets/ml-latest-small/folds/0/results/" + pr2_rec.output_filename)


# user knn
#knn_output_file = "datasets/ml-latest-small/folds/0/outputs/userknn.csv"
#UserKNN(fold0_train, fold0_test, sep=",", output_file=knn_output_file).compute()
#utils.evaluate("KNN Algorithm", knn_output_file, fold0_test, "./datasets/ml-latest-small/folds/0/results/userknn.csv")

# most popular
#most_pop_output_file = "datasets/ml-latest-small/folds/0/outputs/mostpop.csv"
#MostPopular(fold0_train, fold0_test, sep=',', output_file=most_pop_output_file).compute()
#utils.evaluate("Most Pop Algorithm", most_pop_output_file, fold0_test, "./datasets/ml-latest-small/folds/0/results/mostpop.csv")


#BPRMF
#bprmf_output_file = "datasets/ml-latest-small/folds/0/outputs/bprmf.csv"
#BprMF(fold0_train, fold0_test, sep=',', output_file=bprmf_output_file).compute()
#utils.evaluate("BPR-MF Algorithm", bprmf_output_file, fold0_test, "./datasets/ml-latest-small/folds/0/results/bprmf.csv")

reord = lod_reordering.LODPersonalizedReordering(fold0_train,
                    "./datasets/ml-latest-small/folds/0/outputs/wikidata_page_rank8020.csv",
                    "./generated_files/wikidata/props_wikidata_movielens_small.csv",
                    cols_used=['user_id', 'movie_id', 'interaction', 'timestamp'],
                    prop_cols=['movieId','title','prop','obj'], policy='random', p_items=0.1)

reord.reorder()

utils.evaluate("Reorder Page Rank 80/20 Algorithm",
               "./datasets/ml-latest-small/folds/0/outputs/wikidata_page_rank8020_lodreorder.csv",
               fold0_test, "./datasets/ml-latest-small/folds/0/results/wikidata_page_rank8020_lodreorder_random.csv")