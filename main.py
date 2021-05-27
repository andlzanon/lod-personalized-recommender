from preprocessing import movielens_small_utils as ml_small
from caserec.recommenders.item_recommendation.userknn import UserKNN
from caserec.recommenders.item_recommendation.most_popular import MostPopular
from caserec.recommenders.item_recommendation.bprmf import BprMF
from recommenders import page_rank_recommender as pagerank
import utils

fold0_train = "./datasets/ml-latest-small/folds/0/train.dat"
fold0_test = "./datasets/ml-latest-small/folds/0/test.dat"

# utils.cross_validation_ml_small(rs=42)
# ml_small.extract_wikidata_prop()

pr_rec = pagerank.PageRankRecommnder("./datasets/ml-latest-small/folds/0", "wikidata_page_rank8020.csv", 10,
                                     "./generated_files/wikidata/props_wikidata_movielens_small.csv")

# pr_rec.run()

utils.evaluate("Page Rank 80/20 Algorithm", pr_rec.output_path, fold0_test,
               "./datasets/ml-latest-small/folds/0/results/" + pr_rec.output_filename)

# user knn
knn_output_file = "datasets/ml-latest-small/folds/0/outputs/userknn.csv"
UserKNN(fold0_train, fold0_test, sep=",", output_file=knn_output_file).compute()
utils.evaluate("KNN Algorithm", knn_output_file, fold0_test, "./datasets/ml-latest-small/folds/0/results/userknn.csv")

# most popular
most_pop_output_file = "datasets/ml-latest-small/folds/0/outputs/mostpop.csv"
MostPopular(fold0_train, fold0_test, sep=',', output_file=most_pop_output_file).compute()
utils.evaluate("Most Pop Algorithm", most_pop_output_file, fold0_test, "./datasets/ml-latest-small/folds/0/results/mostpop.csv")


#BPRMF
bprmf_output_file = "datasets/ml-latest-small/folds/0/outputs/bprmf.csv"
BprMF(fold0_train, fold0_test, sep=',', output_file=bprmf_output_file).compute()
utils.evaluate("BPR-MF Algorithm", bprmf_output_file, fold0_test, "./datasets/ml-latest-small/folds/0/results/bprmf.csv")
