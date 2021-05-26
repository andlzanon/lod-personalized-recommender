from preprocessing import movielens_small_utils as ml_small
from preprocessing import slip_and_cross_validation as split
from recommenders import page_rank_recommender as pagerank

# split.cross_validation_ml_small(rs=42)

# ml_small.extract_wikidata_prop()

pr_rec = pagerank.PageRankRecommnder("./datasets/ml-latest-small/folds/0", "wikidata_page_rank8020", 10,
                            "./generated_files/wikidata/props_wikidata_movielens_small.csv")

# pr_rec.run()

results = pr_rec.evaluate()

for key in results.keys():
    print(str(key) + " " + str(results[key]))

# from caserec.recommenders.item_recommendation.userknn import UserKNN
# UserKNN("./datasets/ml-latest-small/folds/0/train.dat", "./datasets/ml-latest-small/folds/0/test.dat", sep=",", k_neighbors=5).compute()
