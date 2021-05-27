from caserec.utils.split_database import SplitDatabase
from caserec.evaluation.item_recommendation import ItemRecommendationEvaluation


def cross_validation_ml_small(rs: int):
    """
    Split the dataset into cross validation folders
    To read the file use the command: df = pd.read_csv("./datasets/ml-latest-small/folds/0/test.dat", header=None)
    :param rs: random state integer arbitrary number
    :return: folders created on the dataset repository
    """
    SplitDatabase(input_file="./datasets/ml-latest-small/ratings_notime.csv",
                  dir_folds="./datasets/ml-latest-small/", as_binary=True, binary_col=2,
                  sep_read=',', sep_write=',', n_splits=10).k_fold_cross_validation(random_state=rs)


def evaluate(alg_name: str, prediction_file: str, test_file: str, output_file: str):
    results = ItemRecommendationEvaluation(sep=',').evaluate_with_files(prediction_file, test_file)

    f = open(output_file, "w")
    f.write("--- " + alg_name + " --- \n")
    for key in results.keys():
        f.write(str(key) + " " + str(results[key]) + "\n")
    f.close()
