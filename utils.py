import pandas as pd
from caserec.utils.split_database import SplitDatabase
from caserec.evaluation.item_recommendation import ItemRecommendationEvaluation


def split_dataset_by_timestamp(rating_file: str, test_size: float, output_folder: str):
    """
    Split dataset base on timestamp of interaction of user with items, therefore the first items that the user interacted
    will be in the train file and on the test size will be the the last test size percentage of items lately interacted
    :param rating_file: file with ratings of the dataset
    :param test_size: percentage of items per user on the test size
    :param output_folder: output folder of the test and train files
    :return: files of test and train set on the output folder
    """
    dataset = pd.read_csv(rating_file)
    dataset[dataset.columns[2]] = 1
    dataset = dataset.set_index(dataset.columns[0])

    test_set = pd.DataFrame()
    train_set = pd.DataFrame()
    for u in dataset.index.unique():
        u_set = dataset.loc[u]
        u_set = u_set.sort_values('timestamp', ascending=False)

        split = int(u_set.size * test_size)

        test_user = u_set.iloc[:split, :].reset_index()
        train_user = u_set.iloc[split:, :].reset_index()

        test_set = pd.concat([test_set, test_user], ignore_index=True)
        train_set = pd.concat([train_set, train_user], ignore_index=True)

    test_set.to_csv(output_folder + "/test.dat", mode='w', header=False, index=False)
    train_set.to_csv(output_folder + "/train.dat", mode='w', header=False, index=False)


def cross_validation_ml_small(rs: int):
    """
    Split the dataset into cross validation folders
    To read the file use the command: df = pd.read_csv("./datasets/ml-latest-small/folds/0/test.dat", header=None)
    :param rs: random state integer arbitrary number
    :return: folders created on the dataset repository
    """
    SplitDatabase(input_file="./datasets/ml-latest-small/ratings.csv",
                  dir_folds="./datasets/ml-latest-small/", as_binary=True, binary_col=2, header=1,
                  sep_read=',', sep_write=',', n_splits=10).k_fold_cross_validation(random_state=rs)


def evaluate(alg_name: str, prediction_file: str, test_file: str):
    """
    Evaluate the output file
    :param alg_name: name of the algorithm
    :param prediction_file: file of the predictions made by an algorithm
    :param test_file: test file path
    :return: file on results folder with results
    """

    output_file = prediction_file.replace('outputs', 'results')
    results = ItemRecommendationEvaluation(sep=',').evaluate_with_files(prediction_file, test_file)

    f = open(output_file, "w")
    f.write("--- " + alg_name + " --- \n")
    for key in results.keys():
        f.write(str(key) + " " + str(results[key]) + "\n")
    f.close()
