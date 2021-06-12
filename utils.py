import numpy as np
import pandas as pd
from caserec.utils.split_database import SplitDatabase
from caserec.evaluation.item_recommendation import ItemRecommendationEvaluation
from pygini import gini
from scipy.stats import entropy


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


def evaluate(alg_name: str, prediction_file: str, train_file: str, test_file: str):
    """
    Evaluate the output file
    :param alg_name: name of the algorithm
    :param prediction_file: file of the predictions made by an algorithm
    :param train_file: train file path
    :param test_file: test file path
    :return: file on results folder with results
    """

    output_file = prediction_file.replace('outputs', 'results')
    accurate_metrics = ItemRecommendationEvaluation(sep=',').evaluate_with_files(prediction_file, test_file)
    diver_metrics = evaluate_diversity(prediction_file, train_file)
    print(diver_metrics)

    f = open(output_file, "w")
    f.write("--- " + alg_name + " --- \n")
    for key in accurate_metrics.keys():
        f.write(str(key) + " " + str(accurate_metrics[key]) + "\n")

    for key in diver_metrics.keys():
        f.write(str(key) + " " + str(diver_metrics[key]) + "\n")
    f.close()


def evaluate_diversity(prediction_file: str, train_file: str):
    """
    Evaluate the diversity of the recommenders considering the gini index, entropy and coverage of items
    :param prediction_file: output file path of the recommender
    :param train_file: train file path used by the recommender
    :return: the gini, entropy and coverage metrics for top 1, 3, 5 and 10 rankings
    """

    ats = [1, 3, 5, 10]
    train = pd.read_csv(train_file, header=None)
    out = pd.read_csv(prediction_file, header=None)
    out.columns = ['user_id', 'item_id', 'score']
    total_items = train[train.columns[1]].unique().size
    div = {}

    # for all rankings, generate a dataset only with the ranking size considering the full output of the recommenders
    # get the probability of every item happening and calculate gini, entropy and coverage for each
    for at in ats:
        out_at = pd.DataFrame()
        out_u = out.set_index('user_id')
        for u in out_u.index.unique():
            out_at = pd.concat([out_at, out_u.loc[u][:at].reset_index()], ignore_index=True)

        out_at = out_at.set_index('item_id')
        out_at['count'] = out_at.index.value_counts()
        out_at['prob'] = out_at['count'] / out_at.index.size
        out_at = out_at[~out_at.index.duplicated(keep='first')]
        l = out_at['prob'].to_list()
        probs = l + [0 for i in range(0, total_items - len(l))]

        g = gini(np.array(probs))
        en = entropy(probs, base=10)
        cov = len(l) / total_items

        div['GINI@' + str(at)] = g
        div['ENTROPY@' + str(at)] = en
        div['COVERAGE@' + str(at)] = cov

    return div