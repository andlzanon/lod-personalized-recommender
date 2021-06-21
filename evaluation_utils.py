import numpy as np
import pandas as pd
from caserec.evaluation.item_recommendation import ItemRecommendationEvaluation
from pygini import gini
from scipy.stats import entropy
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon


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
    total_items = len(train[train.columns[1]].unique())
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
        out_at['prob'] = out_at['count'] / len(out_at.index)
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


def statistical_relevance(proposed: str, baseline: str, dataset: str, metrics: list, method='both'):
    """
    Function that calculates the statistical relevance with p_value
    :param proposed: proposed method final name. e.g.: path[policy=last_items=01_reorder=10]
    :param baseline: baseline name e.g.: bprmf, knn, mostpop, wikidata_page_rank8020
    :param dataset: path to the folds of the chosen dataset. e.g. "./datasets/ml-latest-small/folds/"
    :param metrics: list with the methods do analyse from the list ["PREC, RECALL", "MAP", "NDCG", "GINI", "ENTROPY", "COVERAGE"]
    :param method: method to test the statistical relevance, either 'ttest', 'wilcoxon' or 'both' that is the default value
    :return: the statistical relevance of the proposed with the baseline for the metrics chosen for @1, @3, @5 and @10
    """
    ats = [1, 3, 5, 10]

    base_results = {}
    for m in metrics:
        for at in ats:
            base_results[m + "@" + str(at)] = []

    prop_results = {}
    for m in metrics:
        for at in ats:
            prop_results[m + "@" + str(at)] = []

    for i in range(0, 10):
        base_path = dataset + str(i) + "/results/" + baseline + ".csv"
        prop_path = dataset + str(i) + "/results/" + baseline + "_lodreorder_" + proposed + ".csv"

        try:
            base_df = file_to_df(base_path, baseline)
            prop_df = file_to_df(prop_path, proposed)
        except FileNotFoundError:
            continue

        for m in metrics:
            for at in ats:
                base_value = float(base_df[(base_df['metric'] == m) & (base_df['@'] == at)]['value'])
                base_results[m + "@" + str(at)].append(base_value)

                prop_value = float(prop_df[(prop_df['metric'] == m) & (prop_df['@'] == at)]['value'])
                prop_results[m + "@" + str(at)].append(prop_value)

    for m in metrics:
        for at in ats:
            key = m + "@" + str(at)
            print("---" + key + "---")
            base_list = base_results[key]
            print("Results of the baseline algorithm: " + str(base_list) +
                  " mean: " + str(sum(base_list)/len(base_list)) + " -> " + baseline)

            prop_list = prop_results[key]
            print("Results of the proposed algorithm: " + str(prop_list) +
                  " mean: " + str(sum(prop_list)/len(prop_list)) + " -> " + proposed)

            if method == 'ttest':
                tt, tp = ttest_rel(base_list, prop_list)
                print("p-value with t-test: " + str(tp))
            elif method == 'wilcoxon':
                wt, wp = wilcoxon(base_list, prop_list)
                print("p-value with wilcoxon: " + str(wp))
            else:
                wt, wp = wilcoxon(base_list, prop_list)
                print("p-value with wilcoxon: " + str(wp))
                tt, tp = ttest_rel(base_list, prop_list)
                print("p-value with t-test: " + str(tp))


def file_to_df(file_path: str, algorithm: str):
    """
    Transform a result file into
    :param file_path: path to the result file of the algorithm
    :param algorithm: name of the algorithm
    :return: pandas df with the results
    """
    df = pd.DataFrame()
    f = open(file_path, 'r')
    for l in f.readlines()[1:]:
        l = l.split("\n")[0]
        metric_value = l.split('@')
        if metric_value[0] != metric_value[-1]:
            metric = metric_value[0]
            at_value = metric_value[-1].split(" ")
            at = int(at_value[0])
            value = at_value[-1]
            df = df.append({'alg': algorithm, '@': at, 'metric': metric, 'value': value}, ignore_index=True)
    f.close()
    return df
