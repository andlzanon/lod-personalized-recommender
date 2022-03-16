import numpy as np
import pandas as pd
from caserec.evaluation.item_recommendation import ItemRecommendationEvaluation
from openpyxl import load_workbook
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
    if "ncf" not in prediction_file:
        diver_metrics = evaluate_diversity(prediction_file, train_file)
    else:
        diver_metrics = {}

    # print(diver_metrics)

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
        agg_div = len(l)
        cov = agg_div / total_items

        div['AGG_DIV@' + str(at)] = agg_div
        div['GINI@' + str(at)] = g
        div['ENTROPY@' + str(at)] = en
        div['COVERAGE@' + str(at)] = cov

    return div


def statistical_relevance(proposed: str, baseline: str, dataset: str, metrics: list, method='both', save=False):
    """
    Function that calculates the statistical relevance with p_value
    :param proposed: proposed method final name. e.g.: path[policy=last_items=01_reorder=10]
    :param baseline: baseline name e.g.: bprmf, knn, mostpop, wikidata_page_rank8020
    :param dataset: path to the folds of the chosen dataset. e.g. "./datasets/ml-latest-small/folds/"
    :param metrics: list with the methods do analyse from the list
                    ["PREC, RECALL", "MAP", "NDCG", "GINI", "ENTROPY", "AGG_DIV", "COVERAGE"]
    :param method: method to test the statistical relevance, either 'ttest', 'wilcoxon' or 'both' that is the default value
    :param save: flag to save file in dataset directory
    :return: the statistical relevance of the proposed with the baseline for the metrics chosen for @1, @3, @5 and @10
    """
    div_metrics = ["GINI", "ENTROPY", "COVERAGE", "AGG_DIV"]
    results = pd.DataFrame(columns=['METRIC',
                                    'PROPOSED NAME', 'PROPOSED MEAN',
                                    'BASELINE NAME', 'BASELINE MEAN',
                                    'WILCOXON', 'TTEST'])

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

            if ("reorder=10" in proposed) and (m in div_metrics) and (at == 10):
                continue

            key = m + "@" + str(at)
            print("---" + key + "---")

            base_list = base_results[key]
            base_mean = sum(base_list)/len(base_list)
            print("Results of the baseline algorithm: " + str(base_list) +
                  " mean: " + str(base_mean) + " -> " + baseline)

            prop_list = prop_results[key]
            prop_mean = sum(prop_list)/len(prop_list)
            print("Results of the proposed algorithm: " + str(prop_list) +
                  " mean: " + str(prop_mean) + " -> " + proposed)

            wp = None
            tp = None
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

            results = results.append({'METRIC': key,
                            'PROPOSED NAME': proposed, 'PROPOSED MEAN': prop_mean,
                            'BASELINE NAME': base_mean, 'BASELINE MEAN': baseline,
                            'WILCOXON': wp, 'TTEST': tp}, ignore_index=True)

    if save:
        p = dataset[:-6] + "results_" + proposed + ".xlsx"
        try:
            book = load_workbook(p)
            writer = pd.ExcelWriter(p, mode='r+')
            writer.book = book
            writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        except FileNotFoundError:
            writer = pd.ExcelWriter(p, mode='w+')
        results.to_excel(writer, sheet_name=baseline, index=False)
        writer.save()
        writer.close()
        print("--- FILE SAVE AT " + p + "---")


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


def evaluate_explanations(file_name: str, m_items: list, m_props: list, total_items: dict, total_props: dict,
                          train_set: pd.DataFrame, prop_set: pd.DataFrame):
    """
    Diversity of explanation metrics. We will use 10 metrics: mean and std item and prop user diversity
    total item and prop aggregate diversity and entropy and gini
    :param file_name: the name of the results file
    :param m_items: distribution of different historic items shown to user in explanations
    :param m_props:distribution of different KG props shown to user in explanations
    :param total_items: dict of items, quantity of times used in explanations
    :param total_props: dict of prop, quantity of times used in explanations
    :param train_set: pandas dataframe of the train set
    :param prop_set: pandas dataframe of the properties extracted from the KG
    :return: file saved with metrics
    """
    t_items = len(train_set[train_set.columns[0]].unique())
    t_props = len(prop_set[prop_set.columns[-1]].unique())
    items_distr = list(total_items.values())
    props_distr = list(total_props.values())
    item_probs = items_distr + [0 for i in range(0, t_items - len(items_distr))]
    props_probs = props_distr + [0 for i in range(0, t_props - len(props_distr))]

    mean_useritem_aggr = "Mean user item aggregate diversity: " + str(np.array(m_items).mean())
    std_useritem_aggr = "Std user item aggregate diversity: " + str(np.array(m_items).std())
    mean_userprop_aggr = "Mean user property aggregate diversity: " + str(np.array(m_props).mean())
    std_userprop_aggr = "Std user property aggregate diversity: " + str(np.array(m_props).std())
    total_items_str = "Total items aggregate diversity: " + str(len(total_items))
    total_props_str = "Total property aggregate diversity: " + str(len(total_props))
    ientropy = "Items entropy: " + str(entropy(item_probs))
    pentropy = "Props entropy: " + str(entropy(props_probs))
    igini = "Items Gini index: " + str(gini(np.array(item_probs, dtype=np.float64)))
    pgini = "Props Gini index: " + str(gini(np.array(props_probs, dtype=np.float64)))

    f = open(file_name, mode="w", encoding='utf-8')
    f.write(file_name + "\n")
    f.write(mean_useritem_aggr + "\n")
    f.write(std_useritem_aggr + "\n")
    f.write(mean_userprop_aggr + "\n")
    f.write(std_userprop_aggr + "\n")
    f.write(total_items_str + "\n")
    f.write(total_props_str + "\n")
    f.write(ientropy + "\n")
    f.write(pentropy + "\n")
    f.write(igini + "\n")
    f.write(pgini + "\n")
    f.close()

    print("\n" + file_name)
    print(mean_useritem_aggr)
    print(std_useritem_aggr)
    print(mean_userprop_aggr)
    print(std_userprop_aggr)
    print(total_items_str)
    print(total_props_str)
    print(ientropy)
    print(pentropy)
    print(igini)
    print(pgini)


def explanation_file_to_df(file_path: str, algorithm: str):
    """
    Function that generates a dataframe with the offline explanations metrics for the algorithm
    "algorithm" on file "file path"
    :param file_path: path of the file
    :param algorithm: name of the algorithm
    :return: dataframe with offline explanation metrics of the algorithm retrieved from file
    """
    df = pd.DataFrame()
    f = open(file_path, 'r')
    for l in f.readlines()[1:]:
        l = l.split("\n")[0]
        metric_value = l.split(':')
        if metric_value[0] != metric_value[-1]:
            metric = metric_value[0]
            at_value = metric_value[1].split(" ")
            value = at_value[-1]
            df = df.append({'alg': algorithm, 'metric': metric, 'value': value}, ignore_index=True)
            # print(str(metric) + " " + str(value))
    f.close()
    return df


def statistical_relevance_explanations(rec_alg: str, dataset: str, reordered: int):
    """
    Compute statistical relevance test for explanations evaluation
    :param rec_alg: recommendation algorithm used
    :param dataset: dataset (ml or last-fm)
    :param reordered: 1 if the recommendations were reordred by proposed reordering system 0 if not
    :return: excel file with wilcoxon and ttest statistical relevance tests for all metrics
    """
    basline_results = []
    proposed_results = []

    path = "./datasets/"
    if dataset == "ml":
        path = path + "ml-latest-small"
    else:
        path = path + "hetrec2011-lastfm-2k"
    path_base = path + "/folds/"

    for i in range(0, 10):
        path = path_base + str(i) + "/results/explanations/" + "reordered_recs=" + str(reordered)
        path_baseline = path + "_expl_alg=explod_" + str(rec_alg) + \
                        "_lodreorder_path[policy=last_items=01_reorder=10_hybrid].csv"

        path_proposed = path + "_expl_alg=diverse_" + str(rec_alg) + \
                        "_lodreorder_path[policy=last_items=01_reorder=10_hybrid].csv"

        basline_results.append(explanation_file_to_df(path_baseline, "explod").set_index("metric"))
        proposed_results.append(explanation_file_to_df(path_proposed, "diverse").set_index("metric"))
        path = path_base

    metrics = list(basline_results[0].index.unique())
    results_df = pd.DataFrame(columns=["METRIC", "PROPOSED_NAME",
                                       "PROPOSED_MEAN", "BASELINE_NAME",
                                       "BASELINE_MEAN", "WILCOXON", "TTEST"])
    for m in metrics:
        m_baseline = []
        m_proposed = []
        for i in range(0, 10):
            m_baseline.append(float(basline_results[i].loc[m].value))
            m_proposed.append(float(proposed_results[i].loc[m].value))

        baseline_mean = np.array(m_baseline).mean()
        print(m + " baseline results: ", m_baseline)
        proposed_mean = np.array(m_proposed).mean()
        print(m + " proposed results: ", m_proposed)

        wt, wp = wilcoxon(m_baseline, m_proposed)
        print("p-value with wilcoxon: " + str(wp))
        tt, tp = ttest_rel(m_baseline, m_proposed)
        print("p-value with t-test: " + str(tp))

        results_df = results_df.append({"METRIC": m, "PROPOSED_NAME": "diverse",
                                        "PROPOSED_MEAN": proposed_mean,
                                        "BASELINE_NAME": "explod",
                                        "BASELINE_MEAN": baseline_mean,
                                        "WILCOXON": wp, "TTEST": tp}, ignore_index=True)

    p = path_base[:-6] + "explanation_results_reordered=" + str(reordered) + ".xlsx"
    try:
        book = load_workbook(p)
        writer = pd.ExcelWriter(p, mode='r+')
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    except FileNotFoundError:
        writer = pd.ExcelWriter(p, mode='w+')
    results_df.to_excel(writer, sheet_name=rec_alg, index=False)
    writer.save()
    writer.close()
    print("--- FILE SAVE AT " + p + "---")



