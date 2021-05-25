from caserec.utils.split_database import SplitDatabase


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
