from caserec.utils.split_database import SplitDatabase


def cross_validation_ml_small(rs: int):
    """
    Split the dataset into cross validation folders
    :param rs: random state integer arbitrary number
    :return: folders created on the dataset repository
    """
    SplitDatabase(input_file="./datasets/ml-latest-small/ratings.csv",
                  dir_folds="./datasets/ml-latest-small/", sep_read=',', sep_write=',',
                  header=True, names=['userId','movieId','rating','timestamp'],
                  n_splits=10).k_fold_cross_validation(random_state=rs)