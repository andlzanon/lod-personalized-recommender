import pandas as pd
import time
from preprocessing import dbpedia_utils as from_dbpedia
from preprocessing import wikidata_utils as from_wikidata
from caserec.utils.split_database import SplitDatabase

ml_small_path = "./datasets/ml-latest-small/ratings_implicit.csv"
original_ml_small_path = "./datasets/ml-latest-small/ratings.csv"
movies_ml_small_path = "./datasets/ml-latest-small/movies.csv"
link_ml_small_path = "./datasets/ml-latest-small/links.csv"

db_uri_ml_small_path = "./generated_files/dbpedia/ml-latest-small/uri_dbpedia_movielens_small.csv"
db_nan_ml_small_path = "./generated_files/dbpedia/ml-latest-small/nan_uri_dbpedia_movielens_small.csv"
db_final_ml_small_path = "./generated_files/dbpedia/ml-latest-small/final_uri_dbpedia_movielens_small.csv"

wikidata_props_ml_small = "./generated_files/wikidata/props_wikidata_movielens_small.csv"
dbpedia_props_ml_small = "./generated_files/dbpedia/ml-latest-small/props_dbpedia_movielens_small.csv"


def read_movie_info():
    """
    Function that reads the name of the movies of the small movielens dataset
    :return: pandas DataFrame with the movieId column as index and title as value
    """
    return pd.read_csv(movies_ml_small_path, usecols=['movieId', 'title']).set_index(['movieId'])


def read_links_info():
    """
    Function that reads the imbd set of the movies of the small movielens dataset
    :return: pandas DataFrame with the movieId column as index and imdbId as value
    """
    return pd.read_csv(link_ml_small_path, usecols=['movieId', 'imdbId']).set_index(['movieId'])


def read_uri_info():
    """
    Function that reads the name of the movies of the small movielens dataset with the uri from
    the generate_movies_uri_dbpedia_dataset function
    :return: pandas DataFrame with the movieId column as index and title and uri as value
    """
    return pd.read_csv(db_uri_ml_small_path, ).set_index(['movieId'])


def read_nan_info():
    """
    Function that reads the name of the movies of the small movielens dataset with the uri from
    the generate_recovery_uri_dbpedia function
    :return: pandas DataFrame with the movieId column as index and title as value
    """
    return pd.read_csv(db_nan_ml_small_path).set_index(['movieId'])


def read_final_uri_dbpedia_dataset():
    """
    Function that reads the name of the movies of the small movielens dataset
    :return: pandas DataFrame with the movieId column as index and title as value
    """
    return pd.read_csv(db_final_ml_small_path).set_index(['movieId'])


def read_user_item_interaction():
    """
    Function that reads the user interactions with the movies of the small movielens dataset
    :return: pandas DataFrame of the dataset
    """
    return pd.read_csv(ml_small_path)


def implicit_feedback():
    """
    Function that transform the dataset into implicit feedback
    :return: csv file of the dataset
    """
    df = pd.read_csv(original_ml_small_path)
    df = df[df.rating > 3.5]
    df = df.set_index('userId')
    implicit = pd.DataFrame()

    for u in df.index.unique():
        u_set = df.loc[u]
        if len(u_set) >= 5:
            implicit = pd.concat([implicit, u_set.reset_index()], ignore_index=True)

    implicit.to_csv(ml_small_path, header=None, index=False)


def __get_movie_strings(full_name: str):
    """
    Function that, given a full name on the movielens dataset: Initially, the string is separated by " (" string to
    exclude the year, than the first step is to extract the first and eventually second and third names. Finally, the
    articles of the pt, en, es, it, fr and de are placed on the beginning of the string
    :param full_name: movie name on movielens dataset that follows the patters  "name (year)"; "name, The (year)";
    "name (a.k.a. second_name) (year)" and "name (second_name) (third_name) (year)" and combinations
    :return: a list with all possible movie names on dbpedia
    """

    # remove year of string, if there is no year, then return the full name
    try:
        all_names = full_name.split(" (")
        all_names = all_names[:-1]
        format_names = [all_names[0]]
    except IndexError:
        return [full_name]

    if len(all_names) > 1:
        for i in range(1, len(all_names)):
            # get names on a.k.a parenthesis, else get name between parenthesis
            if all_names[i].find("a.k.a. ") != -1:
                format_names.append(all_names[i].split("a.k.a. ")[1][:-1])
            else:
                format_names.append(all_names[i][:-1])

    # place articles in front of strings
    for i in range(0, len(format_names)):
        fn = format_names[i]
        has_coma = fn.split(", ")
        if len(has_coma[-1]) <= 4:
            fn = has_coma[-1] + ' ' + fn[:-5]
        format_names[i] = fn

    return format_names


def generate_movies_uri_dbpedia_dataset():
    """
    Function that generates the dataset with movieId, title and uri from dbpedia by invoking, for every title of movie
    the function get_movie_uri_from_dbpedia. Because the API can cause timeout, the DBPedia API is called 10 times until
    it successfully works, if it does not, then the dataset is saved in disk with the current state of the dataset
    :return: A DataFrame with movieId, title and uri from dbpedia
    """

    movies = read_movie_info()
    movies_uri = pd.DataFrame(index=movies.index, columns=['uri'])

    print("Obtaining the URIs of the small MovieLens dataset")
    for index, row in movies.iterrows():
        names = __get_movie_strings(row[0])
        for name in names:
            uri_name = ""
            n = 0
            while True:
                try:
                    uri_name = from_dbpedia.get_movie_uri_from_dbpedia(name)
                except Exception as e:
                    n = n + 1
                    print("n:" + str(n) + " Exception: " + str(e))

                    if n == 10:
                        full_movies = pd.concat([movies, movies_uri], axis=1)
                        full_movies.to_csv(db_uri_ml_small_path, index=True)
                        break

                    continue
                break

            if uri_name != "":
                movies_uri.loc[index] = uri_name
                print("id: " + str(index) + " uri: " + str(uri_name))
                break

    full_movies = pd.concat([movies, movies_uri], axis=1)
    full_movies.to_csv(db_uri_ml_small_path, index=True)
    print("Finished Obtaining the URIs of the small MovieLens dataset")


def generate_recovery_uri_dbpedia():
    """
    Function that generates the dataset OF NAN uris from the generate_movies_uri_dbpedia_dataset with movieId, title and
    uri from dbpedia by invoking, for every title of movie
    the function get_recovery_movie_uri. Because the API can cause timeout, the DBPedia API is called 10 times until
    it successfully works, if it does not, then it goes to the next movie. A manual check to the dataset generated from
    this method MUST be done due to the fact that the SPARQL query of the invoked function is 50% accurate only.
    :return: A DataFrame with movieId, title and uri from dbpedia
    """
    movies = read_uri_info()
    links = read_links_info()
    movies_nan_uri = movies[movies['uri'].isna()]

    print("Obtaining the NaN URIs of the small MovieLens dataset")
    for index, row in movies_nan_uri.iterrows():
        prev_names = __get_movie_strings(row[0])
        names = prev_names
        imdbid = links.loc[index, 'imdbId']

        for name in prev_names:
            s1 = name.split("-")
            s2 = name.split(": ")
            if len(s1) > 1:
                names = names + s1
            if len(s2) > 1:
                names = names + s2

        for name in names:
            uri_name = ""
            n = 0
            while True:
                try:
                    uri_name = from_dbpedia.get_recovery_movie_uri(name, format(imdbid, '07d'))
                except Exception as e:
                    n = n + 1
                    print("n:" + str(n) + " Exception: " + str(e))
                    if n == 10:
                        break

                    continue
                break

            if uri_name != "":
                movies_nan_uri.at[index, 'uri'] = uri_name
                print("id: " + str(index) + " uri: " + str(uri_name) + " movie name: " + movies_nan_uri.loc[
                    index, 'title'])
                break
            else:
                print("id: " + str(index) + " URI not found for name \"" + name + "\"")

    movies_nan_uri.to_csv("./generated_files/dbpedia/nan_uri_dbpedia_movielens_small.csv", index=True)
    print("Finished Obtaining the NaN URIs of the small MovieLens dataset")


def merge_dbpedia_dataset():
    """
    Merges the dataset obtained from the two sparql queries, of the generate_movies_uri_dbpedia_dataset and
    generate_recovery_uri_dbpedia functions, with a 97% of small movie lens coverage
    :return: final dataset with movieid, movie title and uri of dbpedia
    """
    uris = read_uri_info()
    nan_uris = read_nan_info()

    final = uris.fillna(nan_uris)
    final.to_csv(db_final_ml_small_path, index=True)


def extract_wikidata_prop():
    """
    Obtain all the relevant triples of the movies from the wikidata and output the percentage of coverage from all the
    movies on the dataset
    :return: a csv file with all properties related to the movies form the latest small movielens dataset
    """

    # read movies link dataset and add the full imdbid column that matches with the wikidata format "ttXXXXXXX"
    all_movies = read_movie_info()
    links = read_links_info()
    s_all_movies = len(all_movies)
    links['full_imdbId'] = links['imdbId'].apply(lambda x: "tt" + str(format(x, '07d')))

    # create output, final dataframe with all properties of movies
    all_movie_props = pd.DataFrame(columns=['movieId', 'title', 'prop', 'obj'])

    # obtaind properties of movies in 300 movies batches
    begin = 0
    end = 350
    total = len(links)

    # Obtain data from wikidata
    print("Start obtaining movie data")
    while end <= total:
        results = from_wikidata.get_movie_data_from_wikidata(links.iloc[begin:end])
        all_movie_props = all_movie_props.append(results)
        print("From " + str(begin) + " to " + str(end - 1) + " obtained from Wikidata")
        begin = end
        end = end + 300
        time.sleep(60)
    print("End obtaining movie data")

    # save output
    all_movie_props.to_csv(wikidata_props_ml_small, mode='w', header=True, index=False)
    print("Coverage: " + str(len(all_movie_props['movieId'].unique())) + " obtained of " + str(s_all_movies)
          + ". Percentage: " + str(len(all_movie_props['movieId'].unique()) / s_all_movies))
    print('Output file generated')


def obtain_dbpedia_props():
    """
    Function that obtains the properties of all movies from the dbpedia
    :param movies_set: data set of movies with columns movie id and movie dbpedia uri
    :param cols: columns of data frame with all movie properties
    :return: a data frame with all movie properties
    """
    movies_uri_set = read_final_uri_dbpedia_dataset()
    movies_uri_set = movies_uri_set[movies_uri_set['uri'].notnull()]
    all_movie_props = pd.DataFrame(columns=['movie_id', 'prop', 'obj'])
    for movie_id, row in movies_uri_set.iterrows():
        movie_uri = row[1]
        n = 0
        while True:
            try:
                props_list_dic = from_dbpedia.get_props_of_movie_from_dbpedia(movie_id, movie_uri)
                all_movie_props = all_movie_props.append(props_list_dic, ignore_index=True)
                print("Obtained data of movie: " + str(movie_id))
            except Exception as e:
                print(e)
                n = n + 1
                if n == 10:
                    print("Number of tries exceeded, the file will be saved...")
                    all_movie_props.to_csv(dbpedia_props_ml_small, mode='w', header=True, index=False)
                    return
            break

    all_movie_props.to_csv(dbpedia_props_ml_small, mode='w', header=True, index=False)


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
