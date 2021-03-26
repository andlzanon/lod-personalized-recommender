import pandas as pd
from preprocessing import dbpedia_utils as from_dbpedia

ml_small_path = "./datasets/ml-latest-small/movies.csv"
link_ml_small_path = "./datasets/ml-latest-small/links.csv"
uri_ml_small_path = "./generated_files/dbpedia/uri_dbpedia_movielens_small.csv"
nan_ml_small_path = "./generated_files/dbpedia/nan_uri_dbpedia_movielens_small.csv"
final_ml_small_path = "./generated_files/dbpedia/final_uri_dbpedia_movielens_small.csv"


def read_movie_info():
    """
    Function that reads the name of the movies of the small movielens dataset
    :return: pandas DataFrame with the movieId column as index and title as value
    """
    return pd.read_csv(ml_small_path, usecols=['movieId', 'title']).set_index(['movieId'])


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
    return pd.read_csv(uri_ml_small_path,).set_index(['movieId'])


def read_nan_info():
    """
    Function that reads the name of the movies of the small movielens dataset with the uri from
    the generate_recovery_uri_dbpedia function
    :return: pandas DataFrame with the movieId column as index and title as value
    """
    return pd.read_csv(nan_ml_small_path).set_index(['movieId'])


def read_final_sml_dbpedia_dataset():
    """
    Function that reads the name of the movies of the small movielens dataset
    :return: pandas DataFrame with the movieId column as index and title as value
    """
    return pd.read_csv(final_ml_small_path).set_index(['movieId'])


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
                        full_movies.to_csv(uri_ml_small_path, index=True)
                        break

                    continue
                break

            if uri_name != "":
                movies_uri.loc[index] = uri_name
                print("id: " + str(index) + " uri: " + str(uri_name))
                break

    full_movies = pd.concat([movies, movies_uri], axis=1)
    full_movies.to_csv(uri_ml_small_path, index=True)
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
                print("id: " + str(index) + " uri: " + str(uri_name) + " movie name: " + movies_nan_uri.loc[index, 'title'])
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
    final.to_csv(final_ml_small_path, index=True)