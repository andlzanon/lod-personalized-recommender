import pandas as pd
from preprocessing import dbpedia_utils as from_dbpedia


def read_movie_info():
    return pd.read_csv("./datasets/ml-latest-small/movies.csv", usecols=['movieId', 'title'])


def generate_movies_uri_dbpedia_dataset():
    movies = read_movie_info()
    all_movies_names = movies['title']
    movies_uri = []

    i = 1
    total = len(all_movies_names)
    print("Obtaining the URIs of the small MovieLens dataset")
    for name in all_movies_names:
        print(str(i) + " of " + str(total) + " movies")
        name = name.split(" (")[0]
        movies_uri.append(from_dbpedia.get_movie_uri_from_dbpedia(name))
        i += 1

    movies['dbpedia_uri'] = movies_uri
    movies.to_csv("../generated_files/dbpedia/uri_dbpedia_movielens_small", index=False)
    print("Finished Obtaining the URIs of the small MovieLens dataset")