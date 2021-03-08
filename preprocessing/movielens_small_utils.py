import pandas as pd
from preprocessing import dbpedia_utils as from_dbpedia


def read_movie_info():
    return pd.read_csv("./datasets/ml-latest-small/movies.csv", usecols=['movieId', 'title']).set_index(['movieId'])


def __get_movie_strings(full_name: str):
    all_names = full_name.split(" (")
    format_names = [all_names[0]]
    if len(all_names) > 2:
        if all_names[1].find("a.k.a. ") != -1:
            format_names.append(all_names[1].split("a.k.a. ")[1][:-1])

    for i in range(0, len(format_names)):
        fn = format_names[i]
        has_coma = fn.split(", ")
        if len(has_coma) > 1:
            fn = has_coma[-1] + ' ' + ''.join(has_coma[:-1])
        format_names[i] = fn

    return format_names


def generate_movies_uri_dbpedia_dataset():
    movies = read_movie_info()
    movies_uri = pd.DataFrame(index=movies.index, columns=['uri'])

    print("Obtaining the URIs of the small MovieLens dataset")
    for index, row in movies.iterrows():
        names = __get_movie_strings(row[0])
        for name in names:
            while True:
                try:
                    uri_name = from_dbpedia.get_movie_uri_from_dbpedia(name)
                except:
                    continue
                break

            if uri_name != "":
                movies_uri.loc[index] = uri_name
                print("id: " + str(index) + " uri: " + str(uri_name))
                break

    full_movies = pd.concat([movies, movies_uri], axis=1)
    full_movies.to_csv("./generated_files/dbpedia/uri_dbpedia_movielens_small", index=True)
    print("Finished Obtaining the URIs of the small MovieLens dataset")
