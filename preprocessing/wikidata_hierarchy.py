import numpy as np
import pandas as pd
import time
import re
from SPARQLWrapper import SPARQLWrapper, JSON

wikidata_hierarchy_props_ml_small = "../generated_files/wikidata/props_hierarchy_wikidata_movielens_small.csv"
wikidata_hierarchy_props_lastfm_small = "../generated_files/wikidata/last-fm/props_hierarchy_wikidata_lastfm_small.csv"


def get_movie_hierarchy_data_from_wikidata(slice_prop_list: list):
    """
    Function that consults the wikidata KG for a slice of the movies set
    :param slice_prop_list: list of the properties to extract the super properties
    :return: JSON with the results of the query
    """
    items_label = ""
    for i in range(0, len(slice_prop_list)):
        ilabel = slice_prop_list[i]
        name_t = ilabel.translate({ord(c): " " for c in "!@#$^\"*()[]{};:,./<>?\|`~-=_+"})
        name_s = re.sub("([^\x00-\x7F])+", " ", ilabel)
        if ilabel == name_s and ilabel == name_t:
            items_label += " ""\"""" + ilabel + """\"@en """

    endpoint_url = "https://query.wikidata.org/sparql"

    query = """SELECT DISTINCT ?item ?itemLabel ?superClassLabel ?hyperClassLabel
            WHERE {
               ?item rdfs:label ?itemLabel .
               ?item wdt:P279 ?superClass .
               ?superClass wdt:P279 ?hyperClass .
                VALUES ?itemLabel {""" + items_label + """} .
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }ORDER BY ?item"""

    user_agent = "WikidataExplanationBotIntegration/1.0 https://www.wikidata.org/wiki/User:Andrelzan) " \
                 "wiki-bot-explanation-integration/1.0"

    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_dic = results_movies_to_dict(results)
    return results_dic


def results_movies_to_dict(props_movie: dict):
    """
    Function that returns vector of dictionaries with the results from the wikidata to insert into data frame of all
    movie properties
    :param props_movie: results of the returned result from the wikidata query
    :return: vector of dictionaries of the results of the wikidata query that can be appendable to a pandas df
    """

    # change the index to the imdbId in order to add the movie_id based on the imdb latter
    filter_props = []
    for line in props_movie["results"]["bindings"]:
        item_label = line["itemLabel"]["value"]
        super_label = line["superClassLabel"]["value"]
        hyper_label = line["hyperClassLabel"]["value"]

        dict_props1 = {"obj": item_label,
                       "super_obj": super_label}
        dict_props2 = {"obj": super_label,
                       "super_obj": hyper_label}

        filter_props.append(dict_props1)
        filter_props.append(dict_props2)

    return filter_props


def extract_wikidata_prop_hierarchy(prop_set):
    """
    Obtain all the relevant triples of the movies properties and super properties from the wikidata
    :return: a csv file with all properties related to the movies form the latest small movielens dataset
    """

    # read properties from all movies
    all_props = list(prop_set['obj'].unique())

    # create output, final dataframe with all properties and super properties of movies
    all_movie_props = pd.DataFrame(columns=['obj', 'super_obj'])

    # obtaind properties of movies in 200 movies batches
    begin = 0
    end = 200
    total = len(all_props)

    # Obtain data from wikidata
    print("Start obtaining movie data")
    while end <= total:
        results = get_movie_hierarchy_data_from_wikidata(all_props[begin:end])
        all_movie_props = all_movie_props.append(results)
        all_movie_props = all_movie_props.drop_duplicates()
        print("From " + str(begin) + " to " + str(end - 1) + " obtained superprops from Wikidata")
        begin = end
        end = end + 200
        time.sleep(60)
    print("End obtaining movie data")

    # save output
    all_movie_props.to_csv(wikidata_hierarchy_props_lastfm_small, mode='w', header=True, index=False)
    print('Output file generated')


# prop_set = pd.read_csv("../generated_files/wikidata/props_wikidata_movielens_small.csv")
prop_set = pd.read_csv("../generated_files/wikidata/last-fm/props_artists_id.csv")
extract_wikidata_prop_hierarchy(prop_set)
