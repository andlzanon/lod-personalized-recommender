import numpy as np
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON


def get_movie_data_from_wikidata(slice_movie_set: pd.DataFrame):
    """
    Function that consults the wikidata KG for a slice of the movies set
    :param slice_movie_set: slice of the movie data set with movie id as index and imdbId, Title, year and imdbUrl
    as columns
    :return: JSON with the results of the query
    """
    imdbIdList = slice_movie_set['full_imdbId'].to_list()

    imdbs = ""
    for i in range(0, len(imdbIdList)):
        imdbId = imdbIdList[i]
        imdbs += " ""\"""" + imdbId + """\" """

    endpoint_url = "https://query.wikidata.org/sparql"

    query = """SELECT DISTINCT
      ?itemLabel
      ?propertyItemLabel
      ?valueLabel ?imdbId
    WHERE 
    {
      ?item wdt:P345 ?imdbId .
      ?item ?propertyRel ?value.
      VALUES ?imdbId {""" + imdbs + """} .
      ?propertyItem wikibase:directClaim ?propertyRel .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } .
      FILTER( 
        ?propertyRel = wdt:P1476 || ?propertyRel = wdt:P179 || ?propertyRel = wdt:P915 ||
        ?propertyRel = wdt:P136 || ?propertyRel = wdt:P170 ||  ?propertyRel = wdt:P495 || 
        ?propertyRel = wdt:P57 || ?propertyRel = wdt:P58 || ?propertyRel = wdt:P161 ||
        ?propertyRel = wdt:P725 || ?propertyRel = wdt:P1431 ||  ?propertyRel = wdt:P1040 ||
        ?propertyRel = wdt:P86 || ?propertyRel = wdt:P162 ||  ?propertyRel = wdt:P272 || 
        ?propertyRel = wdt:P344 || ?propertyRel = wdt:P166 || ?propertyRel = wdt:P1411 || 
        ?propertyRel = wdt:P2554 || ?propertyRel = wdt:P2515 || ?propertyRel = wdt:P840 ||
        ?propertyRel = wdt:P921 || ?propertyRel = wdt:P175
      )  
    }
    ORDER BY ?imdbId"""

    user_agent = "WikidataExplanationBotIntegration/1.0 https://www.wikidata.org/wiki/User:Andrelzan) " \
                 "wiki-bot-explanation-integration/1.0"

    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_dic = results_to_dict(slice_movie_set, results)
    return results_dic


def results_to_dict(slice_movie_set: pd.DataFrame, props_movie: dict):
    """
    Function that returns vector of dictionaries with the results from the dbpedia to insert into data frame of all
    movie properties
    :param slice_movie_set: slice of the movie data set with movie id as index and imdbId, Title, year and imdbUrl
    as columns
    :param props_movie: results of the returned result from the wikidata query
    :return: vector of dictionaries of the results of the wikidata query that can be appendable to a pandas df
    """

    # change the index to the imdbId in order to add the movie_id based on the imdb latter
    slice_movie_set.reset_index(level=0, inplace=True)
    slice_movie_set = slice_movie_set.set_index("full_imdbId")

    filter_props = []
    for line in props_movie["results"]["bindings"]:
        m_title = line["itemLabel"]["value"]
        m_prop = line["propertyItemLabel"]["value"]
        m_obj = line["valueLabel"]["value"]
        m_imdb = line["imdbId"]["value"]

        dict_props = {"movieId": slice_movie_set.loc[m_imdb, 'movieId'], "imdbId": m_imdb, "title": m_title, "prop": m_prop,
                      "obj": m_obj}
        filter_props.append(dict_props)

    return filter_props
