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
    results_dic = results_movies_to_dict(slice_movie_set, results)
    return results_dic


def get_entity_by_name(slice_artist_set: pd.DataFrame):
    endpoint_url = "https://query.wikidata.org/sparql"
    artist_set = slice_artist_set.copy()
    artist_set['lower_name'] = artist_set['name'].str.lower()
    artist_set = artist_set.set_index('lower_name')

    filter_sentence = "FILTER("
    for name in artist_set.index:
        filter_sentence = filter_sentence + """CONTAINS(LCASE(?name), \"""" + name + """\") || """
    filter_sentence = filter_sentence[:-4] + ")"

    query = """
    SELECT DISTINCT
      ?item ?name
    WHERE 
    {  
      {
          VALUES ?instance {wd:Q215380 wd:Q5741069 wd:Q56816954 wd:Q9212979 wd:Q9212979 wd:Q3736859 wd:Q6168416}.
          ?item wdt:P31 ?instance .
          ?item rdfs:label ?name .
          """ + filter_sentence + """} 
      UNION
      {
          VALUES ?professions {wd:Q177220 wd:Q639669 wd:Q488205}
          ?item wdt:P31 wd:Q5 .
          ?item wdt:P106 ?professions .
          ?item rdfs:label ?name .
          """ + filter_sentence + """} 
      
      FILTER(LANG(?name) = "en") 
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } .
   } ORDER BY ?name
   """

    user_agent = "WikidataExplanationBotIntegration/1.0 https://www.wikidata.org/wiki/User:Andrelzan) " \
                 "wiki-bot-explanation-integration/1.0"

    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    filter_props = []
    last = ""
    for line in results["results"]["bindings"]:
        try:
            wiki_id = line["item"]["value"]
            artist_name = line["name"]["value"]
            low_name = artist_name.lower()
            artist_id = int(artist_set.loc[low_name, 'id'])
            if last != low_name:
                print("Artist: " + str(artist_name) + " artist_id: " + str(artist_id) + " wiki_id: " + wiki_id)
                results_dic = {"wiki_id": wiki_id, "id": artist_id, "name": artist_name}
                filter_props.append(results_dic)
            last = low_name
        except KeyError:
            continue

    return filter_props


def get_artists_data_by_id_wikidata(slice_artist_set: pd.DataFrame):
    """
    Function that returns the data of artists based on their lastfm-id
    :param slice_artist_set: slice of the dataset with last-fm artists ids
    :returns dictionary with results
    """
    artist_set = slice_artist_set.copy()
    artist_set['lastfm_id'] = slice_artist_set['url'].str.split('/').str[-1]
    lastfm_list = artist_set['lastfm_id'].to_list()

    lastfm_ids = ""
    for i in range(0, len(lastfm_list)):
        id = lastfm_list[i]
        lastfm_ids += " ""\"""" + id + """\" """
        lowerid = id.lower()
        lastfm_ids += " ""\"""" + lowerid + """\" """

    endpoint_url = "https://query.wikidata.org/sparql"

    query = """
    SELECT DISTINCT
      ?itemLabel
      ?propertyItemLabel
      ?valueLabel ?lastfmId
    WHERE 
    {
      ?item wdt:P3192 ?lastfmId .
      ?item ?propertyRel ?value.
      VALUES ?lastfmId {""" + lastfm_ids + """} .
      ?propertyItem wikibase:directClaim ?propertyRel .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } .
      FILTER( 
        ?propertyRel = wdt:P19|| ?propertyRel = wdt:P1412 || ?propertyRel = wdt:P103 ||
        ?propertyRel = wdt:P101 || ?propertyRel = wdt:P463 || ?propertyRel = wdt:P937 ||
        ?propertyRel = wdt:P412 || ?propertyRel =  wdt:P69|| ?propertyRel = wdt:P140 ||
        ?propertyRel = wdt:P3828 || ?propertyRel =  wdt:P641 || ?propertyRel =  wdt:P710 ||
        ?propertyRel =  wdt:P571 ||  ?propertyRel =  wdt:P17 || ?propertyRel =  wdt:P740 || 
        ?propertyRel = wdt:P2031 || ?propertyRel =  wdt:P495|| ?propertyRel =  wdt:P1411 ||
        ?propertyRel =  wdt:P358 || ?propertyRel =  wdt:P136|| ?propertyRel =  wdt:P264 || 
        ?propertyRel =  wdt:P800 ||  ?propertyRel =  wdt:P737 || ?propertyRel =  wdt:P166 || 
        ?propertyRel =  wdt:P1875 ||  ?propertyRel =  wdt:P527|| ?propertyRel =  wdt:P21 || 
        ?propertyRel =  wdt:P27 || ?propertyRel =  wdt:P569|| ?propertyRel =  wdt:P106 || 
        ?propertyRel =  wdt:P1303 || ?propertyRel = wdt:P551 || ?propertyRel = wdt:P1037 )
    }
    ORDER BY ?lastfmId
    """

    user_agent = "WikidataBotIntegration/1.0 https://www.wikidata.org/wiki/User:Andrelzan) "

    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_dic = results_artists_to_dict(artist_set, results)
    return results_dic


def results_artists_to_dict(slice_artist_set, props_artists):
    slice_artist_set.reset_index(level=0, inplace=True)
    slice_artist_set = slice_artist_set.set_index("lastfm_id")

    filter_props = []
    for line in props_artists["results"]["bindings"]:
        m_title = line["itemLabel"]["value"]
        m_prop = line["propertyItemLabel"]["value"]
        m_obj = line["valueLabel"]["value"]
        m_lastfmid = line["lastfmId"]["value"]

        try:
            id = slice_artist_set.loc[m_lastfmid, 'id']
        except KeyError:
            sep = m_lastfmid.split("+")
            sep = [w.capitalize() for w in sep]
            lastfm_id = "+".join(sep)
            id = slice_artist_set.loc[lastfm_id, 'id']

        dict_props = {"id": id, "lastfm_id": m_lastfmid, "artist": m_title,
                      "prop": m_prop,
                      "obj": m_obj}
        filter_props.append(dict_props)

    return filter_props


def results_movies_to_dict(slice_movie_set: pd.DataFrame, props_movie: dict):
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

        dict_props = {"movieId": slice_movie_set.loc[m_imdb, 'movieId'], "imdbId": m_imdb, "title": m_title,
                      "prop": m_prop,
                      "obj": m_obj}
        filter_props.append(dict_props)

    return filter_props
