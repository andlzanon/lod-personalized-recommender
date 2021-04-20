from SPARQLWrapper import SPARQLWrapper, JSON


def get_movie_uri_from_dbpedia(movie_name: str):
    """
    Function that returns the uri of a movie passed as parameter
    :param movie_name: name of the movie
    :return: uri of the dbpedia resource of the movie_name
    """

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql_query = """
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        PREFIX : <http://dbpedia.org/resource/>
        PREFIX dbpedia2: <http://dbpedia.org/property/>
        PREFIX dbpedia: <http://dbpedia.org/>       
        PREFIX dct:	<http://purl.org/dc/terms/> 
        PREFIX ns:<http://www.w3.org/ns/prov#>    
        SELECT DISTINCT ?uri ?label WHERE
        {
            {
                ?uri a dbo:Film .
                ?uri rdfs:label ?label .
                FILTER (REGEX(?label, "^""" + movie_name + """$|^""" + movie_name + """", "i"))
                FILTER (lang(?label) = 'en')
            }
            UNION 
            {
                ?uri a dbo:Film .
                ?uri rdfs:label ?label .
                FILTER (REGEX(?label, "^""" + movie_name + """$|^""" + movie_name + """", "i"))
            }
        } 
    """

    sparql.setQuery(sparql_query)

    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    try:
        uri = results["results"]["bindings"][0].get("uri").get("value")
    except IndexError:
        uri = ""
    return uri


def get_recovery_movie_uri(movie_name: str, imdbid: str):
    """
    Function that returns the uri of a movie and the imdbid of the same movie passed as parameter
    :param movie_name: name of the movie
    :param imdbid: imdbid of the movie
    :return: uri of the dbpedia resource of the movie_name
    """

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql_query = """
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX dc: <http://purl.org/dc/elements/1.1/>
            PREFIX : <http://dbpedia.org/resource/>
            PREFIX dbpedia2: <http://dbpedia.org/property/>
            PREFIX dbpedia: <http://dbpedia.org/>       
            PREFIX dct:	<http://purl.org/dc/terms/> 
            PREFIX ns:<http://www.w3.org/ns/prov#>    
            SELECT DISTINCT ?uri ?label ?abs WHERE
            {
                {
                    ?uri dbo:imdbId \"""" + imdbid + """\" 
                }
                UNION 
                {
                    ?uri a dbo:Film .
                    ?uri foaf:name ?label .
                    FILTER (REGEX(?label, "^""" + movie_name + """$|^""" + movie_name + """", "i"))
                }
                UNION
                {
                    ?uri a dbo:Film .
                    ?uri rdfs:comment	?abs .
                    FILTER (CONTAINS(?abs, \"""" + movie_name + """\"))
                }          
            } 
        """

    sparql.setQuery(sparql_query)

    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    try:
        uri = results["results"]["bindings"][-1].get("uri").get("value")
    except IndexError:
        uri = ""
    return uri


def get_props_of_movie_from_dbpedia(movie_id: int, movie_uri: str):
    """
    Function that obtains the tuples (properties, resources) of a movie form its' uri
    :param movie_id: id of the movie on the data set
    :param movie_uri: path to the movie resource from dbpedia
    :return: dict with properties of movies, along with its' movie id
    """

    print(movie_uri)

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery("""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    PREFIX : <http://dbpedia.org/resource/>
    PREFIX dbpedia2: <http://dbpedia.org/property/>
    PREFIX dbpedia: <http://dbpedia.org/>       
    PREFIX dct:	<http://purl.org/dc/terms/> 
    PREFIX ns:<http://www.w3.org/ns/prov#>    
    SELECT DISTINCT *
    WHERE { 
        <""" + movie_uri + """> ?prop ?obj.
        FILTER( 
            ?prop = dbo:cinematography || ?prop = dbo:director || ?prop = dbo:distributor || 
            ?prop = dbo:editing || ?prop = dbo:musicComposer || ?prop = dbo:producer || 
            ?prop = dbo:starring || ?prop = dct:subject || ?prop = foaf:name
        )   
    }
    """)

    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    list_results = results_to_dict(movie_id, results)
    return list_results


def results_to_dict(movie_id: int, props_movie: dict):
    """
    Function that returns vector of dictionaries with the results from the dbpedia to insert into data frame of all
    movie properties
    :param movie_id: movie id of the movie
    :param props_movie: properties returned from dbpedia
    :return: vector of dictionaries with the results from the dbpedia
    """
    filter_props = []
    for p in props_movie["results"]["bindings"]:
        dict_props = {'movie_id': movie_id, "prop": p["prop"]["value"], "obj": p["obj"]["value"]}
        filter_props.append(dict_props)

    return filter_props
