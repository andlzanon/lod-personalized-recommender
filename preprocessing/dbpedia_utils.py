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
