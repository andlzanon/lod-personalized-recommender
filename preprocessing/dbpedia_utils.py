from SPARQLWrapper import SPARQLWrapper, JSON


def get_movie_uri_from_dbpedia(movie_name: str):
    """
    Function the uri of a movie passed as parameter
    :param movie_name: name of the movie
    :return: uri of the dbpedia resource of the movie_name
    """

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
    SELECT DISTINCT ?uri WHERE
    {
        ?uri a dbo:Film .
        ?uri rdfs:label ?name .
        FILTER (REGEX(?name, "^""" + movie_name + """$|^""" + movie_name + """", "i"))
        FILTER (lang(?name) = 'en')
    } 
    """)

    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"][0].get("uri").get("value")
