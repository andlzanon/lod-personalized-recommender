import pandas as pd
import time
import requests
from caserec.utils.split_database import SplitDatabase
from preprocessing import wikidata_utils as from_wikidata
import traceback

lastfm_path = "./datasets/hetrec2011-lastfm-2k/user_artists.dat"
artistis_path = "./datasets/hetrec2011-lastfm-2k/artists.dat"
artist_dbpedia_id = "./datasets/hetrec2011-lastfm-2k/mappingLinkedData.tsv"
user_artists = "./datasets/hetrec2011-lastfm-2k/user_artists.dat"
interactions = "./datasets/hetrec2011-lastfm-2k/interactions.csv"

artist_wiki_id = "./generated_files/wikidata/last-fm/artists_wiki_id.csv"
artist_prop = "./generated_files/wikidata/last-fm/props_artists_id.csv"
final_artist_uri = "./generated_files/wikidata/last-fm/final_artists_wiki_id.csv"


def read_artists() -> pd.DataFrame:
    """
    Function that read the file artists.dat from the last-fm dataset
    :return: df with file data
    """
    return pd.read_csv(artistis_path, sep='\t')


def read_dbpedia_uri() -> pd.DataFrame:
    """
    Function that reads the file containing the mapping from artist id to DPPedia uri.
    This dataset is with full credit to the Information Systems Lab @ Polytechnic University of Bari
    (https://github.com/sisinflab/LinkedDatasets/blob/master/last_fm/mappingLinkedData.tsv)
    :return:
    """
    df = pd.read_csv(artist_dbpedia_id, sep='\t')
    df.columns = ['id', 'uri']
    return df


def read_artists_uri() -> pd.DataFrame:
    """
    Function that read the file artists.dat from the last-fm dataset
    :return: df with file data
    """
    return pd.read_csv(artist_wiki_id, sep=',')


def read_final_artists_uri() -> pd.DataFrame:
    """
    Function that reads the file with the final uris, with the merge from the dbpedia and the obtained with the
    SPARQL query
    :return: df with file data
    """
    return pd.read_csv(final_artist_uri, sep=',')


def read_props_set() -> pd.DataFrame:
    """
    Function that reads the property file set
    :return: df property set
    """
    return pd.read_csv(artist_prop, sep=',')


def cross_validation_lasfm(rs: int):
    """
    Split the dataset into cross validation folders
    To read the file use the command: df = pd.read_csv("./datasets/hetrec2011-lastfm-2k/folds/0/test.dat", header=None)
    :param rs: random state integer arbitrary number
    :return: folders created on the dataset repository
    """
    SplitDatabase(input_file=interactions,
                  dir_folds="./datasets/hetrec2011-lastfm-2k/", as_binary=True, binary_col=2,
                  sep_read=',', sep_write=',', n_splits=10).k_fold_cross_validation(random_state=rs)


def user_artist_filter_interaction(n_inter: int, n_iter_flag=False):
    """
    Function that reduces the dataset to contain only the artists that we could obtain data from
    :param n_inter: minimum number of interactions for each user
    :param n_iter_flag: flag to filter or not by number of interactions
    :return: file
    """
    interac = pd.read_csv(user_artists, sep='\t')
    interac = interac.set_index('userID')
    props = read_props_set()

    filter_interactions = interac[interac['artistID'].isin(list(props['id'].unique()))]

    implicit = pd.DataFrame()
    if n_iter_flag:
        for u in filter_interactions.index.unique():
            u_set = filter_interactions.loc[u]
            if len(u_set) >= n_inter:
                implicit = pd.concat([implicit, u_set.reset_index()], ignore_index=True)

        implicit.reset_index()
        implicit.to_csv(interactions, header=None, index=False)

    filter_interactions = filter_interactions.reset_index()
    filter_interactions.to_csv(interactions, header=None, index=False)


def merge_uri():
    """
    Function that validates the and merges the artists uri obtained with the ones from the dbpedia obtained by the
    sisinf lab
    :return: final dataset with the wikidata uri
    """

    # create final dataset, the valid column indicates if the base extracted by the sisinf lab and by me match
    # and read datasets
    merge = pd.DataFrame(columns=['id', 'uri', 'valid'])
    wikidata_uris = read_artists_uri()
    wikidata_uris = wikidata_uris.set_index('id')

    dbpedia_uris = read_dbpedia_uri()
    dbpedia_uris = dbpedia_uris.set_index('id')

    artists = read_artists()
    total = len(artists)

    # set params and url for api call
    url = "https://www.wikidata.org/w/api.php"
    params = {'action': 'wbgetentities', 'sites': 'enwiki', 'format': 'json'}

    # get all unique ids in both datasets
    artists = list(set(list(dbpedia_uris.index.unique()) + list(wikidata_uris.index.unique())))
    for artist in artists:
        # try to get the wikidata id from the dbpedia and check validation in our base
        dict = {'id': artist}
        try:
            dbpedia_artist = dbpedia_uris.loc[artist]
            title = dbpedia_artist['uri'].split("/")[-1]
            params['titles'] = title
            req = requests.get(url=url, params=params)
            resp = req.json()
            wiki_entity = list(resp['entities'].keys())[0]
            valid = 1
            if wiki_entity != '-1':
                # if finds the first entity on our base validates, if don't find and there are more keys is not valid
                try:
                    if len(wikidata_uris[(wikidata_uris['wiki_id'] == 'http://www.wikidata.org/entity/' + wiki_entity)]) == 1:
                        valid = 1
                except KeyError:
                    if len(list(resp['entities'].keys())) > 1:
                        valid = 0
                    print("Wiki Id not found")

                dict['uri'] = wiki_entity
                dict['valid'] = valid
                print("id: " + str(dict['id']) + " uri: " + str(dict['uri']) + " valid: " + str(dict['valid']))
                merge = merge.append(dict, ignore_index=True)

            else:
                try:
                    wikidata_artist = wikidata_uris.loc[artist]['wiki_id']

                    if type(wikidata_artist) == str:
                        valid = 1
                        dict['uri'] = wikidata_artist.split("/")[-1]
                        dict['valid'] = valid
                        print("id: " + str(dict['id']) + " uri: " + str(dict['uri']) + " valid: " + str(dict['valid']))
                        merge = merge.append(dict, ignore_index=True)
                    else:
                        valid = 0
                        for uri in list(wikidata_artist):
                            dict['uri'] = uri.split("/")[-1]
                            dict['valid'] = valid
                            print("id: " + str(dict['id']) + " uri: " + str(dict['uri']) + " valid: " + str(
                                dict['valid']))
                            merge = merge.append(dict, ignore_index=True)
                # enters here if there is a DBPedia URI but there is not an wikidata id associated with it
                except KeyError:
                    print("Wikidata id not in both bases. Id: " + str(artist))

        # if not find the id in the dbpedia, get on our base. If there is only one line, tha wikidata is consistent
        # and the id is validated, if there is more than one line, the ids are not consistent and, therefore, not valid
        except KeyError:
            wikidata_artist = wikidata_uris.loc[artist]['wiki_id']

            if type(wikidata_artist) == str:
                valid = 1
                dict['uri'] = wikidata_artist.split("/")[-1]
                dict['valid'] = valid
                print("id: " + str(dict['id']) + " uri: " + str(dict['uri']) + " valid: " + str(dict['valid']))
                merge = merge.append(dict, ignore_index=True)
            else:
                valid = 0
                for uri in list(wikidata_artist):
                    dict['uri'] = uri.split("/")[-1]
                    dict['valid'] = valid
                    print("id: " + str(dict['id']) + " uri: " + str(dict['uri']) + " valid: " + str(dict['valid']))
                    merge = merge.append(dict, ignore_index=True)

        except requests.exceptions.ConnectionError:
            merge.to_csv(final_artist_uri, mode='w', header=True, index=False)
            print("Coverage: " + str(len(merge['id'].unique())) + " obtained of " + str(total)
                  + ". Percentage: " + str(len(merge['id'].unique()) / total))
            print('Output file generated')

        time.sleep(10)

    merge.to_csv(final_artist_uri, mode='w', header=True, index=False)
    print("Coverage: " + str(len(merge['id'].unique())) + " obtained of " + str(total)
          + ". Percentage: " + str(len(merge['id'].unique()) / total))
    print('Output file generated')


def extract_wikidata_prop():
    """
    Obtain all the relevant triples of the artists from the wikidata and output the percentage of coverage from all the
    movies on the dataset
    :return: a csv file with all properties related to the movies form the latest small movielens dataset
    """

    # read artists dataset
    artists = read_artists()
    n = len(artists['id'].unique())
    uris = read_final_artists_uri()

    artists_dict = artists.set_index('id').to_dict()
    uris['name'] = uris.apply(lambda x: artists_dict['name'][x['id']], axis=1)
    uris['lastfm_id'] = uris.apply(lambda x: artists_dict['url'][x['id']], axis=1)

    # create output, final dataframe with all properties of artists
    artists_props = pd.DataFrame(columns=['id', 'artist', 'prop', 'obj', 'wiki_id'])

    # obtaind properties of artists in 50 movies batches
    begin = 0
    step = 150
    end = begin + step
    total = len(uris)
    t = 0

    # Obtain data from wikidata
    print("Start obtaining artists data")
    while end <= total:
        try:
            results = from_wikidata.get_artists_data_by_id_wikidata(uris.iloc[begin:end])
            artists_props = artists_props.append(results, ignore_index=True)
            print("From " + str(begin) + " to " + str(end - 1) + " obtained from Wikidata")
            begin = end
            end = end + step
            t = 0
        except Exception as e:
            print("--- ERROR ---")
            print(e)
            traceback.print_exc()
            t = t + 1
            if t == 5:
                break
        time.sleep(30)

    artists_props['name'] = artists_props.apply(lambda x: artists_dict['name'][x['id']], axis=1)
    artists_props = artists_props.sort_values(by='id')
    # save output
    print("End obtaining movie data")
    artists_props.to_csv(artist_prop_lastid, mode='w', header=True, index=False)
    print("Coverage: " + str(len(artists_props['id'].unique())) + " obtained of " + str(n)
          + ". Percentage: " + str(len(artists_props['id'].unique()) / n))
    print('Output file generated')


def extract_artistis_wiki_id():
    # read movies link dataset and add the full imdbid column that matches with the wikidata format "ttXXXXXXX"
    artists = read_artists()
    total = len(artists)

    try:
        artists_id = read_artists_uri()
        last_name = str(artists_id.iloc[-1]['name']).lower()
        begin = artists.loc[(artists['name'].str.lower() == last_name)].index.values[0] + 1
    except Exception:
        artists_id = pd.DataFrame(columns=['wiki_id', 'id', 'name'])
        begin = 0

    print("Start obtaining artists data")
    step = 20
    end = begin + step
    print("begin: " + str(begin) + ", step: " + str(step))
    t = 0
    c = 0

    while end <= total:
        try:
            results = from_wikidata.get_entity_by_name(artists.iloc[begin:end])
            if len(results) > 0:
                artists_id = artists_id.append(results, ignore_index=True)
            print("From " + str(begin) + " to " + str(end - 1) + " obtained from Wikidata on t = " + str(t))
            begin = end
            end = end + step
            t = 0
            c = c + 1

        except Exception as e:
            print("#### ERROR ####")
            print(e)
            t = t + 1
            c = c + 1
            #traceback.print_exc()

        if c % step == 0:
            print("#### SAVING FILE ####")
            artists_id.to_csv(artist_wiki_id, mode='w', header=True, index=False)

        time.sleep(20)

    artists_id.to_csv(artist_wiki_id, mode='w', header=True, index=False)
    print("Coverage: " + str(len(artists_id['id'].unique())) + " obtained of " + str(total)
          + ". Percentage: " + str(len(artists_id['id'].unique()) / total))
    print('Output file generated')
