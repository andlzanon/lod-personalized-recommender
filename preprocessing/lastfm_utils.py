import pandas as pd
import time
import traceback
import SPARQLWrapper
import urllib
from preprocessing import wikidata_utils as from_wikidata

lastfm_path = "./datasets/hetrec2011-lastfm-2k/user_artists.dat"
artistis_path = "./datasets/hetrec2011-lastfm-2k/artists.dat"

artist_prop_lastid = "./generated_files/wikidata/last-fm/props_artists_id.csv"
artist_wiki_id = "./generated_files/wikidata/last-fm/artists_wiki_id.csv"
artist_prop = "./generated_files/wikidata/last-fm/props_artists.csv"


def read_artists() -> pd.DataFrame:
    """
    Function that read the file artists.dat from the last-fm dataset
    :return: df with file data
    """
    return pd.read_csv(artistis_path, sep='\t')


def extract_wikidata_prop():
    """
    Obtain all the relevant triples of the artists from the wikidata and output the percentage of coverage from all the
    movies on the dataset
    :return: a csv file with all properties related to the movies form the latest small movielens dataset
    """

    # read artists dataset
    artists = read_artists()

    # create output, final dataframe with all properties of artists
    artists_props = pd.DataFrame(columns=['id', 'artist', 'prop', 'obj', 'lastfm_id'])

    # obtaind properties of artists in 50 movies batches
    begin = 0
    end = 50
    total = len(artists)

    # Obtain data from wikidata
    print("Start obtaining artists data")
    while end <= total:
        try:
            results = from_wikidata.get_artists_data_by_id_wikidata(artists.iloc[begin:end])
        except (SPARQLWrapper.SPARQLExceptions.URITooLong, urllib.error.HTTPError):
            end = end - 10
            results = from_wikidata.get_artists_data_from_id_wikidata(artists.iloc[begin:end])

        artists_props = artists_props.append(results)
        print("From " + str(begin) + " to " + str(end - 1) + " obtained from Wikidata")
        begin = end
        end = end + 50
        time.sleep(10)
    print("End obtaining movie data")

    # save output
    artists_props.to_csv(artist_prop_lastid, mode='w', header=True, index=False)
    print("Coverage: " + str(len(artists_props['id'].unique())) + " obtained of " + str(total)
          + ". Percentage: " + str(len(artists_props['id'].unique()) / total))
    print('Output file generated')


def extract_artistis_wiki_id():
    # read movies link dataset and add the full imdbid column that matches with the wikidata format "ttXXXXXXX"
    artists = read_artists()
    total = len(artists)
    artists_id = pd.DataFrame(columns=['wiki_id', 'id', 'name'])

    print("Start obtaining artists data")
    begin = 0
    step = 20
    end = begin + step

    while end <= total:
        t = 0
        while t < 3:
            try:
                results = from_wikidata.get_entity_by_name(artists.iloc[begin:end])

                artists_id = artists_id.append(results)
                print("From " + str(begin) + " to " + str(end - 1) + " obtained from Wikidata")
                begin = end
                end = end + step
                break

            except Exception:
                traceback.print_exc()
                t = t + 1
                end = end - 5

        time.sleep(20)

    artists_id.to_csv(artist_wiki_id, mode='w', header=True, index=False)
    print("Coverage: " + str(len(artists_id['id'].unique())) + " obtained of " + str(total)
          + ". Percentage: " + str(len(artists_id['id'].unique()) / total))
    print('Output file generated')


