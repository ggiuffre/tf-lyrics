from tflyrics.genius import Genius
import itertools
import pytest
import tensorflow as tf



def test_creation():
    """A Genius object can be instantiated, and all of its arguments are
    optional."""

    # create mock values for each argument to the Genius constructor:
    test_args = {
        'token': 'mock_access_token',
        'api_url': 'https://api.genius.com'
        }

    # verify that arguments to the constructor are all optional:
    args_as_tuple = zip(test_args.keys(), test_args.values())
    for l in range(len(test_args)):
        for args in itertools.combinations(args_as_tuple, l):
            p = Genius(**dict(args_as_tuple))
            assert isinstance(p, Genius)

def test_request():
    """A genius object can query the web API it has been created for, and
    can return responses from the API in the form of dictionaries."""

    g = Genius()
    endpoint = '/search'
    params = {'q': 'Abba'}
    resp_1 = g.request(endpoint, params)
    assert (resp_1 is None) or isinstance(resp_1['response']['hits'], list)

    endpoint = '/non-existing-endpoint'
    resp_2 = g.request(endpoint)
    assert resp_2 is None

def test_get_artist_id():
    """A Genius object can fetch the Genius ID of an existing artist, and
    returns -1 if no match could be found."""

    g = Genius()
    artist_id = g.get_artist_id('Abba')
    assert isinstance(artist_id, int) and artist_id >= 0

    artist_id = g.get_artist_id('tRqZDkVuppLYZcSplOOn')
    assert isinstance(artist_id, int) and artist_id == -1

def test_artist_name_parts():
    """A Genius object can return a list of artist names contained in a larger,
    composite artist name."""

    comp_name = 'Artist 1 & Band Number One'
    names = Genius.artist_name_parts(comp_name)
    assert 'Artist 1' in names
    assert 'Band Number One' in names
    assert all(n in comp_name for n in names)

    comp_name = 'John Smith and Sons'
    names = Genius.artist_name_parts(comp_name)
    assert 'John Smith' in names
    assert 'Sons' in names
    assert all(n in comp_name for n in names)

    comp_name = 'Someone & Someone Else'
    names = Genius.artist_name_parts(comp_name, min_length=50)
    assert names == [comp_name]

def test_resources():
    """A Genius object can provide a generator of popular songs by a
    specified artist, where each element is the Genius ID of one song by that
    artist."""

    g = Genius()
    for song in g.resources('Maroon 5'):
        assert isinstance(song, int)

    n_songs = 4
    gen = g.resources('James Blake', n_songs=n_songs)
    assert len(list(gen)) <= n_songs

    gen = g.resources('tRqZDkVuppLYZcSplOOn')
    assert len(list(gen)) == 0

def test_get_text():
    """A Genius object can return a string containing the lyrics of a song,
    given the Genius ID of the song."""

    g = Genius()
    song_id = 262016
    lyrics = g.get_text(song_id)
    assert isinstance(lyrics, str)

    song_id = tf.constant(song_id)
    lyrics = g.get_text(song_id)
    assert isinstance(lyrics, str)
