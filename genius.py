import os, requests, random
from bs4 import BeautifulSoup



class Genius:
    """A proxy to the Genius API.

    A Genius object is a proxy to access the Genius song lyrics API.
    Each Genius object has a token with which it can access the API,
    and provides instance methods to query the API in different ways,
    each time using its access token.

    Getting the lyrics of a song or set of songs, getting the songs by
    an artist, and getting an artist's unique ID on the API are among the
    methods that a Genius object offers.
    """

    def __init__(self, token: str = None, api_url: str = None):
        """Create a Genius object.

        Create a Genius object, endowed with a token that allows it to
        access the genius.com API. If a token is not provided, the Genius
        object will attempt to get the token from the environment variable
        'GENIUS_ACCESS_TOKEN'. This variable can be set by adding "export GENIUS_ACCESS_TOKEN='<token here>'" to your .bashrc file, for
        example. For generality, a custom URL to the Genius API can be
        provided (with the 'api_url' argument); the default is obviously
        'https://api.genius.com'.

        :param token: a token to access the Genius API
        :param api_url: a string with the location of the Genius API
        """

        self.token = token or os.environ['GENIUS_ACCESS_TOKEN']
        self.api_url = api_url or 'https://api.genius.com'

    def request(self, endpoint: str = '/', params: dict = {}) -> dict:
        """Get the response to a request sent to the Genius API.

        Get the response of the Genius API to an HTTP GET request. The response
        is a dictionary. The GET request is sent to an optionally specified
        end-point, with optional query parameters.

        :param endpoint: the endpoint of the request to the Genius API
        :param params: a dictionary containing parameters to the request
        :return: a dictionary representing the response of the API
        """

        headers = {'Authorization': 'Bearer ' + self.token}
        complete_url = self.api_url + endpoint
        response = requests.get(complete_url, params=params, headers=headers)
        response = response.json()

        return response

    def get_artist_id(self, artist_name: str) -> int:
        """Get the Genius ID of an artist.

        Get the unique identifier of an artist in the Genius database, by
        providing the name of the artist. Returns -1 if no match can be
        found.

        :param artist_name: the name of the artist
        :return: an integer that uniquely identifies the artist
        """

        params = {'q': artist_name}
        hits_response = self.request('/search', params)

        artist_id = -1

        if hits_response['meta']['status'] == 200:
            for hit in hits_response['response']['hits']:
                hit_artist = hit['result']['primary_artist']['name'].lower()
                artist_name = artist_name.lower()
                name_parts = Genius.artist_name_parts(artist_name)
                if (artist_name in hit_artist
                    or any(name in hit_artist for name in name_parts)):
                    artist_id = hit['result']['primary_artist']['id']
                    break

        return artist_id

    @staticmethod
    def artist_name_parts(complex_name: str, min_length: int = 10) -> list:
        """Get the names of artists that have been joined into a bigger name.

        Get a list containing the names of artists that have been joined to
        form a larger, composite name: for example get ['Sting', 'The Police']
        from 'Sting and The Police', or from 'Sting & The Police'. An optional
        minimum length can be provided, under which no attempts are made to
        find artist names inside the composite name (which is then assumed to
        be a non-composite name that happens to contain special characters).

        :param complex_name: composite name (name containing multiple names)
        :min_length: the minimum length that a name must have to be split
        :return: the (sorted) list of artists that form a composite name
        """

        parts = [complex_name]

        if len(complex_name) > min_length:
            parts += (
                  [name.strip() for name in complex_name.split('&')]
                + [name.strip() for name in complex_name.split('and')]
                + [name.strip() for name in complex_name.split('/')]
                + [name.strip() for name in complex_name.split('-')])

        return sorted(set(parts))

    def popular_songs(self, artist_name: str, n_songs: int = 10) -> int:
        """Generate the IDs of popular songs by a certain artist.

        One at a time, yield the unique identifiers of popular songs by a
        certain artist, given the artist's name and the amount of songs to
        be retrieved.

        :param artist_name: the name of an artist
        :param n_songs: the amount of songs to be retrieved
        :yield: an integer that uniquely identifies the song
        """

        artist_id = self.get_artist_id(artist_name)
        endpoint = '/artists/{id}/songs'.format(id=artist_id)

        per_page = 10
        page_num = 0
        songs_seen = 0
        songs_chosen = 0
        titles = []

        while songs_chosen < n_songs:
            page_num += 1
            data = {
                'sort': 'popularity',
                'page': page_num,
                'per_page': per_page}

            songs_response = self.request(endpoint, data)
            limit = min(page_num * per_page, n_songs) - songs_chosen
            for song in songs_response['response']['songs'][:limit]:
                songs_seen += 1
                title = song['title']
                if title not in titles:
                    titles.append(title)
                    songs_chosen += 1
                    yield song['id']

            if songs_response['response']['next_page'] is None:
                return

    def get_song_lyrics(self, song_id: int) -> str:
        """Get the lyrics of a song identified by a string.

        Get the lyrics of a song, by providing its identifier on Genius.

        :param song_id: the identifier of a song on Genius
        :return: the lyrics of that song
        """

        lyrics = ''

        endpoint = '/songs/' + str(song_id)
        song_response = self.request(endpoint, {})

        if song_response['meta']['status'] == 200:
            song_path = song_response['response']['song']['path']

            if song_path is not None:
                # get the HTML of the song's web page:
                page_url = 'http://genius.com' + song_path
                song_page = requests.get(page_url)
                html = BeautifulSoup(song_page.text, 'html.parser')

                # remove script tags that pollute the lyrics:
                [s.extract() for s in html('script')]

                # find the 'lyrics' tag:
                lyrics = html.find('div', class_='lyrics').get_text()

        return lyrics

    def get_artists_lyrics(self, artists: list, per_artist: int = 10) -> str:
        """Get the most popular lyrics by specific artists.

        Get a string containing the most popular lyrics by a specified set of
        artists, shuffled and concatenated. The artists are specified by their
        names in a list, and the number of songs to be retrieved for each
        artist is specified as an integer.

        :param artists: a list of artist names
        :param per_artist: the number of songs to be retrieved for each name
        :return: a string containing lyrics by those artists
        """

        songs = []

        # gather the IDs of songs to be downloaded:
        for artist in artists:
            for s in self.popular_songs(artist, per_artist):
                print('Adding {} by {}'.format(s, artist))
                songs.append(s)

        text = ''

        # gather the actual lyrics of the songs:
        random.shuffle(songs)
        for s in songs:
            print('Downloading {}'.format(s))
            lyrics = self.get_song_lyrics(s)
            text += Genius.clean_unicode(lyrics)

        return text

    @staticmethod
    def clean_unicode(text):
        """Clean a unicode text string."""

        text = text.replace('  ', ' ')
        text = text.replace('‘', '\'')
        text = text.replace('’', '\'')
        text = text.replace('“', '"')
        text = text.replace('”', '"')
        text = text.replace(' – ', ' - ')
        text = text.replace('–', ' - ')
        text = text.replace(' — ', ' - ')
        text = text.replace('—', ' - ')
        text = text.replace('\u200b', '')
        text = text.replace('…', '...')

        return text
