import os, requests
from bs4 import BeautifulSoup
from progress.bar import Bar



class Genius:
    """Interface of a user to the Genius API."""

    def __init__(self, token=None):
        """Create a Genius object.

        Create a Genius object, endowed with a token that allows it to
        access the genius.com API."""

        self.token = token or os.environ['GENIUS_ACCESS_TOKEN']

    def request(self, endpoint, query_data):
        """Get a JSON response from the genius.com API."""

        base_url = 'https://api.genius.com'
        headers = {'Authorization': 'Bearer ' + self.token}
        complete_url = base_url + endpoint
        response = requests.get(complete_url, params=query_data, headers=headers)
        response = response.json()

        return response

    def get_song_id(self, song_title, artist_name):
        """Get the Genius ID of a song by an artist."""

        query_data = {'q': song_title + ' ' + artist_name}
        hits_response = self.request('/search', query_data)

        song_id = None

        if hits_response['meta']['status'] == 200:
            for hit in hits_response['response']['hits']:
                hit_artist = hit['result']['primary_artist']['name'].lower()
                artist_name = artist_name.lower()
                name_parts = Genius.artist_name_parts(artist_name)
                if (artist_name in hit_artist
                    or any(name in hit_artist for name in name_parts)):
                    song_id = hit['result']['id']
                    break

        return song_id

    def get_artist_id(self, artist_name):
        """Get the Genius ID of an artist."""

        query_data = {'q': artist_name}
        hits_response = self.request('/search', query_data)

        artist_id = None

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
    def artist_name_parts(artist_name, min_length=10):
        """Get the names of artists that have been joined into a bigger name.

        For example get ['Sting', 'The Police'] from 'Sting and The Police',
        or from 'Sting & The Police'."""

        parts = []

        if len(artist_name) > min_length:
            parts = ([artist_name]
                + [name.strip() for name in artist_name.split('&')]
                + [name.strip() for name in artist_name.split('and')]
                + [name.strip() for name in artist_name.split('/')]
                + [name.strip() for name in artist_name.split('-')])

        return set(parts)

    def popular_songs(self, artist_name, n_songs=10):
        """Generate a certain number of popular songs by a certain artist."""

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

    def get_song_lyrics(self, song_id):
        """Get the lyrics of a song with a certain Genius ID."""

        lyrics = None

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

    def get_artists_lyrics(self, artists, songs_per_artist=10):
        """Get the most popular lyrics by specific artists."""

        text = ''
        for artist in artists:
            with Bar(artist, max=songs_per_artist) as bar:
                for s in self.popular_songs(artist, songs_per_artist):
                    bar.next()
                    lyrics = self.get_song_lyrics(s)
                    lyrics = lyrics.replace('‘', '\'')
                    lyrics = lyrics.replace('’', '\'')
                    lyrics = lyrics.replace('“', '"')
                    lyrics = lyrics.replace('”', '"')
                    lyrics = lyrics.replace(' – ', ' - ')
                    lyrics = lyrics.replace('–', ' - ')
                    lyrics = lyrics.replace(' — ', ' - ')
                    lyrics = lyrics.replace('—', ' - ')
                    lyrics = lyrics.replace('\u200b', '')
                    text += lyrics

        return text
