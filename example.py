import os
import genius
import poet

token = os.environ['GENIUS_ACCESS_TOKEN']
g = genius.Genius(token)
artists = ['Bob Dylan', 'Bob Marley', 'The Beatles']
text = g.get_artists_lyrics(artists, songs_per_artist=2)

p = poet.Poet()
p.train_on(text, n_epochs=3)
poem = p.generate(u'Hey ')
print(poem)
