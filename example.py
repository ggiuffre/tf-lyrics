from genius import Genius
from poet import Poet

g = Genius()
artists = ['Bob Dylan', 'Tim Buckley', 'The Beatles']
text = g.get_artists_lyrics(artists, songs_per_artist=15)

p = Poet()
p.train_on(text, n_epochs=30)
poem = p.generate(u'Hey ', n_gen_chars=2000)
print(poem)
