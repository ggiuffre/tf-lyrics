from genius import Genius
from poet import Poet

g = Genius()
artists = ['Bob Dylan', 'Tim Buckley', 'The Beatles']
text = ''.join([t for t in g.get_artists_lyrics(artists, per_artist=5)])

p = Poet()
p.train_on(text, n_epochs=10)
poem = p.generate('Hey ', n_gen_chars=1000)
print(poem)
