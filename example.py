from genius import Genius
from poet import Poet

g = Genius()
artists = ['Bob Dylan', 'Tim Buckley', 'The Beatles']
text = g.get_artists_lyrics(artists, per_artist=15)

p = Poet()
p.train_on(text, n_epochs=10)
poem = p.generate('Hey ', n_gen_chars=1000)
print(poem)
