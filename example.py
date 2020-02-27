import genius
import poet

# TODO remove token!
token = 'q_IbWwdY0HeRaaoD4Xr6q2nDFSktXbZ7dQ8uBkmPVhQ80RcfY6AV9aCTNatHCVlv'
g = genius.Genius(token)
artists = ['Bob Dylan', 'Bob Marley', 'The Beatles']
text = g.get_artists_lyrics(artists, songs_per_artist=2)

p = poet.Poet()
p.train_on(text, n_epochs=3)
poem = p.generate(u'Hey ')
print(poem)
