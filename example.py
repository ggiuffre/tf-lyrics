from tflyrics import Poet, LyricsGenerator

artists = ['Bob Dylan', 'Tim Buckley', 'The Beatles']
gen = LyricsGenerator(artists, per_artist=5)

p = Poet(batch_size=16)
p.train_on(gen, n_epochs=10)
poem = p.generate(start_string='Hey ', n_gen_chars=1000)
print(poem)
