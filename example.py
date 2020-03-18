from tflyrics import Poet, LyricsGenerator

artists = ['Bob Dylan', 'Tim Buckley', 'The Beatles']
gen = LyricsGenerator(artists, per_artist=5)
ds = gen.as_dataset(batch_size=4)

p = Poet()
p.train_on(ds, n_epochs=10)
poem = p.generate(start_string='Hey ', n_gen_chars=1000)
print(poem)
