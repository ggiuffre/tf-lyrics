# tf-lyrics

Generate intriguing lyrics with TensorFlow and an internet connection.

This Python API allows you to easily select the lyrics of certain artists
on genius.com and train a deep network to generate text that sounds similar
to those lyrics.

Example:

```python
g = Genius()
artists = ['Bob Dylan', 'Tim Buckley', 'The Beatles']
text = g.get_artists_lyrics(artists, songs_per_artist=15)

p = Poet()
p.train_on(text, n_epochs=30)
poem = p.generate(start_string='Hey ', n_gen_chars=2000)
print(poem)
```

A `Genius` object is just a proxy to the
[Genius API](https://genius.com/developers), while a `Poet` object is a
wrapper to a recurrent TensorFlow model.

Work in progress.
