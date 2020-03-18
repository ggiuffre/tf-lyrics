# tflyrics

Generate intriguing lyrics with TensorFlow and an internet connection.

`tflyrics` is a Python package that allows you to easily select lyrics of
specific artists from [genius.com](https://genius.com/), and train a deep
neural network to generate text that sounds similar to those lyrics. This
work was inspired from [The Unreasonable Effectiveness of Recurrent Neural
Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and
[Text generation with an
RNN](https://www.tensorflow.org/tutorials/text/text_generation).

Example:

```python
artists = ['Bob Dylan', 'Tim Buckley', 'The Beatles']
gen = LyricsGenerator(artists, per_artist=5)
ds = gen.as_dataset(batch_size=4)

p = Poet()
p.train_on(ds, n_epochs=10)
poem = p.generate(start_string='Hey ', n_gen_chars=1000)
print(poem)
```

`LyricsGenerator` objects make it easy for you to create efficient data
pipelines that feed from the Genius API directly into a TensorFlow model. A
`Poet` object is a wrapper around a recurrent TensorFlow model.

Note that the Genius API requires you to have an **access token**. Without
that, `tflyrics` won't be able to get lyrics for you. You can get an access
token for free at [docs.genius.com](https://docs.genius.com/). Once you have
it you can either pass it under the `token` argument of a `LyricsGenerator`
constructor, or store it as en environment variable (with `export
GENIUS_ACCESS_TOKEN='<your token here>'`). `tflyrics` will detect this
environment variable automatically, if it exists.
