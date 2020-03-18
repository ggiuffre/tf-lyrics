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
