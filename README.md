# tf-lyrics

Generate intriguing lyrics with TensorFlow and an internet connection.

This Python API allows you to easily select the lyrics of certain artists
on [genius.com](https://genius.com/), and train a deep network to generate
text that sounds similar to those lyrics.

Example:

```python
artists = ['Bob Dylan', 'Tim Buckley', 'The Beatles']
gen = LyricsGenerator(artists, per_artist=5)
ds = gen.as_dataset(batch_size=4)

p = Poet()
p.train_on(ds, n_epochs=20)
poem = p.generate(start_string='Hey ', n_gen_chars=1000)
print(poem)
```

A `LyricsGenerator` is an object that makes it easy to create a flexible data
pipeline from the Genius API directly to your TensorFlow model. A `Poet`
object is a wrapper around a recurrent TensorFlow model.

*work in progress*

Inspired by [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and [Text generation with an RNN](https://www.tensorflow.org/tutorials/text/text_generation)
