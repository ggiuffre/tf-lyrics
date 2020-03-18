# tf-lyrics

Generate intriguing lyrics with TensorFlow and an internet connection.

This Python API allows you to easily select the lyrics of certain artists
on [genius.com](https://genius.com/) and train a deep network to generate
text that sounds similar to those lyrics.

Example:

```python
g = Genius()
artists = ['Bob Dylan', 'Tim Buckley', 'The Beatles']
lyrics = g.artists_lyrics(artists, per_artist=15)
text = ''.join([l for l in lyrics])

p = Poet()
p.train_on(text, n_epochs=30)
poem = p.generate(start_string='Hey ', n_gen_chars=2000)
print(poem)
```

A `Genius` object is just a proxy to the Genius API, while a `Poet` object is a
wrapper to a recurrent TensorFlow model.

*work in progress*

Inspired by [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and [Text generation with an RNN](https://www.tensorflow.org/tutorials/text/text_generation)
