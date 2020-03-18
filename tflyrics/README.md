# tflyrics codebase

The codebase of `tflyrics` comprises the following main classes:

* `Genius` is a proxy to the Genius API.
* `LyricsGenerator` is an adapter between a `Genius` object and a
`tf.data.Dataset` object.
* `Poet` is a simplified interface to a TensorFlow model that can be trained
to predict a text at the character level, and can subsequently be used to
generate text.
