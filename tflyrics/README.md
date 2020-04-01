# Source code of `tflyrics`

The source code of `tflyrics` comprises the following main classes:

* `Poet` is a simplified interface to a TensorFlow model that can be trained
generate text.
to predict a text at the character level, and can subsequently be used to
* `LyricsGenerator` is an adapter between a `Genius` object and a
`tf.data.Dataset` object.
* `Genius` is a proxy to the Genius API.
* `TextProvider` in an abstract proxy to a collection of text resources.
Each `Genius` object is also a `TextProvider`.
