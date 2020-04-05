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

## You inject a `TextProvider` as a dependency to a `LyricsGenerator`

The basic pattern with which `tflyrics` is intended to be used is that of
dependency injection: a client delegates the creation of a service it needs
to another object (the injector). In this case:

* you are the injector
* the `LyricsGenerator` is the client/consumer
* a `TextProvider` is a dependency

In other words: a `LyricsGenerator` client needs a `TextProvider` object; there
can be many different classes that implement the `TextProvider` interface; so
the choice of which `TextProvider` to create is delegated to you, the end user,
before you create a `LyricsGenerator`.
