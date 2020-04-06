# Source code of `tflyrics`

The source code of `tflyrics` comprises the following main classes:

* `Poet` is a simplified interface to a TensorFlow model that can be trained
to predict a text at the character level, and can subsequently be used to
generate text.
* `LyricsGenerator` is an adapter between a `Genius` object (or any other
`TextProvider` object) and a TensorFlow dataset.
* `Genius` is a proxy to the Genius API.
* `TextProvider` in an abstract proxy to a collection of text resources.
A `Genius` object is also a `TextProvider`.

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

## You implement your own `TextProvider` if you want to scrape another web API

Genius is a web API that provides text organised as a set of songs. There are
of course other services that provide text as a set of resources, and you might
want to use one of that services instead of Genius. To this end, you just need
to implement your own concrete `TextProvider`, making sure that it has two
methods called `resources` and `get_text` that do what `TextProvider`
specifies (see the documentation
[page](https://ggiuffre.github.io/tf-lyrics/reference/textprovider.html)
for that class).
