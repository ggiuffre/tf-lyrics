import tensorflow as tf
from tflyrics.genius import Genius
from tflyrics.constants import default_vocab



class LyricsGenerator:
    """An adapter between the Genius API and a TF Dataset.

    A LyricsGenerator object queries the Genius API and provides a TensorFlow
    Dataset that can be fed to a model for training. Each sample of the dataset
    is a sequence of unicode characters taken from song lyrics. More precisely,
    each example in a LyricsGenerator is a character sequence that maps to a
    sequence with same length but shifted content: for example, there could be
    a LyricsGenerator where "Hello wo" maps to "ello wor", and "ello wor" maps
    to "llo worl".
    """

    def __init__(self, artists: list = [], per_artist: int = 5,
        vocabulary: list = None, token: str = None):
        """Create a LyricsGenerator object.

        Create a LyricsGenerator object that will provide lyrics from a
        specified set of artists, and will filter those lyrics to only include
        characters from a vocabulary of specified unicode characters.

        :param artists: list of artists whose songs should be included
        :param per_artist: number of songs to include per artist
        :param vocabulary: the unicode characters accepted by the object
        :param token: a token to access the Genius API
        """

        # declare what characters the dataset can accept:
        self.vocabulary = vocabulary or default_vocab
        keys_tensor = tf.constant(self.vocabulary)
        vals_tensor = tf.range(tf.size(keys_tensor))
        self.char2idx = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)

        # create a Genius object to fetch lyrics:
        self.genius = Genius(token)

        # mark which songs should be downloaded:
        self.songs = []
        for a in artists:
            for s in self.genius.popular_songs(a, per_artist):
                print('LyricsGenerator:',
                    'adding {} by {} to wishlist'.format(s, a))
                self.songs.append(s)

        # shuffle the list of songs to be downloaded:
        self.songs = tf.random.shuffle(self.songs)

    def as_dataset(self, batch_size: int = None, seq_length: int = 100) -> tf.data.Dataset:
        """Get a TensorFlow dataset equivalent to this object.

        Get a TensorFlow dataset whose samples are substrings of song lyrics
        provided by a Genius object. More specifically, each sample in the
        database is a pair of substrings that have the same size but are
        shifted with respect to each other: e.g. [("Hello", "ello "), ("ello ",
        "llo W"), ("llo W", "lo Wo"), ("lo Wo", "o Wor"), ("o Wor", " Worl"),
        ...].

        :param batch_size: batch size of the dataset
        :param seq_length: length of each substring that forms a sample
        :return: a TensorFlow Dataset of substrings
        """

        # create a dataset of song titles (IDs, actually):
        songs_dataset = tf.data.Dataset.from_tensor_slices(self.songs)

        # create a function that maps song IDs to song lyrics:
        def get_song_lyrics(s):
            l = tf.py_function(self.genius.get_song_lyrics, [s], tf.string)
            return tf.data.Dataset.from_tensors(tf.reshape(l, ()))

        # create a dataset of song lyrics, where each sample is a song:
        lyrics_dataset = songs_dataset.interleave(
            get_song_lyrics,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
            ).filter(
            lambda x: tf.size(tf.strings.bytes_split(x)) > 0
            )

        # program the dataset to be cached as soon as possible:
        cached_dataset = lyrics_dataset.cache()

        # get a dataset where each sample is now a substring of song lyrics:
        processed_dataset = cached_dataset.map(
            lambda x: self.preprocess(x, seq_length)
            ).unbatch()

        # map each substring to the substring of successor characters:
        split_input_label = lambda chunk: (chunk[:-1], chunk[1:])
        dataset = processed_dataset.map(split_input_label)

        # optionally batch the dataset:
        if batch_size is not None:
            dataset = dataset.batch(batch_size, drop_remainder=True)

        return dataset

    @tf.function
    def preprocess(self, text: str, seq_length: int = 100) -> list:
        """Split the lyrics of a song into multiple substrings.

        Preprocess a string containing lyrics, extracting substrings that
        have a fixed, specified size from it. Return the substrings as
        sequences of integers rather than characters.

        :param text: a string containing lyrics
        :param seq_length: fixed size of each output sequence
        :return: a list of substrings extracted from the song, as lists of ints
        """

        # convert the string to a list of integers:
        text_as_chars = tf.strings.bytes_split(text)
        text_as_int = tf.map_fn(
            fn=lambda c: self.char2idx.lookup(c),
            elems=text_as_chars,
            dtype=tf.int32)
        text_as_int = tf.boolean_mask(text_as_int, text_as_int > -1)

        # compute the number of characters in the text:
        text_size = tf.size(text_as_int)

        # increase the sequence length by 1, for character-level prediction:
        seq_length += 1

        # create subsequences from the original sequence:
        trail  = tf.truncatemod(text_size, seq_length)
        n_seqs = tf.truncatediv(text_size, seq_length)
        to_keep = text_size - trail
        sequences = tf.reshape(text_as_int[:to_keep], [n_seqs, seq_length])

        # shuffle the substrings:
        sequences = tf.random.shuffle(sequences)

        return sequences
